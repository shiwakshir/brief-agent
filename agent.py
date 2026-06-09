import os
import json
import asyncio
from dotenv import load_dotenv
from openai import AzureOpenAI
import foundry_iq

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-12-01-preview"
)

MODEL = os.getenv("AZURE_MODEL_DEPLOYMENT")


def call_model(system_prompt, user_message, temperature=0.3):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=temperature
    )
    return response.choices[0].message.content


def safe_json(raw):
    """Strip markdown fences and parse JSON safely."""
    try:
        return json.loads(raw)
    except Exception:
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(cleaned)
        except Exception:
            # Last resort: return as string wrapped in dict
            return {"raw": cleaned}


# ── STEP 1: PARSE ──────────────────────────────────────────────────────────────
def step_parse(brief):
    system = """You are a senior research methodology expert.
Extract structured information from a research brief.
Return ONLY valid JSON with exactly these fields:
{
  "core_question": "the main research question in one sentence",
  "category": "the market or topic category (e.g. health drinks, financial services)",
  "target_audience": "who the research is about, including demographics",
  "geography": "markets or regions mentioned, or 'Global' if not specified",
  "research_objective": "what the researcher wants to discover or validate",
  "client_hypotheses": ["list", "of", "stated", "assumptions", "or", "hypotheses"],
  "methodology_hints": "any methodology mentioned (qual/quant/survey/etc) or 'Not specified'"
}"""

    raw = call_model(system, f"Parse this research brief:\n\n{brief}")
    return safe_json(raw)


# ── STEP 2: GENERATE PROMPTS ───────────────────────────────────────────────────
def step_generate(parsed):
    system = """You are an expert in consumer search behaviour and information-seeking patterns.
Generate exactly 6 varied prompts that real people - not researchers - would type or ask
about this topic. Make them natural, conversational, and semantically diverse.
Vary the framing: some curious, some sceptical, some seeking validation, some practical.
Return ONLY a valid JSON array of exactly 6 strings. No other text."""

    user = f"""Generate 6 natural consumer-style prompts for:
Category: {parsed.get('category', '')}
Core question: {parsed.get('core_question', '')}
Target audience: {parsed.get('target_audience', '')}
Geography: {parsed.get('geography', '')}"""

    raw = call_model(system, user)
    return safe_json(raw)


# ── STEP 3: QUERY (with persona variants) ─────────────────────────────────────
def step_query(prompts, parsed):
    """Query the model for each prompt AND for 3 persona variants to detect bias."""
    
    base_system = """You are a helpful assistant. Answer the question naturally and informatively."""
    
    responses = []
    for i, prompt in enumerate(prompts):
        response = call_model(base_system, prompt, temperature=0.5)
        responses.append({"prompt": prompt, "response": response})

    # Persona variant queries - same core question, different cultural framing
    geography = parsed.get('geography', 'Global')
    category = parsed.get('category', 'this topic')
    core_q = parsed.get('core_question', '')

    personas = [
        {"label": "Western English-speaker", "framing": f"As someone in the US or UK, {core_q}"},
        {"label": "Global South perspective", "framing": f"As someone in a developing economy, what should I know about {category}?"},
        {"label": "Sceptical consumer", "framing": f"I don't trust what brands say about {category}. What's the real story?"},
        {"label": "First-time researcher", "framing": f"I'm new to {category} and want to understand it properly. Where do I start?"}
    ]

    persona_responses = []
    for p in personas:
        resp = call_model(base_system, p["framing"], temperature=0.5)
        persona_responses.append({
            "persona": p["label"],
            "prompt": p["framing"],
            "response": resp
        })

    return {"base_responses": responses, "persona_responses": persona_responses}


# ── STEP 4: CLUSTER ────────────────────────────────────────────────────────────
def step_cluster(query_data):
    system = """You are an expert in discourse analysis, AI bias detection, and semiotics.
Analyse these AI responses across base queries and persona variants.
Identify dominant assumption themes across ALL responses.
Return ONLY valid JSON with this exact structure:
{
  "dominant_assumptions": [
    {
      "theme": "short name for assumption",
      "frequency": "X/10 responses reflect this",
      "description": "what AI assumes here in one sentence",
      "example_quote": "a short illustrative phrase from the responses"
    }
  ],
  "dominant_perspective": "one sentence: whose worldview dominates these responses",
  "geographic_bias": "one sentence: any geographic or cultural slant detected",
  "language_register": "one sentence: what level of literacy/expertise does AI assume in the reader",
  "persona_divergence": "one sentence: how much do responses change across personas - high/medium/low with explanation"
}"""

    all_responses = ""
    for r in query_data["base_responses"]:
        all_responses += f"\nPROMPT: {r['prompt']}\nRESPONSE: {r['response']}\n---"
    for r in query_data["persona_responses"]:
        all_responses += f"\nPERSONA: {r['persona']}\nPROMPT: {r['prompt']}\nRESPONSE: {r['response']}\n---"

    raw = call_model(system, f"Analyse these AI responses:\n{all_responses}")
    return safe_json(raw)


# ── STEP 5: GAP ANALYSIS ───────────────────────────────────────────────────────
def step_gap_analysis(parsed, clusters):
    system = """You are a senior market research strategist with deep expertise in research design.
Compare what AI assumes about a topic versus what the actual research audience needs.
Return ONLY valid JSON with this exact structure:
{
  "overrepresented": [
    {"perspective": "name", "explanation": "why this is over-emphasised by AI"}
  ],
  "underrepresented": [
    {"perspective": "name", "explanation": "what AI misses or ignores"}
  ],
  "audience_mismatch": "one paragraph on how AI worldview differs from the target audience reality",
  "risk_to_research": "one paragraph on what bad research design decisions could follow from accepting AI assumptions",
  "unknown_unknowns": [
    "Question AI and the brief both missed but that likely matters to the target audience",
    "Another such question",
    "A third such question"
  ]
}"""

    user = f"""
Research target audience: {parsed.get('target_audience', '')}
Research geography: {parsed.get('geography', '')}
Research objective: {parsed.get('research_objective', '')}

AI dominant assumptions: {json.dumps(clusters.get('dominant_assumptions', []))}
AI dominant perspective: {clusters.get('dominant_perspective', '')}
AI geographic bias: {clusters.get('geographic_bias', '')}
AI persona divergence: {clusters.get('persona_divergence', '')}

What are the critical gaps between AI worldview and research reality?"""

    raw = call_model(system, user)
    return safe_json(raw)


# ── STEP 6A: TEMPORAL DRIFT DETECTION ─────────────────────────────────────────
def step_temporal_drift(parsed, clusters):
    system = """You are an expert in AI training data analysis and market trend forecasting.
Assess how likely the AI's assumptions about this topic are to be temporally stale -
i.e. reflecting the world as it was 2-3 years ago rather than today.
Return ONLY valid JSON with this exact structure:
{
  "overall_drift_risk": "Low / Medium / High",
  "drift_explanation": "one paragraph explaining why this topic is or isn't prone to AI temporal drift",
  "stale_assumptions": [
    {
      "assumption": "the AI assumption that may be outdated",
      "why_stale": "what has likely changed since AI training",
      "research_implication": "what to probe in fieldwork to get current reality"
    }
  ],
  "fast_moving_dimensions": ["list of sub-topics within this category that change fastest"],
  "recommendation": "one actionable sentence for the researcher"
}"""

    user = f"""
Category: {parsed.get('category', '')}
Geography: {parsed.get('geography', '')}
AI dominant assumptions: {json.dumps(clusters.get('dominant_assumptions', []))}
AI dominant perspective: {clusters.get('dominant_perspective', '')}

Assess temporal drift risk for these AI assumptions."""

    raw = call_model(system, user)
    return safe_json(raw)


# ── STEP 6B: HYPOTHESIS CONTAMINATION SCORE ───────────────────────────────────
def step_hypothesis_contamination(parsed, clusters):
    system = """You are an expert in research epistemology and AI-generated content analysis.
Assess whether the client's stated hypotheses are genuinely original insights
or whether they are simply reflecting AI consensus - i.e. things the client
learned from AI-generated content rather than real market intelligence.
Return ONLY valid JSON with this exact structure:
{
  "hypotheses_assessed": [
    {
      "hypothesis": "the client's stated hypothesis",
      "contamination_score": 0,
      "score_label": "Low / Medium / High / Critical",
      "explanation": "why this hypothesis does or doesn't match AI consensus",
      "recommendation": "how to treat this hypothesis in research design"
    }
  ],
  "overall_contamination_level": "Low / Medium / High / Critical",
  "overall_explanation": "one paragraph summarising the contamination picture",
  "genuinely_original_hypotheses": ["any hypotheses that appear NOT to be AI-derived"],
  "most_dangerous_assumption": "the single hypothesis most likely to corrupt research findings if left unchallenged"
}"""

    hypotheses = parsed.get('client_hypotheses', [])
    if not hypotheses:
        hypotheses = ["No explicit hypotheses stated - inferred from brief language"]

    user = f"""
Client hypotheses: {json.dumps(hypotheses)}
Research category: {parsed.get('category', '')}
AI dominant assumptions for this category: {json.dumps(clusters.get('dominant_assumptions', []))}
AI dominant perspective: {clusters.get('dominant_perspective', '')}

Score each hypothesis for AI contamination (0=completely original, 100=pure AI consensus)."""

    raw = call_model(system, user)
    result = safe_json(raw)

    # Ensure scores are integers
    if "hypotheses_assessed" in result:
        for h in result["hypotheses_assessed"]:
            try:
                h["contamination_score"] = int(h["contamination_score"])
            except Exception:
                h["contamination_score"] = 50
    return result


# ── STEP 6C: COMPETITOR INTELLIGENCE ──────────────────────────────────────────
def step_competitor_intelligence(query_data, parsed):
    system = """You are an expert in brand intelligence and competitive analysis.
Analyse these AI responses and identify which brands, products, companies,
or solutions appear organically - without being prompted.
These are the brands that dominate AI's answers about this category,
which means participants have likely already been exposed to them
before they reach your focus group or survey.
Return ONLY valid JSON with this exact structure:
{
  "brands_mentioned": [
    {
      "brand": "brand or product name",
      "frequency": "how many responses mention it",
      "framing": "how AI frames this brand (positive/neutral/negative/as default)",
      "priming_risk": "Low / Medium / High - risk that respondents are pre-primed"
    }
  ],
  "category_leader_in_ai": "which brand or type of solution AI positions as default/best",
  "invisible_competitors": "one sentence on what types of solutions AI never mentions",
  "discussion_guide_implication": "one actionable paragraph on how to handle competitive priming in fieldwork"
}"""

    all_text = "\n---\n".join([
        f"PROMPT: {r['prompt']}\nRESPONSE: {r['response']}"
        for r in query_data["base_responses"] + query_data["persona_responses"]
    ])

    user = f"""Category: {parsed.get('category', '')}
Geography: {parsed.get('geography', '')}

Analyse these AI responses for organic brand/competitor mentions:
{all_text}"""

    raw = call_model(system, user)
    return safe_json(raw)


# ── STEP 6D: ASSUMPTION ARCHAEOLOGY (grounded by Foundry IQ) ───────────────────
def step_assumption_archaeology(parsed, clusters, gaps):
    # First, ground the analysis in real cited web sources via Foundry IQ.
    category = parsed.get('category', 'this topic')
    geography = parsed.get('geography', 'Global')
    retrieval_query = (
        f"What types of sources, reports, publications and institutional voices "
        f"shape how the topic of '{category}' is discussed and understood, "
        f"particularly in {geography}? Identify dominant industry, media and "
        f"academic sources and whose perspective they represent."
    )

    iq = foundry_iq.retrieve(retrieval_query)

    grounding_block = ""
    if iq.get("grounded"):
        cite_lines = []
        for c in iq.get("citations", [])[:8]:
            cite_lines.append(f"- {c.get('title','Source')} ({c.get('url','')}): {c.get('snippet','')}")
        grounding_block = (
            "\n\nGROUNDED EVIDENCE FROM FOUNDRY IQ (real web sources, use these to "
            "anchor your analysis and reference the source types you actually see):\n"
            + (iq.get("answer", "") or "")
            + "\n\nCited sources:\n" + "\n".join(cite_lines)
        )

    system = """You are an expert in the sociology of knowledge, media studies, and source analysis.
Work out which kinds of sources shaped how AI and the wider information landscape
understand this topic. Where grounded evidence from real web sources is provided,
base your source landscape on what those sources actually show.
Return ONLY valid JSON with this exact structure:
{
  "source_landscape": [
    {
      "source_type": "type of source (e.g. Western industry reports, mainstream news, academic papers)",
      "influence_level": "Dominant / Significant / Marginal / Absent",
      "what_it_contributes": "what assumptions or framings this source type introduces",
      "whose_voice": "whose perspective or interest this source type typically represents"
    }
  ],
  "absent_voices": [
    "type of source or perspective absent from the dominant understanding of this topic"
  ],
  "dominant_narrative_origin": "one paragraph: the single most influential source type and why it dominates",
  "implication_for_research": "one paragraph: what this means for how you design and interpret your research",
  "decolonisation_note": "one sentence: if applicable, how Western or Anglo-centric the framing of this topic is",
  "grounded": false,
  "sources": []
}"""

    user = f"""
Category: {parsed.get('category', '')}
Geography: {parsed.get('geography', '')}
Target audience: {parsed.get('target_audience', '')}
AI dominant assumptions: {json.dumps(clusters.get('dominant_assumptions', []))}
AI dominant perspective: {clusters.get('dominant_perspective', '')}
Underrepresented perspectives: {json.dumps(gaps.get('underrepresented', []))}
{grounding_block}

Trace where the assumptions about this topic come from."""

    raw = call_model(system, user)
    result = safe_json(raw)

    # Attach the real Foundry IQ provenance so the UI can show citations.
    result["grounded"] = bool(iq.get("grounded"))
    result["sources"] = iq.get("citations", []) if iq.get("grounded") else []
    return result




# ── STEP 6E: QUAL vs QUANT ROUTING ────────────────────────────────────────────
def step_qual_quant_routing(parsed, gaps, temporal_drift):
    system = """You are a senior research methodology director with 20 years of experience
in both qualitative and quantitative research design.
Based on the nature of the gaps identified, recommend the optimal methodology mix.
Return ONLY valid JSON with this exact structure:
{
  "recommended_approach": "Qualitative / Quantitative / Mixed Methods - with brief rationale",
  "methodology_breakdown": [
    {
      "dimension": "the research dimension or question",
      "recommended_method": "specific method (e.g. in-depth interviews, laddering, projective techniques, conjoint, segmentation survey)",
      "rationale": "why this method for this dimension",
      "ai_bias_risk": "how AI assumptions could corrupt this dimension if wrong method chosen"
    }
  ],
  "critical_qual_dimensions": ["dimensions that CANNOT be captured by survey - require qual"],
  "projective_techniques_needed": true,
  "projective_rationale": "why or why not projective techniques are needed",
  "sample_design_notes": "one paragraph on sample composition to counteract AI perspective bias",
  "stimulus_material_warning": "one sentence on any stimulus materials that may carry AI-contaminated framing"
}"""

    user = f"""
Research objective: {parsed.get('research_objective', '')}
Target audience: {parsed.get('target_audience', '')}
Geography: {parsed.get('geography', '')}
Methodology hints from brief: {parsed.get('methodology_hints', 'Not specified')}

Underrepresented perspectives: {json.dumps(gaps.get('underrepresented', []))}
Unknown unknowns: {json.dumps(gaps.get('unknown_unknowns', []))}
Audience mismatch: {gaps.get('audience_mismatch', '')}
Temporal drift risk: {temporal_drift.get('overall_drift_risk', 'Unknown')}

Recommend the methodology mix most likely to surface findings AI would not have predicted."""

    raw = call_model(system, user)
    return safe_json(raw)


# ── STEP 11: CONFIDENCE SYNTHESIS ─────────────────────────────────────────────
def step_confidence(parsed, clusters, gaps, temporal_drift, contamination, methodology):
    """Final synthesis: a single research-design confidence score with reasoning."""
    system = """You are the lead reviewer signing off on a research design.
Based on all the analysis, produce a single Research Design Confidence Score (0-100):
how likely is this research, as currently framed, to tell the client something new
rather than confirm what AI would already have said?
Lower scores mean the brief is heavily contaminated and needs rework before fieldwork.
Higher scores mean the design is well-positioned to discover real insight.
Return ONLY valid JSON with this exact structure:
{
  "confidence_score": 0,
  "confidence_label": "Strong / Adequate / Fragile / Compromised",
  "headline": "one plain, direct sentence a researcher could say to their client",
  "score_rationale": "one short paragraph explaining the score",
  "top_three_risks": [
    "the single most important thing to fix before fieldwork",
    "the second",
    "the third"
  ],
  "what_would_raise_it": "one sentence on what change would most improve the score"
}"""

    user = f"""
Research objective: {parsed.get('research_objective', '')}
Target audience: {parsed.get('target_audience', '')}

Overall hypothesis contamination: {contamination.get('overall_contamination_level', 'Unknown')}
Most dangerous assumption: {contamination.get('most_dangerous_assumption', 'None identified')}
Temporal drift risk: {temporal_drift.get('overall_drift_risk', 'Unknown')}
Audience mismatch: {gaps.get('audience_mismatch', '')}
Number of blind spots found: {len(gaps.get('underrepresented', []))}
Recommended approach: {methodology.get('recommended_approach', '')}

Score this research design's confidence (0=will only confirm what AI assumes, 100=well-positioned to find something new)."""

    raw = call_model(system, user)
    result = safe_json(raw)
    if "confidence_score" in result:
        try:
            result["confidence_score"] = int(result["confidence_score"])
        except Exception:
            result["confidence_score"] = 50
    return result


# ── MASTER PIPELINE ────────────────────────────────────────────────────────────
def _safe_step(fn, fallback, *args, **kwargs):
    """Run a pipeline step with error recovery so one failure can't kill the run."""
    try:
        result = fn(*args, **kwargs)
        if result is None:
            return fallback
        return result
    except Exception as e:
        fb = dict(fallback) if isinstance(fallback, dict) else fallback
        if isinstance(fb, dict):
            fb["_error"] = str(e)
        return fb


def run_brief(brief, progress_callback=None):
    """
    Run the full BRIEF pipeline with per-step error recovery.
    progress_callback(step_name, step_number, total_steps) called at each step.
    Returns a dict of all results plus a confidence synthesis.
    """
    total = 11

    def progress(name, n):
        if progress_callback:
            progress_callback(name, n, total)

    progress("Reading your brief", 1)
    parsed = _safe_step(step_parse, {
        "core_question": "Could not parse", "category": "Unknown",
        "target_audience": "Unknown", "geography": "Global",
        "research_objective": "Unknown", "client_hypotheses": [],
        "methodology_hints": "Not specified"
    }, brief)

    progress("Working out how people actually ask about this", 2)
    prompts = _safe_step(step_generate, [], parsed)
    if not isinstance(prompts, list):
        raw_val = prompts.get("raw", "") if isinstance(prompts, dict) else ""
        prompts = [p.strip() for p in raw_val.split("\n") if p.strip()][:6]
    if not prompts:
        prompts = [parsed.get("core_question", "Tell me about this topic")]

    progress("Asking AI the same question 10 different ways", 3)
    query_data = _safe_step(step_query, {"base_responses": [], "persona_responses": []}, prompts, parsed)

    progress("Finding the patterns in what AI says", 4)
    clusters = _safe_step(step_cluster, {
        "dominant_assumptions": [], "dominant_perspective": "Analysis unavailable",
        "geographic_bias": "Unknown", "language_register": "Unknown",
        "persona_divergence": "Unknown"
    }, query_data)

    progress("Comparing AI's picture to your actual audience", 5)
    gaps = _safe_step(step_gap_analysis, {
        "overrepresented": [], "underrepresented": [],
        "audience_mismatch": "Analysis unavailable", "risk_to_research": "",
        "unknown_unknowns": []
    }, parsed, clusters)

    progress("Checking how much of this is already out of date", 6)
    temporal_drift = _safe_step(step_temporal_drift, {
        "overall_drift_risk": "Unknown", "drift_explanation": "Analysis unavailable",
        "stale_assumptions": [], "fast_moving_dimensions": [], "recommendation": ""
    }, parsed, clusters)

    progress("Scoring the client's assumptions against AI consensus", 7)
    contamination = _safe_step(step_hypothesis_contamination, {
        "hypotheses_assessed": [], "overall_contamination_level": "Unknown",
        "overall_explanation": "Analysis unavailable",
        "genuinely_original_hypotheses": [], "most_dangerous_assumption": ""
    }, parsed, clusters)

    progress("Identifying which brands AI puts in the room", 8)
    competitor_intel = _safe_step(step_competitor_intelligence, {
        "brands_mentioned": [], "category_leader_in_ai": "Analysis unavailable",
        "invisible_competitors": "", "discussion_guide_implication": ""
    }, query_data, parsed)

    progress("Tracing where AI's assumptions come from", 9)
    archaeology = _safe_step(step_assumption_archaeology, {
        "source_landscape": [], "absent_voices": [],
        "dominant_narrative_origin": "Analysis unavailable",
        "implication_for_research": "", "decolonisation_note": "",
        "grounded": False, "sources": []
    }, parsed, clusters, gaps)

    progress("Working out how to actually research this properly", 10)
    methodology = _safe_step(step_qual_quant_routing, {
        "recommended_approach": "Analysis unavailable", "methodology_breakdown": [],
        "critical_qual_dimensions": [], "projective_techniques_needed": False,
        "projective_rationale": "", "sample_design_notes": "",
        "stimulus_material_warning": ""
    }, parsed, gaps, temporal_drift)

    progress("Scoring overall research design confidence", 11)
    confidence = _safe_step(step_confidence, {
        "confidence_score": 50, "confidence_label": "Adequate",
        "headline": "Analysis complete.", "score_rationale": "",
        "top_three_risks": [], "what_would_raise_it": ""
    }, parsed, clusters, gaps, temporal_drift, contamination, methodology)

    return {
        "parsed": parsed,
        "prompts": prompts,
        "query_data": query_data,
        "clusters": clusters,
        "gaps": gaps,
        "temporal_drift": temporal_drift,
        "contamination": contamination,
        "competitor_intel": competitor_intel,
        "archaeology": archaeology,
        "methodology": methodology,
        "confidence": confidence
    }