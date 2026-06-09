"""
Foundry IQ integration for BRIEF.

Calls the Azure AI Search agentic-retrieval (knowledge base) endpoint to ground
Assumption Archaeology in real, cited web sources via Grounding with Bing.

The knowledge base 'brief-knowledge-base' was created in the Microsoft Foundry
portal with a Web knowledge source, gpt-4.1-mini as the reasoning model, and
medium retrieval reasoning effort.

Everything here fails soft: if retrieval is unavailable for any reason, callers
get an empty result and can fall back to model-only analysis, so a live demo
never breaks.
"""

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT", "").rstrip("/")
SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY", "")
KNOWLEDGE_BASE = os.getenv("FOUNDRY_KNOWLEDGE_BASE", "brief-knowledge-base")
API_VERSION = "2025-11-01-preview"

# Retrieval timeout in seconds. Web-grounded agentic retrieval can take a while.
RETRIEVE_TIMEOUT = 60


def is_configured():
    """True only when all the credentials needed for a real call are present."""
    ok = bool(SEARCH_ENDPOINT and SEARCH_KEY and KNOWLEDGE_BASE)
    if not ok:
        print(f"[FoundryIQ] Not configured. endpoint={bool(SEARCH_ENDPOINT)} key={bool(SEARCH_KEY)} kb={bool(KNOWLEDGE_BASE)}")
    return ok


def retrieve(query, max_subqueries=4):
    """
    Run an agentic-retrieval query against the Foundry IQ knowledge base.

    Returns a dict:
      {
        "grounded": True/False,          # did we get real grounded results?
        "answer": "synthesised text or extractive content",
        "citations": [ {"title": ..., "url": ..., "snippet": ...}, ... ]
      }
    On any failure returns {"grounded": False, "answer": "", "citations": []}
    so the caller can fall back cleanly.
    """
    empty = {"grounded": False, "answer": "", "citations": []}

    if not is_configured():
        return empty

    url = f"{SEARCH_ENDPOINT}/knowledgeBases/{KNOWLEDGE_BASE}/retrieve?api-version={API_VERSION}"
    headers = {
        "Content-Type": "application/json",
        "api-key": SEARCH_KEY,
    }
    body = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": query}],
            }
        ],
        "knowledgeSourceParams": [
            {
                "knowledgeSourceName": "brief-web-sources",
                "kind": "web",
            }
        ],
        "outputMode": "answerSynthesis",
    }

    try:
        resp = requests.post(url, headers=headers, json=body, timeout=RETRIEVE_TIMEOUT)
        if resp.status_code != 200:
            print(f"[FoundryIQ] Retrieve failed: HTTP {resp.status_code}")
            print(f"[FoundryIQ] URL: {url}")
            print(f"[FoundryIQ] Response: {resp.text[:600]}")
            return empty
        data = resp.json()
        print(f"[FoundryIQ] Retrieve OK (200). Keys: {list(data.keys())}")
    except Exception as e:
        print(f"[FoundryIQ] Exception: {type(e).__name__}: {e}")
        return empty

    # The response shape: 'response' carries synthesized/extractive content,
    # 'references' carries the source citations.
    answer_text = ""
    try:
        response_blocks = data.get("response", [])
        parts = []
        for block in response_blocks:
            for c in block.get("content", []):
                if c.get("type") == "text" and c.get("text"):
                    parts.append(c["text"])
        answer_text = "\n".join(parts).strip()
    except Exception:
        answer_text = ""

    citations = []
    try:
        refs = data.get("references", [])
        print(f"[FoundryIQ] {len(refs)} references returned. Sample: {json.dumps(refs[0])[:400] if refs else 'none'}")

        def first(*vals):
            for v in vals:
                if v:
                    return v
            return ""

        for ref in refs:
            sd = ref.get("sourceData", {}) or {}
            # Web references often nest the useful fields under sourceData,
            # and some shapes put url/title at the top level. Try everything.
            title = first(
                ref.get("title"), sd.get("title"), sd.get("name"),
                ref.get("name"), sd.get("pageTitle")
            ) or "Source"
            url = first(
                ref.get("url"), sd.get("url"), sd.get("link"),
                sd.get("sourceUrl"), ref.get("docKey")
            )
            snippet = first(
                sd.get("content"), sd.get("snippet"), sd.get("text"),
                ref.get("content"), sd.get("description")
            )
            # Skip empty references with no url and no title worth showing
            if not url and title == "Source" and not snippet:
                continue
            citations.append({
                "title": title,
                "url": url,
                "snippet": str(snippet)[:280],
            })

        # De-duplicate by url
        seen = set()
        deduped = []
        for c in citations:
            key = c["url"] or c["title"]
            if key in seen:
                continue
            seen.add(key)
            deduped.append(c)
        citations = deduped
        print(f"[FoundryIQ] Parsed {len(citations)} usable citations.")
    except Exception as e:
        print(f"[FoundryIQ] reference parse error: {e}")
        citations = []

    grounded = bool(answer_text or citations)
    return {"grounded": grounded, "answer": answer_text, "citations": citations}