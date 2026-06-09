import os
import json
import time
import threading
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from agent import run_brief

app = Flask(__name__)

# In-memory stores keyed by session id, with timestamps for cleanup
results_store = {}
progress_store = {}
session_times = {}

SESSION_TTL = 1800          # 30 minutes
MAX_BRIEF_LENGTH = 8000     # guard against runaway input
MIN_BRIEF_LENGTH = 50
RUN_TIMEOUT = 300           # 5 minute hard ceiling per analysis


def _cleanup_old_sessions():
    """Remove sessions older than the TTL so memory doesn't grow unbounded."""
    now = time.time()
    stale = [sid for sid, t in session_times.items() if now - t > SESSION_TTL]
    for sid in stale:
        results_store.pop(sid, None)
        progress_store.pop(sid, None)
        session_times.pop(sid, None)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyse", methods=["POST"])
def analyse():
    _cleanup_old_sessions()

    data = request.get_json(silent=True) or {}
    brief = (data.get("brief") or "").strip()

    # Input validation and sanitisation
    if not brief:
        return jsonify({"error": "Please paste a research brief to analyse."}), 400
    if len(brief) < MIN_BRIEF_LENGTH:
        return jsonify({"error": "That brief is too short. Add more detail about the audience, objective, and any client hypotheses."}), 400
    if len(brief) > MAX_BRIEF_LENGTH:
        brief = brief[:MAX_BRIEF_LENGTH]

    session_id = os.urandom(8).hex()
    progress_store[session_id] = []
    results_store[session_id] = None
    session_times[session_id] = time.time()

    def run_in_background():
        def on_progress(step_name, step_number, total_steps):
            progress_store[session_id].append({
                "step": step_name,
                "number": step_number,
                "total": total_steps,
                "pct": int((step_number / total_steps) * 100)
            })
        try:
            result = run_brief(brief, progress_callback=on_progress)
            results_store[session_id] = {"status": "done", "data": result}
        except Exception as e:
            results_store[session_id] = {"status": "error", "message": str(e)}

    thread = threading.Thread(target=run_in_background, daemon=True)
    thread.start()

    return jsonify({"session_id": session_id})


@app.route("/progress/<session_id>")
def progress(session_id):
    if session_id not in progress_store:
        return jsonify({"error": "Unknown or expired session."}), 404

    def generate():
        last_sent = 0
        start = time.time()
        while True:
            # Hard timeout guard
            if time.time() - start > RUN_TIMEOUT:
                yield f"data: {json.dumps({'type': 'result', 'status': 'error', 'message': 'Analysis timed out. Please try again with a shorter brief.'})}\n\n"
                break

            events = progress_store.get(session_id, [])
            result = results_store.get(session_id)

            while last_sent < len(events):
                evt = events[last_sent]
                yield f"data: {json.dumps({'type': 'progress', **evt})}\n\n"
                last_sent += 1

            if result is not None:
                yield f"data: {json.dumps({'type': 'result', **result})}\n\n"
                break

            time.sleep(0.5)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"
        }
    )


@app.errorhandler(500)
def handle_500(e):
    return jsonify({"error": "Something went wrong on the server. Please try again."}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)