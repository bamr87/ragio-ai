"""Ragio-Lab — single-file FastAPI wrapper.

    uvicorn app:app --reload --port 8000

Endpoints
  GET  /                      minimal UI (static/index.html)
  GET  /healthz               liveness: {"status":"ok"}; touches neither SQLite nor Qdrant
  GET  /readyz                readiness: SQLite + Qdrant checks → 200 "ok" or 503 "degraded"
  GET  /version               {"name","version","commit","built_at"}: release-please version, $GIT_SHA, $BUILT_AT
  GET  /api/episodes          catalog with pipeline status
  GET  /api/episodes/{id}     one episode + its windows (a readable transcript)
  GET  /api/search?q=         retrieval only (grouped by episode, timestamps)
  POST /api/ask               {"question": ...} → answer + sources (JSON)
  POST /api/ask/stream        same, as Server-Sent Events (sources first, then tokens)
  GET  /api/clip?ep&start&end MP3 clip cut from the episode (cached)
  POST /api/listen            {"episode_id","start","end","prompt"} → audio-native model answer
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from ragio import __version__, clips, db, index, rag

app = FastAPI(title="Ragio-Lab", version=__version__)

# Probes (fleet standard UPS-BE-10, UPS-BE-11) are unauthenticated and never cached.
NO_STORE = {"Cache-Control": "no-store"}
QDRANT_PROBE_TIMEOUT = 1  # seconds: /readyz answers well inside 2 s when Qdrant is down


class AskRequest(BaseModel):
    question: str


class ListenRequest(BaseModel):
    episode_id: str
    start: float
    end: float
    prompt: str = "Describe what you hear: who is speaking, the tone, and any music or sound design."


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    # Resolve next to this file so `uvicorn app:app` works from any working directory.
    return (Path(__file__).resolve().parent / "static" / "index.html").read_text()


def _probe_error(exc: Exception) -> str:
    """One short line saying why a readiness check failed."""
    lines = str(exc).strip().splitlines()
    reason = f"{type(exc).__name__}: {lines[0]}" if lines else type(exc).__name__
    return reason[:200]


def _check_db() -> str:
    try:
        with db.tx() as conn:
            conn.execute("SELECT 1")
    except Exception as exc:  # any failure means not ready; the reason goes into the response
        return _probe_error(exc)
    return "ok"


def _check_qdrant() -> str:
    try:
        client = index.qdrant(timeout=QDRANT_PROBE_TIMEOUT)
        try:
            client.get_collections()
        finally:
            client.close()
    except Exception as exc:  # any failure means not ready; the reason goes into the response
        return _probe_error(exc)
    return "ok"


@app.get("/healthz")
def healthz() -> JSONResponse:
    """Liveness: the process answers. Touches no dependency."""
    return JSONResponse({"status": "ok"}, headers=NO_STORE)


@app.get("/readyz")
def readyz() -> JSONResponse:
    """Readiness: SQLite answers SELECT 1 and Qdrant lists its collections. Ollama is not checked."""
    checks = {"db": _check_db(), "qdrant": _check_qdrant()}
    ready = all(result == "ok" for result in checks.values())
    return JSONResponse({"status": "ok" if ready else "degraded", "checks": checks},
                        status_code=200 if ready else 503, headers=NO_STORE)


@app.get("/version")
def version() -> JSONResponse:
    """Build identity: the release-please version, plus the build's commit and time (null when unset)."""
    return JSONResponse({
        "name": "ragio-lab",
        "version": __version__,
        "commit": os.environ.get("GIT_SHA") or None,
        "built_at": os.environ.get("BUILT_AT") or None,
    }, headers=NO_STORE)


@app.get("/api/episodes")
def episodes():
    with db.tx() as conn:
        rows = conn.execute(
            "SELECT id, title, published, duration_sec, page_url, status FROM episodes ORDER BY published DESC"
        ).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/episodes/{episode_id}")
def episode(episode_id: str):
    with db.tx() as conn:
        ep = db.get_episode(conn, episode_id)
        if not ep:
            raise HTTPException(404, "unknown episode")
        wins = db.get_windows(conn, episode_id, 0, 10**9)
    return {**{k: ep[k] for k in ep.keys() if k != "audio_path"},
            "windows": [{**dict(w), "start_hms": rag.fmt_ts(w["start"])} for w in wins]}


@app.get("/api/search")
def search(q: str = Query(..., min_length=2)):
    return rag.search(q)


@app.post("/api/ask")
def ask(req: AskRequest):
    return rag.ask(req.question)


@app.post("/api/ask/stream")
def ask_stream(req: AskRequest):
    def gen():
        for event in rag.ask_stream(req.question):
            yield f"data: {json.dumps(event)}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/clip")
def clip(ep: str, start: float, end: float):
    if end <= start or end - start > 600:
        raise HTTPException(400, "clip must be 0–600 seconds long")
    try:
        path = clips.cut_clip(ep, start, end)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    return FileResponse(path, media_type="audio/mpeg", filename=path.name)


@app.post("/api/listen")
def listen(req: ListenRequest):
    if req.end <= req.start or req.end - req.start > 180:
        raise HTTPException(400, "deep-listen clips must be 0–180 seconds long")
    try:
        wav = clips.cut_clip(req.episode_id, req.start, req.end, fmt="wav")
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    try:
        return {"answer": clips.listen(wav, req.prompt)}
    except NotImplementedError as exc:
        raise HTTPException(501, str(exc)) from exc
