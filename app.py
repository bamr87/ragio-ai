"""Ragio-Lab — single-file FastAPI wrapper.

    uvicorn app:app --reload --port 8000

Endpoints
  GET  /                      minimal UI (static/index.html)
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
import math
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

from ragio import clips, db, rag

app = FastAPI(title="Ragio-Lab", version="0.1.0")


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


def _check_window(start: float, end: float, max_seconds: int, what: str) -> None:
    """400 unless start and end are finite, start >= 0 and the window is 0–max_seconds long."""
    # NaN fails every comparison, so it would pass the length check below on its own.
    if not (math.isfinite(start) and math.isfinite(end)) or start < 0:
        raise HTTPException(400, "start and end must be finite numbers of seconds, start >= 0")
    if end <= start or end - start > max_seconds:
        raise HTTPException(400, f"{what} must be 0–{max_seconds} seconds long")


def _cut(episode_id: str, start: float, end: float, fmt: str = "mp3") -> Path:
    """clips.cut_clip, with its failures as HTTP errors: 404 no audio, 503 no ffmpeg, 502 ffmpeg failed."""
    try:
        return clips.cut_clip(episode_id, start, end, fmt=fmt)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except clips.FfmpegMissing as exc:
        raise HTTPException(503, str(exc)) from exc
    except clips.ClipError as exc:
        raise HTTPException(502, str(exc)) from exc


@app.get("/api/clip")
def clip(ep: str, start: float, end: float):
    _check_window(start, end, 600, "clip")
    path = _cut(ep, start, end)
    return FileResponse(path, media_type="audio/mpeg", filename=path.name)


@app.post("/api/listen")
def listen(req: ListenRequest):
    _check_window(req.start, req.end, 180, "deep-listen clips")
    wav = _cut(req.episode_id, req.start, req.end, fmt="wav")
    try:
        return {"answer": clips.listen(wav, req.prompt)}
    except NotImplementedError as exc:
        raise HTTPException(501, str(exc)) from exc
