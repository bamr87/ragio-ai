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
    from pathlib import Path
    return Path("static/index.html").read_text()


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
    if req.end - req.start > 180:
        raise HTTPException(400, "keep deep-listen clips under 3 minutes")
    wav = clips.cut_clip(req.episode_id, req.start, req.end, fmt="wav")
    try:
        return {"answer": clips.listen(wav, req.prompt)}
    except NotImplementedError as exc:
        raise HTTPException(501, str(exc)) from exc
