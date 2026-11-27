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
import logging
import math
import os
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from qdrant_client.http.exceptions import ApiException as QdrantApiError

from ragio import __version__, clips, db, index, rag

log = logging.getLogger("ragio.app")

# What "a dependency is down" looks like: refused or timed-out sockets (OSError covers
# ConnectionError and TimeoutError), httpx transport/status errors from Ollama and Qdrant's
# REST client, and Qdrant's own API errors. Anything else is a bug and stays a 500.
DEPENDENCY_ERRORS: tuple[type[BaseException], ...] = (OSError, httpx.HTTPError, QdrantApiError)
try:  # llama-index-llms-ollama talks through the ollama client, which has its own error type
    from ollama import ResponseError as OllamaResponseError
    DEPENDENCY_ERRORS += (OllamaResponseError,)
except ImportError:  # pragma: no cover - ollama ships with llama-index-llms-ollama
    pass

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


def _unavailable(exc: Exception) -> HTTPException:
    """A 503 whose detail names the dependency error, e.g. `ConnectionError: … refused`."""
    log.warning("dependency unavailable: %s", _probe_error(exc))
    return HTTPException(503, f"a dependency is unavailable: {_probe_error(exc)}")


@app.get("/api/search")
def search(q: str = Query(..., min_length=2)):
    try:
        return rag.search(q)
    except DEPENDENCY_ERRORS as exc:
        raise _unavailable(exc) from exc


@app.post("/api/ask")
def ask(req: AskRequest):
    try:
        return rag.ask(req.question)
    except DEPENDENCY_ERRORS as exc:
        raise _unavailable(exc) from exc


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


@app.post("/api/ask/stream")
def ask_stream(req: AskRequest):
    # The 200 and headers are sent before rag.ask_stream runs, so a failure here cannot become
    # a status code: it becomes an `error` event, and the stream always ends with `done`.
    def gen():
        try:
            for event in rag.ask_stream(req.question):
                yield _sse(event)
                if event.get("type") == "done":
                    return
        except Exception as exc:  # the stream is already open: report every failure in-band
            log.warning("ask stream failed: %s", _probe_error(exc), exc_info=not isinstance(exc, DEPENDENCY_ERRORS))
            yield _sse({"type": "error", "message": _probe_error(exc)})
        yield _sse({"type": "done"})
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
