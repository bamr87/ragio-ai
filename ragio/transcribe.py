"""Stage 2 — local Whisper transcription with exact timestamps.

Two interchangeable backends:
  * faster-whisper  — CTranslate2; best on Linux/NVIDIA, fine on CPU with int8
  * mlx             — mlx-whisper; fastest option on Apple Silicon (Metal)

Output: data/transcripts/<episode_id>.json  +  rows in the `segments` table.

    python -m ragio transcribe [--limit N] [--backend mlx]
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import config, db


def _transcribe_faster_whisper(path: Path) -> dict:
    from faster_whisper import WhisperModel  # lazy import: heavy

    model = _transcribe_faster_whisper.model  # type: ignore[attr-defined]
    if model is None:
        model = WhisperModel(config.WHISPER_MODEL, device="auto", compute_type=config.WHISPER_COMPUTE)
        _transcribe_faster_whisper.model = model  # type: ignore[attr-defined]
    seg_iter, info = model.transcribe(
        str(path),
        beam_size=5,
        vad_filter=True,               # skips music beds / silence → fewer hallucinations
        word_timestamps=True,
        condition_on_previous_text=False,  # prevents runaway repetition loops on long audio
    )
    segments = []
    for s in seg_iter:
        segments.append({
            "start": round(s.start, 2),
            "end": round(s.end, 2),
            "text": s.text.strip(),
            "words": [{"w": w.word, "s": round(w.start, 2), "e": round(w.end, 2)} for w in (s.words or [])],
        })
    return {"language": info.language, "duration": info.duration, "segments": segments,
            "model": f"faster-whisper/{config.WHISPER_MODEL}/{config.WHISPER_COMPUTE}"}


_transcribe_faster_whisper.model = None  # type: ignore[attr-defined]


def _transcribe_mlx(path: Path) -> dict:
    import mlx_whisper  # lazy import

    out = mlx_whisper.transcribe(
        str(path),
        path_or_hf_repo=config.WHISPER_MLX_REPO,
        word_timestamps=True,
        condition_on_previous_text=False,
    )
    segments = [{
        "start": round(s["start"], 2),
        "end": round(s["end"], 2),
        "text": s["text"].strip(),
        "words": [{"w": w["word"], "s": round(w["start"], 2), "e": round(w["end"], 2)} for w in s.get("words", [])],
    } for s in out["segments"]]
    duration = segments[-1]["end"] if segments else 0.0
    return {"language": out.get("language"), "duration": duration, "segments": segments,
            "model": f"mlx/{config.WHISPER_MLX_REPO}"}


BACKENDS = {"faster-whisper": _transcribe_faster_whisper, "mlx": _transcribe_mlx}


def transcribe_file(path: Path, backend: str = config.WHISPER_BACKEND) -> dict:
    return BACKENDS[backend](path)


def transcribe_pending(limit: int | None = None, backend: str = config.WHISPER_BACKEND) -> None:
    with db.tx() as conn:
        todo = db.episodes_with_status(conn, "downloaded")
    if limit:
        todo = todo[:limit]
    print(f"transcribe[{backend}]: {len(todo)} episodes")
    for ep in todo:
        out_path = config.TRANSCRIPT_DIR / f"{ep['id']}.json"
        t0 = time.time()
        try:
            if out_path.exists():
                result = json.loads(out_path.read_text())
            else:
                result = transcribe_file(Path(ep["audio_path"]), backend)
                result["episode_id"] = ep["id"]
                out_path.write_text(json.dumps(result, ensure_ascii=False))
            with db.tx() as conn:
                conn.execute("DELETE FROM segments WHERE episode_id=?", (ep["id"],))
                conn.executemany(
                    'INSERT INTO segments (episode_id, idx, start, "end", text) VALUES (?,?,?,?,?)',
                    [(ep["id"], i, s["start"], s["end"], s["text"]) for i, s in enumerate(result["segments"])],
                )
                db.set_status(conn, ep["id"], "transcribed")
            speed = (result.get("duration") or 0) / max(time.time() - t0, 1e-6)
            print(f"  ✓ {ep['title'][:60]}  ({len(result['segments'])} segs, {speed:.1f}x realtime)")
        except Exception as exc:  # noqa: BLE001
            with db.tx() as conn:
                db.set_status(conn, ep["id"], "failed", f"transcribe: {exc}")
            print(f"  ✗ {ep['title'][:60]}: {exc}")
