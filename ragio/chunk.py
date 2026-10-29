"""Stage 3 — turn fine-grained Whisper segments into retrieval windows (child nodes).

Whisper emits 2–10 s segments: too small to embed meaningfully on their own.
We merge consecutive segments into ~220-word / ≤120 s windows, overlapping by
one segment so a thought split across a boundary is still retrievable.

Parent → child:  episode  →  window[idx] (start, end)  →  segments (exact words)

    python -m ragio chunk
"""
from __future__ import annotations

from . import config, db


def build_windows(segments: list[dict], target_words: int = config.WINDOW_TARGET_WORDS,
                  max_seconds: float = config.WINDOW_MAX_SECONDS,
                  overlap: int = config.WINDOW_OVERLAP_SEGMENTS) -> list[dict]:
    windows: list[dict] = []
    i = 0
    n = len(segments)
    while i < n:
        j = i
        words = 0
        start = segments[i]["start"]
        while j < n:
            seg_words = len(segments[j]["text"].split())
            too_long = segments[j]["end"] - start > max_seconds
            if j > i and (words + seg_words > target_words or too_long):
                break
            words += seg_words
            j += 1
        chunk = segments[i:j]
        windows.append({
            "idx": len(windows),
            "start": chunk[0]["start"],
            "end": chunk[-1]["end"],
            "text": " ".join(s["text"] for s in chunk).strip(),
        })
        if j >= n:
            break
        i = max(j - overlap, i + 1)  # step back `overlap` segments, always make progress
    return windows


def chunk_pending() -> None:
    with db.tx() as conn:
        todo = db.episodes_with_status(conn, "transcribed")
    print(f"chunk: {len(todo)} episodes")
    for ep in todo:
        with db.tx() as conn:
            segs = [dict(r) for r in conn.execute(
                'SELECT start, "end", text FROM segments WHERE episode_id=? ORDER BY idx', (ep["id"],))]
            wins = build_windows(segs)
            conn.execute("DELETE FROM windows WHERE episode_id=?", (ep["id"],))
            conn.executemany(
                'INSERT INTO windows (episode_id, idx, start, "end", text) VALUES (?,?,?,?,?)',
                [(ep["id"], w["idx"], w["start"], w["end"], w["text"]) for w in wins],
            )
            db.set_status(conn, ep["id"], "chunked")
        print(f"  ✓ {ep['title'][:60]}  {len(segs)} segments → {len(wins)} windows")
