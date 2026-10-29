"""Retrieval evaluation: does the right episode show up in the top results?

eval/questions.jsonl lines look like:
  {"q": "the episode about the man who lost the ability to form new memories", "expect": ["<episode_id>"]}

Start with 20–30 questions you know the answers to; this is the single most useful
signal when tuning chunk size, top-k, hybrid on/off, or a new embedding model.
"""
from __future__ import annotations

import json
from pathlib import Path

from .rag import retrieve


def run_eval(path: str) -> dict:
    qs = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    hit1 = hit3 = hit5 = 0
    for q in qs:
        hits = [h.episode_id for h in retrieve(q["q"])]
        expected = set(q["expect"])
        hit1 += bool(expected & set(hits[:1]))
        hit3 += bool(expected & set(hits[:3]))
        hit5 += bool(expected & set(hits[:5]))
        mark = "✓" if expected & set(hits[:5]) else "✗"
        print(f"  {mark} {q['q'][:70]}")
    n = max(len(qs), 1)
    result = {"n": len(qs), "hit@1": hit1 / n, "hit@3": hit3 / n, "hit@5": hit5 / n}
    print(json.dumps(result, indent=2))
    return result
