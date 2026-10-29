"""Query layer: hybrid retrieval → neighbor expansion → grouped, timestamp-cited answer.

Pipeline for one question:
  1. Hybrid search (dense + BM25) over `ragio_segments` → top-k windows
  2. Expand each hit by ±N neighboring windows from SQLite (parent context)
  3. Merge overlapping ranges, group by episode, rank episodes by best score
  4. Build a prompt whose evidence blocks are labelled [E1 @ 12:34–14:02] ...
  5. Ask the local LLM; return answer + machine-readable sources (with clip URLs)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterator

from llama_index.core import Settings

from . import config, db
from .index import load_index


@dataclass
class Passage:
    episode_id: str
    title: str
    published: str
    page_url: str
    start: float
    end: float
    text: str
    score: float
    lo_idx: int
    hi_idx: int


@dataclass
class EpisodeHit:
    episode_id: str
    title: str
    published: str
    page_url: str
    score: float
    passages: list[Passage] = field(default_factory=list)


def fmt_ts(seconds: float) -> str:
    s = int(seconds)
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


_retriever = None


def retriever():
    global _retriever
    if _retriever is None:
        index = load_index(config.SEGMENT_COLLECTION)
        kwargs = {"similarity_top_k": config.TOP_K}
        if config.HYBRID:
            kwargs |= {"sparse_top_k": config.SPARSE_TOP_K, "vector_store_query_mode": "hybrid"}
        _retriever = index.as_retriever(**kwargs)
    return _retriever


def retrieve(question: str, neighbors: int = config.NEIGHBOR_WINDOWS) -> list[EpisodeHit]:
    nodes = retriever().retrieve(question)
    by_episode: dict[str, EpisodeHit] = {}
    with db.tx() as conn:
        for n in nodes:
            md = n.metadata
            ep_id, idx = md["episode_id"], int(md["idx"])
            rows = db.get_windows(conn, ep_id, idx - neighbors, idx + neighbors)
            if not rows:
                continue
            passage = Passage(
                episode_id=ep_id, title=md.get("title", ""), published=md.get("published", ""),
                page_url=md.get("page_url", ""), start=rows[0]["start"], end=rows[-1]["end"],
                text=" ".join(r["text"] for r in rows), score=float(n.score or 0.0),
                lo_idx=rows[0]["idx"], hi_idx=rows[-1]["idx"],
            )
            hit = by_episode.setdefault(ep_id, EpisodeHit(
                episode_id=ep_id, title=passage.title, published=passage.published,
                page_url=passage.page_url, score=passage.score))
            hit.score = max(hit.score, passage.score)
            hit.passages.append(passage)
    for hit in by_episode.values():
        hit.passages = _merge_overlaps(sorted(hit.passages, key=lambda p: p.lo_idx))
    return sorted(by_episode.values(), key=lambda h: h.score, reverse=True)


def _merge_overlaps(passages: list[Passage]) -> list[Passage]:
    """Two hits in the same episode whose expanded ranges touch become one passage."""
    merged: list[Passage] = []
    for p in passages:
        if merged and p.lo_idx <= merged[-1].hi_idx + 1:
            last = merged[-1]
            with db.tx() as conn:
                rows = db.get_windows(conn, p.episode_id, last.lo_idx, max(last.hi_idx, p.hi_idx))
            last.hi_idx = rows[-1]["idx"]
            last.end = rows[-1]["end"]
            last.text = " ".join(r["text"] for r in rows)
            last.score = max(last.score, p.score)
        else:
            merged.append(p)
    return merged


SYSTEM = (
    "You are Ragio, an analyst with access to transcripts of every Radiolab episode. Answer the question "
    "using ONLY the evidence blocks. Cite evidence inline with its label, e.g. [E2 @ 14:02]. If the evidence "
    "doesn't answer the question, say what is and isn't covered rather than guessing. Be specific about which "
    "episode says what; when multiple episodes touch the topic, contrast them."
)


def build_prompt(question: str, hits: list[EpisodeHit], max_chars: int = 24_000) -> tuple[str, list[dict]]:
    blocks, sources, used = [], [], 0
    label = 0
    for hit in hits:
        for p in hit.passages:
            label += 1
            tag = f"E{label}"
            block = (f"[{tag}] {hit.title} ({hit.published}) @ {fmt_ts(p.start)}–{fmt_ts(p.end)}\n{p.text}\n")
            if used + len(block) > max_chars:
                break
            used += len(block)
            blocks.append(block)
            sources.append({
                "label": tag, "episode_id": hit.episode_id, "title": hit.title, "published": hit.published,
                "page_url": hit.page_url, "start": p.start, "end": p.end,
                "start_hms": fmt_ts(p.start), "end_hms": fmt_ts(p.end), "score": round(p.score, 4),
                "clip_url": f"/api/clip?ep={hit.episode_id}&start={p.start:.1f}&end={p.end:.1f}",
                "excerpt": p.text[:280] + ("…" if len(p.text) > 280 else ""),
            })
    prompt = f"{SYSTEM}\n\n# EVIDENCE\n" + "\n".join(blocks) + f"\n# QUESTION\n{question}\n\n# ANSWER\n"
    return prompt, sources


def ask(question: str) -> dict:
    hits = retrieve(question)
    prompt, sources = build_prompt(question, hits)
    answer = Settings.llm.complete(prompt).text.strip()
    return {"question": question, "answer": answer, "sources": sources,
            "episodes": [{"episode_id": h.episode_id, "title": h.title, "score": round(h.score, 4)} for h in hits]}


def ask_stream(question: str) -> Iterator[dict]:
    """Yields {'type': 'sources', ...} first, then {'type': 'token', 'text': ...} chunks, then {'type': 'done'}."""
    hits = retrieve(question)
    prompt, sources = build_prompt(question, hits)
    yield {"type": "sources", "sources": sources}
    for chunk in Settings.llm.stream_complete(prompt):
        yield {"type": "token", "text": chunk.delta}
    yield {"type": "done"}


def search(question: str) -> list[dict]:
    return [
        {"episode_id": h.episode_id, "title": h.title, "published": h.published, "score": round(h.score, 4),
         "passages": [asdict(p) | {"start_hms": fmt_ts(p.start), "end_hms": fmt_ts(p.end)} for p in h.passages]}
        for h in retrieve(question)
    ]
