"""Stage 4 — embed windows into Qdrant through LlamaIndex.

Two collections:
  * ragio_segments — one point per timestamp window (child). Hybrid dense+BM25.
  * ragio_episodes — one point per episode summary (parent). Built with --summaries
                     once you have an LLM running; answers "which episodes are about X?"

    python -m ragio index [--summaries] [--recreate]
"""
from __future__ import annotations

import uuid

import qdrant_client
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore

from . import config, db

NAMESPACE = uuid.UUID("7d3a4f7e-2d0b-4b83-9d2c-000000000001")


def node_id(episode_id: str, idx: int | None = None) -> str:
    """Qdrant needs UUID point ids; derive them deterministically so re-indexing upserts."""
    key = episode_id if idx is None else f"{episode_id}:{idx}"
    return str(uuid.uuid5(NAMESPACE, key))


def configure_llamaindex() -> None:
    Settings.embed_model = OllamaEmbedding(model_name=config.EMBED_MODEL, base_url=config.OLLAMA_URL)
    Settings.llm = Ollama(model=config.LLM_MODEL, base_url=config.OLLAMA_URL,
                          request_timeout=300.0, context_window=config.LLM_CONTEXT)


def qdrant() -> qdrant_client.QdrantClient:
    return qdrant_client.QdrantClient(url=config.QDRANT_URL)


def vector_store(collection: str) -> QdrantVectorStore:
    kwargs = {}
    if config.HYBRID:
        kwargs = {"enable_hybrid": True, "fastembed_sparse_model": "Qdrant/bm25"}
    return QdrantVectorStore(client=qdrant(), collection_name=collection, **kwargs)


def load_index(collection: str) -> VectorStoreIndex:
    configure_llamaindex()
    return VectorStoreIndex.from_vector_store(vector_store(collection))


def _window_nodes(conn, ep) -> list[TextNode]:
    nodes = []
    for w in db.get_windows(conn, ep["id"], 0, 10**9):
        nodes.append(TextNode(
            id_=node_id(ep["id"], w["idx"]),
            text=f"Radiolab — {ep['title']} ({ep['published']})\n\n{w['text']}",
            metadata={
                "episode_id": ep["id"], "idx": w["idx"], "title": ep["title"],
                "published": ep["published"] or "", "start": w["start"], "end": w["end"],
                "page_url": ep["page_url"] or "",
            },
            excluded_embed_metadata_keys=["episode_id", "idx", "start", "end", "page_url"],
            excluded_llm_metadata_keys=["episode_id", "idx", "page_url"],
        ))
    return nodes


def index_segments(recreate: bool = False, batch_episodes: int = 10) -> None:
    configure_llamaindex()
    client = qdrant()
    if recreate and client.collection_exists(config.SEGMENT_COLLECTION):
        client.delete_collection(config.SEGMENT_COLLECTION)
    store = vector_store(config.SEGMENT_COLLECTION)
    ctx = StorageContext.from_defaults(vector_store=store)
    with db.tx() as conn:
        todo = db.episodes_with_status(conn, "chunked")
    print(f"index: {len(todo)} episodes → {config.SEGMENT_COLLECTION}")
    for i in range(0, len(todo), batch_episodes):
        batch = todo[i:i + batch_episodes]
        with db.tx() as conn:
            nodes = [n for ep in batch for n in _window_nodes(conn, ep)]
        VectorStoreIndex(nodes=nodes, storage_context=ctx, show_progress=True)
        with db.tx() as conn:
            for ep in batch:
                db.set_status(conn, ep["id"], "indexed")
        print(f"  ✓ {len(nodes)} windows from {len(batch)} episodes")


SUMMARY_PROMPT = (
    "You are cataloguing an episode of the radio show Radiolab. Using the transcript below, write:\n"
    "1. A 3–5 sentence summary of what the episode is about and how it unfolds.\n"
    "2. A line 'Themes:' with 5–10 comma-separated themes/topics.\n"
    "3. A line 'People:' with notable guests or figures mentioned.\n"
    "Be concrete and factual; do not invent details.\n\nTITLE: {title}\n\nTRANSCRIPT:\n{text}"
)


def index_episode_summaries(recreate: bool = False, max_chars: int = 60_000) -> None:
    """Parent index: one node per episode with an LLM summary (needs Ollama running)."""
    configure_llamaindex()
    client = qdrant()
    if recreate and client.collection_exists(config.EPISODE_COLLECTION):
        client.delete_collection(config.EPISODE_COLLECTION)
    store = vector_store(config.EPISODE_COLLECTION)
    ctx = StorageContext.from_defaults(vector_store=store)
    with db.tx() as conn:
        eps = conn.execute("SELECT * FROM episodes WHERE status='indexed' ORDER BY published").fetchall()
    print(f"summaries: {len(eps)} episodes → {config.EPISODE_COLLECTION}")
    for ep in eps:
        with db.tx() as conn:
            summary = ep["summary"]
            if not summary:
                text = db.episode_text(conn, ep["id"])[:max_chars]
                summary = Settings.llm.complete(SUMMARY_PROMPT.format(title=ep["title"], text=text)).text.strip()
                conn.execute("UPDATE episodes SET summary=? WHERE id=?", (summary, ep["id"]))
        node = TextNode(
            id_=node_id(ep["id"]),
            text=f"Radiolab — {ep['title']} ({ep['published']})\n\n{summary}",
            metadata={"episode_id": ep["id"], "title": ep["title"], "published": ep["published"] or "",
                      "page_url": ep["page_url"] or "", "duration_sec": ep["duration_sec"] or 0},
            excluded_embed_metadata_keys=["episode_id", "page_url", "duration_sec"],
        )
        VectorStoreIndex(nodes=[node], storage_context=ctx)
        print(f"  ✓ {ep['title'][:70]}")
