"""Central configuration. Everything is overridable via environment variables (see .env.example)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("RAGIO_ROOT", Path(__file__).resolve().parent.parent))
DATA = ROOT / "data"
AUDIO_DIR = DATA / "audio"
TRANSCRIPT_DIR = DATA / "transcripts"
CLIP_DIR = DATA / "clips"
DB_PATH = DATA / "ragio.sqlite"

for _d in (AUDIO_DIR, TRANSCRIPT_DIR, CLIP_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Source -----------------------------------------------------------------
# Radiolab's public podcast feed (confirm at radiolab.org if it ever moves).
FEED_URL = os.environ.get("RAGIO_FEED_URL", "https://feeds.simplecast.com/EmVW7VGp")

# --- Whisper ----------------------------------------------------------------
WHISPER_BACKEND = os.environ.get("RAGIO_WHISPER_BACKEND", "faster-whisper")  # faster-whisper | mlx
WHISPER_MODEL = os.environ.get("RAGIO_WHISPER_MODEL", "large-v3")
WHISPER_COMPUTE = os.environ.get("RAGIO_WHISPER_COMPUTE", "int8")  # faster-whisper compute_type
WHISPER_MLX_REPO = os.environ.get("RAGIO_WHISPER_MLX_REPO", "mlx-community/whisper-large-v3-mlx")

# --- Chunking (child nodes = timestamp windows) --------------------------------
WINDOW_TARGET_WORDS = int(os.environ.get("RAGIO_WINDOW_WORDS", "220"))
WINDOW_MAX_SECONDS = float(os.environ.get("RAGIO_WINDOW_SECONDS", "120"))
WINDOW_OVERLAP_SEGMENTS = int(os.environ.get("RAGIO_WINDOW_OVERLAP", "1"))

# --- Qdrant + embeddings ------------------------------------------------------
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
SEGMENT_COLLECTION = os.environ.get("RAGIO_SEGMENT_COLLECTION", "ragio_segments")
EPISODE_COLLECTION = os.environ.get("RAGIO_EPISODE_COLLECTION", "ragio_episodes")
EMBED_MODEL = os.environ.get("RAGIO_EMBED_MODEL", "nomic-embed-text")  # served by Ollama
HYBRID = os.environ.get("RAGIO_HYBRID", "1") == "1"  # dense + BM25 sparse in Qdrant

# --- LLMs -----------------------------------------------------------------------
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.environ.get("RAGIO_LLM_MODEL", "qwen2.5:14b")  # text brain
LLM_CONTEXT = int(os.environ.get("RAGIO_LLM_CONTEXT", "16384"))
# Optional audio-native model for the "deep listen" path (see ragio/clips.py)
AUDIO_LLM_BACKEND = os.environ.get("RAGIO_AUDIO_BACKEND", "none")  # none | openai-compatible | transformers
AUDIO_LLM_URL = os.environ.get("RAGIO_AUDIO_URL", "http://localhost:8001/v1")
AUDIO_LLM_MODEL = os.environ.get("RAGIO_AUDIO_MODEL", "Qwen/Qwen2-Audio-7B-Instruct")

# --- Retrieval ------------------------------------------------------------------
TOP_K = int(os.environ.get("RAGIO_TOP_K", "8"))
SPARSE_TOP_K = int(os.environ.get("RAGIO_SPARSE_TOP_K", "12"))
NEIGHBOR_WINDOWS = int(os.environ.get("RAGIO_NEIGHBORS", "1"))  # expand each hit by ±N windows
