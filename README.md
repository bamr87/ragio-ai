# Ragio-Lab

A fully local AI engine whose knowledge base is **every Radiolab episode** — audio, transcripts with exact timestamps, and episode context — served through a RAG stack you can query, review, and build on.

```
RSS feed ──► catalog (SQLite) ──► download MP3 ──► Whisper (local) ──► windows ──► Qdrant (LlamaIndex)
                                                                                        │
              browser ◄── FastAPI ◄── Ollama LLM ◄── hybrid retrieval + neighbor expansion
                            │
                            └── ffmpeg clip ──► (optional) Qwen2-Audio "deep listen"
```

## Stack

| Layer | Choice | Why |
|---|---|---|
| Audio ingestion | **Whisper** via `faster-whisper` (Linux/NVIDIA/CPU) or `mlx-whisper` (Apple Silicon) | word-level timestamps, VAD to skip music beds, fully offline |
| Catalog / docstore | **SQLite** | pipeline state per episode; full text of every segment + window; parent/neighbor lookups |
| Vectors | **Qdrant** (Docker, local) | hybrid dense + BM25 in one collection, payload filters, survives restarts |
| RAG framework | **LlamaIndex** | Qdrant + Ollama integrations, retriever abstraction, evaluators |
| Embeddings | `nomic-embed-text` via **Ollama** | one runtime for embeddings and generation |
| Brain | `qwen2.5:14b` (or `llama3.1:8b`) via **Ollama** | text answers with citations |
| Deep listen (optional) | **Qwen2-Audio-7B-Instruct** on the retrieved clip | questions about tone, music, sound design |
| App | **FastAPI**, single `app.py` + one static HTML page | JSON + SSE streaming, clip serving |

## Quick start

```bash
# 0. prerequisites: Python 3.11+, Docker, ffmpeg, Ollama (https://ollama.com)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
#   Apple Silicon: pip install mlx-whisper  and set RAGIO_WHISPER_BACKEND=mlx
cp .env.example .env    # edit, then: set -a; source .env; set +a

make up                 # Qdrant on :6333 (dashboard at http://localhost:6333/dashboard)
ollama pull nomic-embed-text && ollama pull qwen2.5:14b

# 1. first taste: 3 episodes end-to-end
make pipeline
python -m ragio ask "What does Radiolab say about the placebo effect?"

# 2. the whole archive (resumable — rerun any stage any time)
python -m ragio download
python -m ragio transcribe        # the long one: see "Compute plan" below
python -m ragio chunk
python -m ragio index
python -m ragio index --summaries # optional parent index (needs the LLM)

# 3. serve
make serve                        # http://localhost:8000
```

`python -m ragio status` shows how many episodes are at each stage.

## Endpoints

`make serve` serves the UI at `/` and the JSON API under `/api/`; [DESIGN.md § 6](DESIGN.md#6-api-fastapi-apppy) lists every route. Three operational probes sit beside them. They need no authentication and are sent with `Cache-Control: no-store`, so a container health check or a monitor can poll them:

| Route | Answers |
| --- | --- |
| `GET /healthz` | Liveness: `200 {"status":"ok"}`. Touches neither SQLite nor Qdrant. |
| `GET /readyz` | Readiness: `200 {"status":"ok","checks":{"db":"ok","qdrant":"ok"}}` when SQLite answers `SELECT 1` and Qdrant answers within 1 s; otherwise `503` with `"status":"degraded"` and the failing check's error in place of `"ok"`. Ollama is not checked. |
| `GET /version` | `{"name":"ragio-lab","version":"<release-please version>","commit":"$GIT_SHA","built_at":"$BUILT_AT"}`; `commit` and `built_at` are `null` when the variable is unset. |

## Data model: parent → child

```
episode (id, title, published, description, audio_url, page_url, summary, status)
  └── window[idx]  (start, end, text)       ← what gets embedded; ~220 words / ≤120 s, 1-segment overlap
        └── segment[idx] (start, end, text) ← raw Whisper segments; exact word timing lives in the JSON
```

Qdrant point ids are `uuid5(episode_id:idx)`, so re-indexing an episode upserts instead of duplicating. Every Qdrant payload carries `episode_id, idx, title, published, start, end, page_url` — enough to jump straight to the moment in the audio without touching SQLite. SQLite is consulted only to expand a hit with its neighboring windows (parent context) before generation.

## Query flow (`ragio/rag.py`)

1. Hybrid search (dense + BM25) → top-8 windows
2. Expand each hit ±1 window from SQLite; merge overlapping ranges within an episode
3. Group by episode, rank by best score
4. Prompt with labelled evidence `[E3] Title (date) @ 12:04–14:31` and ask the LLM to cite labels
5. Return answer + sources, each with `clip_url` → `/api/clip?ep=…&start=…&end=…` (ffmpeg, cached)

## Troubleshooting clips

The UI plays every source from `/api/clip`, and `/api/listen` cuts the same kind of clip for the audio model. When a cut can't be served, both say why:

| Status | Means | Do |
| --- | --- | --- |
| `400` | `start` or `end` is not a finite number of seconds, `start` is negative, or the window is empty or too long (10 min for a clip, 3 min for deep listen) | Fix the request |
| `404` `no audio for episode …` | The episode is unknown, or its MP3 hasn't been downloaded | `python -m ragio status`, then `python -m ragio download` |
| `503` `ffmpeg not found on PATH` | ffmpeg isn't installed, or isn't on the `PATH` of the process running `make serve` | Install ffmpeg (`brew install ffmpeg`, `apt install ffmpeg`); the next request finds it |
| `502` `ffmpeg failed: …` | ffmpeg ran and failed on that episode's audio; the message is its last error line | Check or re-download the episode's MP3 under `data/audio/`. Nothing was cached, so a retry runs the cut again |

A clip is cached in `data/clips/` only when ffmpeg succeeds. Earlier versions could leave a truncated clip there after a failed cut and serve it from then on: if a clip plays short or not at all, delete `data/clips/`. It is only a cache: there is no need to stop the server, and clips are cut again on demand.

## Compute plan

Radiolab has ~700+ episodes averaging ~45 min ⇒ roughly **500–600 hours of audio**.

| Backend | Hardware | Speed | Wall time for the archive |
|---|---|---|---|
| `mlx-whisper` large-v3 | M-series Mac | ~8–15× realtime | ~40–70 h |
| `faster-whisper` large-v3 fp16 | RTX 3090/4090 | ~30–50× | ~12–20 h |
| `faster-whisper` medium.en int8 | CPU only | ~1–2× | weeks — use `--limit` and let it run |

Transcription is resumable and idempotent (transcript JSON is cached), so run it in the background over several nights. Everything downstream (chunk/index) is minutes, not hours.

## Configuration

All knobs are environment variables with defaults in `ragio/config.py` — chunk size, top-k, hybrid on/off, model names, collection names, audio backend. See `.env.example`.

## Evaluation

Put 20–30 questions with known answers in `eval/questions.jsonl` (`python -m ragio search "…"` shows episode ids), then `python -m ragio eval` prints hit@1/3/5. Re-run after every retrieval change.

## A note on the audio-native model

Ollama serves text and vision models; it does not (as of this writing) run audio-input models, and the model usually called "Qwen2.5-Audio" ships as **Qwen2-Audio-7B-Instruct** (Qwen2.5-Omni is the newer any-to-any successor). Ragio therefore uses a text LLM in Ollama as the brain and treats the audio model as a *tool* invoked on a short retrieved clip: `RAGIO_AUDIO_BACKEND=openai-compatible` with `vllm serve Qwen/Qwen2-Audio-7B-Instruct --port 8001` on a GPU box, or `transformers` in-process. Llama-3.2-Vision is a vision model — no benefit for a radio archive; keep it for a future "episode art / show notes images" feature if you like.

## License

The code in this repository is MIT-licensed: see [LICENSE](LICENSE).

The licence does not cover Radiolab's content. The audio and transcripts belong to WNYC / Radiolab. This project is for personal research and analysis; don't redistribute the audio, clips, or transcripts.

## Layout

```
app.py                FastAPI app (all endpoints)
ragio/config.py       env-driven settings
ragio/db.py           SQLite schema + docstore helpers
ragio/catalog.py      RSS → episodes; MP3 download
ragio/transcribe.py   Whisper backends → transcripts JSON + segments
ragio/chunk.py        segments → windows
ragio/index.py        LlamaIndex + Qdrant (segments + episode summaries)
ragio/rag.py          retrieve / expand / prompt / answer
ragio/clips.py        ffmpeg clips + audio-native "deep listen"
ragio/evaluate.py     retrieval hit-rate
static/index.html     the UI
eval/questions.jsonl  your golden questions
tests/                offline tests (python -m pytest -q; no Qdrant, Ollama or corpus needed)
```

## Documentation

Every document in the repository is listed, one line each, in [docs/README.md](docs/README.md): the requirements in [PRD.md](PRD.md), the architecture in [DESIGN.md](DESIGN.md), how to work here in [AGENTS.md](AGENTS.md), and the constitution, ADRs and amendments under `docs/`.
