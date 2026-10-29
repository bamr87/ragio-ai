<!-- kit: sdlc v0.2.0 · AGENTS.md is the CANONICAL agent instructions file (decision D4, UPS-AGENT-07/08). CLAUDE.md is a pointer to this file and carries no rules of its own. -->

# ragio-ai

> Ragio-Lab: a fully local RAG engine over every Radiolab episode — Whisper transcripts with exact timestamps, SQLite, Qdrant, LlamaIndex, Ollama and a FastAPI web app.

**Read [docs/constitution.md](docs/constitution.md) first.** It wins over this file when they disagree.

## What this repo is

Ragio-Lab turns the Radiolab archive into grounded, second-accurate context for a local LLM: it catalogs the RSS feed, downloads audio, transcribes it locally with Whisper, chunks segments into retrieval windows, indexes them in Qdrant (hybrid dense + BM25), and answers questions with cited, playable clips through a FastAPI app. It serves one researcher running everything on one machine. Product requirements are in [PRD.md](PRD.md), the architecture in [DESIGN.md](DESIGN.md), and the feature catalog in [features/features.yml](features/features.yml). A change is done when `python -m compileall -q ragio app.py` passes, the affected pipeline stage or endpoint has been exercised locally, retrieval changes have been re-run against `python -m ragio eval`, and README/DESIGN/PRD/features reflect it.

## Stack & commands

Python 3.11+, FastAPI, LlamaIndex, Qdrant (Docker), Ollama, faster-whisper / mlx-whisper, ffmpeg, SQLite.

```bash
# install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env && set -a && source .env && set +a
make up                                   # Qdrant on :6333
ollama pull nomic-embed-text && ollama pull qwen2.5:14b

# run
make pipeline                             # 3 episodes end-to-end
python -m ragio status
make serve                                # http://localhost:8000

# test (no automated suite yet — see Standard deviations)
python -m compileall -q ragio app.py
python -m ragio eval                      # retrieval hit@1/3/5 (needs an indexed corpus)

# lint / build: none configured yet; the app is run from source
```

## Layout

| Path | What lives there |
| --- | --- |
| `app.py` | FastAPI app — every HTTP endpoint |
| `ragio/` | The pipeline: `catalog`, `transcribe`, `chunk`, `index`, `rag`, `clips`, `evaluate`, `db`, `config`, CLI in `__main__.py` |
| `static/` | The single-page UI |
| `eval/` | Golden retrieval questions (`questions.jsonl`) |
| `data/` | **Gitignored.** Audio, transcripts, SQLite, clips — never commit |
| `docs/` | Constitution, amendments, ADRs |
| `features/` | The feature catalog (`features/v1`) |

## Conventions

- **Backlog of record:** GitHub Issues with the fleet labels. Pick up issues labelled `agent:ready`; never touch `agent:hold` or `human-review`.
- **Definition of Done:** the fleet default pull request template (inherited). Every box is ticked before merge.
- **Decisions:** ADRs in `docs/adr/NNNN-slug.md`; DESIGN.md § 12 holds the pre-ADR decision log.
- **Commits:** Conventional Commits (`feat fix docs style refactor test chore perf ci build`). The PR title is the squash subject.
- **Branches:** `feature/`, `fix/`, `docs/`, `refactor/`, `test/` or `chore/` off `main`.
- **Changelog:** `CHANGELOG.md` is written by release-please from the PR titles. Never edit it, or a version number, by hand.
- **Features:** a user-facing change updates `features/features.yml` (`RAGIO-NNN`, `prd:` refs) and the PRD in the same PR.
- **Never:** push to `main` directly; commit audio, transcripts, clips, the SQLite DB, `.env` or any secret; redistribute Radiolab content; suppress type errors or leave empty exception handlers.

## Fleet context

Member of the bamr87 fleet. The standards live in the hub: <https://github.com/bamr87/bamr87/tree/main/specs> (the Universal Project Standard). The repo's SDLC declaration is `.github/sdlc.yml`.

## Standard deviations

- UPS-QA-10 waived until 2027-03-01: no automated test suite yet; the pipeline depends on Ollama, Qdrant and Whisper models. First tests are a follow-up.
