"""CLI entry point:  python -m ragio <stage> [options]

Stages run in order and are resumable — each episode carries a status, so you
can stop at any point and re-run the same command to pick up where you left off.
"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ragio")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("catalog", help="parse the RSS feed into SQLite")
    d = sub.add_parser("download", help="download audio for catalogued episodes")
    d.add_argument("--limit", type=int)
    t = sub.add_parser("transcribe", help="run Whisper on downloaded episodes")
    t.add_argument("--limit", type=int)
    t.add_argument("--backend", choices=["faster-whisper", "mlx"])
    sub.add_parser("chunk", help="merge Whisper segments into retrieval windows")
    i = sub.add_parser("index", help="embed windows into Qdrant")
    i.add_argument("--recreate", action="store_true")
    i.add_argument("--summaries", action="store_true", help="also build the per-episode summary index (needs LLM)")
    a = sub.add_parser("ask", help="one-shot question from the terminal")
    a.add_argument("question")
    s = sub.add_parser("search", help="retrieval only, no generation")
    s.add_argument("question")
    sub.add_parser("status", help="pipeline progress")
    e = sub.add_parser("eval", help="retrieval hit-rate against eval/questions.jsonl")
    e.add_argument("--file", default="eval/questions.jsonl")

    args = p.parse_args(argv)

    if args.cmd == "catalog":
        from .catalog import build_catalog
        build_catalog()
    elif args.cmd == "download":
        from .catalog import download_audio
        download_audio(limit=args.limit)
    elif args.cmd == "transcribe":
        from . import config
        from .transcribe import transcribe_pending
        transcribe_pending(limit=args.limit, backend=args.backend or config.WHISPER_BACKEND)
    elif args.cmd == "chunk":
        from .chunk import chunk_pending
        chunk_pending()
    elif args.cmd == "index":
        from .index import index_episode_summaries, index_segments
        index_segments(recreate=args.recreate)
        if args.summaries:
            index_episode_summaries(recreate=args.recreate)
    elif args.cmd == "ask":
        from .rag import ask
        out = ask(args.question)
        print(out["answer"], "\n")
        for s in out["sources"]:
            print(f"  [{s['label']}] {s['title']} @ {s['start_hms']}–{s['end_hms']}  {s['page_url']}")
    elif args.cmd == "search":
        from .rag import search
        print(json.dumps(search(args.question), indent=2))
    elif args.cmd == "status":
        from . import db
        with db.tx() as conn:
            rows = conn.execute("SELECT status, COUNT(*) n FROM episodes GROUP BY status").fetchall()
            n_win = conn.execute("SELECT COUNT(*) FROM windows").fetchone()[0]
        for r in rows:
            print(f"{r['status']:>12}: {r['n']}")
        print(f"{'windows':>12}: {n_win}")
    elif args.cmd == "eval":
        from .evaluate import run_eval
        run_eval(args.file)


if __name__ == "__main__":
    main(sys.argv[1:])
