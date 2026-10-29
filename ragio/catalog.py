"""Stage 0/1 — build the episode catalog from the RSS feed and download audio.

    python -m ragio catalog          # parse feed → episodes table (idempotent)
    python -m ragio download [--limit N]   # fetch MP3s for catalogued episodes
"""
from __future__ import annotations

import hashlib
import re
import time
from datetime import datetime
from pathlib import Path

import feedparser
import httpx

from . import config, db


def _episode_id(guid: str) -> str:
    return hashlib.sha1(guid.encode()).hexdigest()[:12]


def _parse_duration(value) -> int | None:
    """itunes:duration may be '3421' or '57:01' or '1:02:03'."""
    if not value:
        return None
    parts = [int(p) for p in str(value).split(":") if p.strip().isdigit()]
    if not parts:
        return None
    total = 0
    for p in parts:
        total = total * 60 + p
    return total


def _published(entry) -> str | None:
    if getattr(entry, "published_parsed", None):
        return datetime(*entry.published_parsed[:6]).date().isoformat()
    return None


def build_catalog(feed_url: str = config.FEED_URL) -> int:
    feed = feedparser.parse(feed_url)
    if feed.bozo and not feed.entries:
        raise RuntimeError(f"Could not parse feed {feed_url}: {feed.bozo_exception}")
    n_new = 0
    with db.tx() as conn:
        for e in feed.entries:
            guid = e.get("id") or e.get("link") or e.get("title")
            audio_url = next((l.get("href") for l in e.get("enclosures", []) if l.get("href")), None)
            if not audio_url:
                continue
            row = {
                "id": _episode_id(guid),
                "guid": guid,
                "title": re.sub(r"\s+", " ", e.get("title", "")).strip(),
                "published": _published(e),
                "description": re.sub(r"<[^>]+>", "", e.get("summary", "") or ""),
                "page_url": e.get("link"),
                "audio_url": audio_url,
                "duration_sec": _parse_duration(e.get("itunes_duration")),
            }
            cur = conn.execute(
                """INSERT INTO episodes (id, guid, title, published, description, page_url, audio_url, duration_sec)
                   VALUES (:id, :guid, :title, :published, :description, :page_url, :audio_url, :duration_sec)
                   ON CONFLICT(id) DO UPDATE SET title=excluded.title, description=excluded.description,
                       page_url=excluded.page_url, audio_url=excluded.audio_url, duration_sec=excluded.duration_sec""",
                row,
            )
            n_new += cur.rowcount
    print(f"catalog: {len(feed.entries)} entries in feed, {n_new} rows upserted")
    return n_new


def download_audio(limit: int | None = None, sleep: float = 1.0) -> None:
    with db.tx() as conn:
        todo = db.episodes_with_status(conn, "catalogued")
    if limit:
        todo = todo[:limit]
    print(f"download: {len(todo)} episodes to fetch")
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        for ep in todo:
            dest = config.AUDIO_DIR / f"{ep['id']}.mp3"
            try:
                if not dest.exists() or dest.stat().st_size == 0:
                    tmp = dest.with_suffix(".part")
                    with client.stream("GET", ep["audio_url"]) as r:
                        r.raise_for_status()
                        with open(tmp, "wb") as f:
                            for chunk in r.iter_bytes(1 << 16):
                                f.write(chunk)
                    tmp.rename(dest)
                with db.tx() as conn:
                    conn.execute("UPDATE episodes SET audio_path=? WHERE id=?", (str(dest), ep["id"]))
                    db.set_status(conn, ep["id"], "downloaded")
                print(f"  ✓ {ep['published']}  {ep['title'][:70]}")
                time.sleep(sleep)  # be polite to the CDN
            except Exception as exc:  # noqa: BLE001
                with db.tx() as conn:
                    db.set_status(conn, ep["id"], "failed", f"download: {exc}")
                print(f"  ✗ {ep['title'][:70]}: {exc}")
