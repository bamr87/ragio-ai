"""/api/clip and /api/listen error handling (issue #8).

Bad times answer 400, an unknown episode 404, a missing ffmpeg 503 and a failing ffmpeg 502,
and a failed cut leaves nothing in the clip cache. Offline: no Qdrant, Ollama, network or real
ffmpeg. Each test points `ragio.config` at a temp directory and puts a stub `ffmpeg` (or none)
on PATH, so the result is the same whether or not the machine has ffmpeg installed.
"""
from __future__ import annotations

import os
import shlex
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

import app as ragio_app
from ragio import config, db

# A stand-in for ffmpeg. Like the real one, it picks the output format from the extension of its
# last argument and refuses a name it can't map; otherwise it writes OUTPUT to that path, counts the
# call, and exits with EXIT_CODE after printing STDERR.
STUB = """#!/bin/sh
for out; do :; done
echo call >> {counter}
case "$out" in
  *.mp3|*.wav) ;;
  *) echo "Unable to choose an output format for '$out'" >&2; exit 1 ;;
esac
printf %s {output} > "$out"
printf '%s\\n' {stderr} >&2
exit {exit_code}
"""

INVALID = "Invalid data found when processing input"


class ClipErrorTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.clip_dir = self.tmp / "clips"
        self.clip_dir.mkdir()
        self.counter = self.tmp / "ffmpeg-calls"
        audio = self.tmp / "ep1.mp3"
        audio.write_bytes(b"not really audio")

        # config is read once at import; db and clips read these attributes at call time.
        for name, value in (("DB_PATH", self.tmp / "ragio.sqlite"), ("CLIP_DIR", self.clip_dir)):
            patcher = mock.patch.object(config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        with db.tx() as conn:
            conn.execute("INSERT INTO episodes (id, title, audio_path) VALUES (?, ?, ?)",
                         ("ep1", "A test episode", str(audio)))

        self.client = TestClient(ragio_app.app, raise_server_exceptions=False)

    # --- helpers ---------------------------------------------------------------------------------

    def use_path(self, *dirs: Path) -> None:
        """Make PATH exactly `dirs` for the rest of the test."""
        patcher = mock.patch.dict(os.environ, {"PATH": os.pathsep.join(str(d) for d in dirs)})
        patcher.start()
        self.addCleanup(patcher.stop)

    def without_ffmpeg(self) -> None:
        empty = self.tmp / "empty-bin"
        empty.mkdir()
        self.use_path(empty)

    def with_stub_ffmpeg(self, *, exit_code: int, output: str = "partial", stderr: str = "") -> None:
        bin_dir = self.tmp / "stub-bin"
        bin_dir.mkdir()
        stub = bin_dir / "ffmpeg"
        stub.write_text(STUB.format(counter=shlex.quote(str(self.counter)), output=shlex.quote(output),
                                    stderr=shlex.quote(stderr), exit_code=exit_code))
        stub.chmod(stub.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        self.use_path(bin_dir, *os.environ.get("PATH", os.defpath).split(os.pathsep))

    def ffmpeg_calls(self) -> int:
        return len(self.counter.read_text().splitlines()) if self.counter.exists() else 0

    def cached(self) -> list[str]:
        return sorted(p.name for p in self.clip_dir.iterdir())

    def clip(self, ep: str = "ep1", start: str = "10", end: str = "20"):
        return self.client.get("/api/clip", params={"ep": ep, "start": start, "end": end})

    def listen(self, start: str = "10", end: str = "20", episode_id: str = "ep1"):
        # A raw body, because NaN and Infinity are not standard JSON and an encoder may refuse them;
        # Python's json module, which parses the request, accepts both.
        body = f'{{"episode_id": "{episode_id}", "start": {start}, "end": {end}}}'
        return self.client.post("/api/listen", content=body, headers={"content-type": "application/json"})

    # --- 400: times that are not finite, or start before 0 ----------------------------------------

    def test_clip_rejects_non_finite_or_negative_times(self) -> None:
        self.with_stub_ffmpeg(exit_code=0)
        for start, end in (("nan", "5"), ("0", "nan"), ("0", "inf"), ("-inf", "5"), ("-1", "5")):
            with self.subTest(start=start, end=end):
                response = self.clip(start=start, end=end)
                self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(self.ffmpeg_calls(), 0)
        self.assertEqual(self.cached(), [])

    def test_listen_rejects_non_finite_or_negative_times(self) -> None:
        self.with_stub_ffmpeg(exit_code=0)
        for start, end in (("NaN", "5"), ("0", "NaN"), ("0", "Infinity"), ("-Infinity", "5"), ("-1", "5")):
            with self.subTest(start=start, end=end):
                response = self.listen(start=start, end=end)
                self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(self.ffmpeg_calls(), 0)

    def test_window_length_limits_still_apply(self) -> None:
        self.assertEqual(self.clip(start="10", end="10").status_code, 400)
        self.assertEqual(self.clip(start="0", end="601").status_code, 400)
        self.assertEqual(self.listen(start="0", end="181").status_code, 400)

    # --- 404 / 503: unknown episode versus missing ffmpeg -----------------------------------------

    def test_missing_ffmpeg_is_503_naming_ffmpeg(self) -> None:
        self.without_ffmpeg()
        for response in (self.clip(), self.listen()):
            with self.subTest(path=response.request.url.path):
                self.assertEqual(response.status_code, 503, response.text)
                self.assertIn("ffmpeg", response.json()["detail"])
        self.assertEqual(self.cached(), [])

    def test_unknown_episode_is_still_404_without_ffmpeg(self) -> None:
        self.without_ffmpeg()
        for response in (self.clip(ep="missing"), self.listen(episode_id="missing")):
            with self.subTest(path=response.request.url.path):
                self.assertEqual(response.status_code, 404, response.text)
                self.assertIn("no audio for episode missing", response.json()["detail"])

    # --- 502: ffmpeg fails, nothing is cached, a retry cuts again ---------------------------------

    def test_failing_ffmpeg_is_502_and_caches_nothing(self) -> None:
        self.with_stub_ffmpeg(exit_code=183, stderr=INVALID)
        first = self.clip()
        self.assertEqual(first.status_code, 502, first.text)
        self.assertIn("ffmpeg failed", first.json()["detail"])
        self.assertIn(INVALID, first.json()["detail"])
        self.assertEqual(self.cached(), [], "a failed cut must leave no clip or temp file behind")

        second = self.clip()
        self.assertEqual(second.status_code, 502, second.text)
        self.assertEqual(self.ffmpeg_calls(), 2, "the retry must run ffmpeg again, not serve a cached file")
        self.assertEqual(self.cached(), [])

    def test_failing_ffmpeg_on_listen_is_502(self) -> None:
        self.with_stub_ffmpeg(exit_code=1, stderr=INVALID)
        response = self.listen()
        self.assertEqual(response.status_code, 502, response.text)
        self.assertIn(INVALID, response.json()["detail"])
        self.assertEqual(self.cached(), [])

    # --- success: same response, cached under the old name, no temp file left --------------------

    def test_successful_cut_is_cached_and_served_from_the_cache(self) -> None:
        self.with_stub_ffmpeg(exit_code=0, output="ID3 a clip")
        first = self.clip()
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(first.headers["content-type"], "audio/mpeg")
        self.assertEqual(first.content, b"ID3 a clip")
        self.assertEqual(self.cached(), ["ep1_9.5_20.5.mp3"])

        second = self.clip()
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.content, b"ID3 a clip")
        self.assertEqual(self.ffmpeg_calls(), 1, "the second request is served from the cache")

    def test_deleted_clip_cache_is_recreated(self) -> None:
        # The README says data/clips/ is safe to delete while the server runs; config only creates
        # it at import, so the cut has to recreate it rather than answer 404.
        self.with_stub_ffmpeg(exit_code=0, output="ID3 a clip")
        self.clip_dir.rmdir()
        response = self.clip()
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.content, b"ID3 a clip")
        self.assertEqual(self.cached(), ["ep1_9.5_20.5.mp3"])


if __name__ == "__main__":
    unittest.main()
