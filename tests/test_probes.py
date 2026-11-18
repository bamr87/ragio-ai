"""Liveness, readiness and version probes (fleet standard UPS-BE-10 and UPS-BE-11, issue #6).

Runs offline: every test points SQLite at a temp file and Qdrant at a closed port, so no
service, model or corpus is needed. Run with `python -m unittest tests.test_probes` or a
bare `pytest -q` from the repo root.

ragio.config reads its environment once, at import, and pytest imports every test module
into one process, so the tests patch the config attributes per test instead of setting
environment variables at import.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

import app as app_module
import ragio
from ragio import config, db, index

REPO = Path(__file__).resolve().parent.parent
# Port 9 (discard) has no listener on a dev box or a CI runner, so a connect is refused.
CLOSED_QDRANT_URL = "http://127.0.0.1:9"


def fake_qdrant() -> mock.Mock:
    """A Qdrant client that answers, for the happy path."""
    client = mock.Mock(name="QdrantClient")
    client.get_collections.return_value = mock.Mock(collections=[])
    return client


class ProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        for name, value in (("DB_PATH", self.tmp / "ragio.sqlite"), ("QDRANT_URL", CLOSED_QDRANT_URL)):
            patcher = mock.patch.object(config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = TestClient(app_module.app)

    def test_healthz_is_ok_without_touching_sqlite_or_qdrant(self) -> None:
        touched = AssertionError("liveness must not touch SQLite or Qdrant")
        with mock.patch.object(db, "tx", side_effect=touched), \
                mock.patch.object(index, "qdrant", side_effect=touched):
            resp = self.client.get("/healthz")

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"status": "ok"})
        self.assertEqual(resp.headers.get("cache-control"), "no-store")

    def test_readyz_degrades_within_two_seconds_when_qdrant_is_unreachable(self) -> None:
        started = time.monotonic()
        resp = self.client.get("/readyz")
        elapsed = time.monotonic() - started

        self.assertEqual(resp.status_code, 503)
        body = resp.json()
        self.assertEqual(body["status"], "degraded")
        self.assertEqual(body["checks"]["db"], "ok")
        self.assertNotEqual(body["checks"]["qdrant"], "ok")
        self.assertTrue(body["checks"]["qdrant"], "a failed check names its error")
        self.assertLess(elapsed, 2.0)
        self.assertEqual(resp.headers.get("cache-control"), "no-store")

    def test_readyz_is_ok_when_sqlite_and_qdrant_answer(self) -> None:
        client = fake_qdrant()
        with mock.patch.object(index, "qdrant", return_value=client) as make_client:
            resp = self.client.get("/readyz")

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"status": "ok", "checks": {"db": "ok", "qdrant": "ok"}})
        client.get_collections.assert_called_once_with()
        # The probe asks for a short client timeout, so a Qdrant that never answers cannot stall it.
        self.assertLessEqual(make_client.call_args.kwargs["timeout"], 1)

    def test_readyz_degrades_when_sqlite_cannot_open(self) -> None:
        unreadable = self.tmp / "no-such-dir" / "ragio.sqlite"
        with mock.patch.object(config, "DB_PATH", unreadable), \
                mock.patch.object(index, "qdrant", return_value=fake_qdrant()):
            resp = self.client.get("/readyz")

        self.assertEqual(resp.status_code, 503)
        body = resp.json()
        self.assertEqual(body["status"], "degraded")
        self.assertNotEqual(body["checks"]["db"], "ok")
        self.assertEqual(body["checks"]["qdrant"], "ok")

    def test_version_reports_the_release_version_commit_and_build_time(self) -> None:
        build = {"GIT_SHA": "0123abc", "BUILT_AT": "2026-11-15T12:00:00Z"}
        with mock.patch.dict(os.environ, build):
            resp = self.client.get("/version")

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers.get("cache-control"), "no-store")
        body = resp.json()
        self.assertEqual(body["name"], "ragio-lab")
        self.assertEqual(body["version"], ragio.__version__)
        self.assertEqual(body["version"], app_module.app.openapi()["info"]["version"])
        self.assertEqual(body["commit"], "0123abc")
        self.assertEqual(body["built_at"], "2026-11-15T12:00:00Z")

    def test_version_commit_and_built_at_are_null_when_unset(self) -> None:
        unset = {k: v for k, v in os.environ.items() if k not in ("GIT_SHA", "BUILT_AT")}
        with mock.patch.dict(os.environ, unset, clear=True):
            resp = self.client.get("/version")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIsNone(body["commit"])
        self.assertIsNone(body["built_at"])

    def test_version_constant_is_managed_by_release_please(self) -> None:
        manifest = json.loads((REPO / ".release-please-manifest.json").read_text())
        release_config = json.loads((REPO / "release-please-config.json").read_text())
        marked = [line for line in (REPO / "ragio" / "__init__.py").read_text().splitlines()
                  if "x-release-please-version" in line]

        self.assertEqual(ragio.__version__, manifest["."])
        self.assertIn("ragio/__init__.py", release_config["packages"]["."].get("extra-files", []))
        self.assertEqual(len(marked), 1)
        self.assertIn(f'"{ragio.__version__}"', marked[0])


if __name__ == "__main__":
    unittest.main()
