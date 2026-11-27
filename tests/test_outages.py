"""Ask flow under dependency outages (issue #21).

Runs offline: retrieval and the LLM are patched, so no Qdrant, Ollama or corpus is needed.
A dependency failure (Qdrant or Ollama unreachable) is a 503 with the reason in `detail` on
the JSON endpoints, and an `error` event followed by `done` on the SSE stream. A request
that fails validation is still a 422.
"""
from __future__ import annotations

import json
import unittest
from unittest import mock

from fastapi.testclient import TestClient

import app as app_module
from ragio import rag

REFUSED = ConnectionError("[Errno 111] Connection refused")


def sse_events(body: str) -> list[dict]:
    return [json.loads(chunk[len("data: "):]) for chunk in body.split("\n\n") if chunk.startswith("data: ")]


def one_hit() -> list[rag.EpisodeHit]:
    passage = rag.Passage(episode_id="ep1", title="Placebo", published="2007-05-17", page_url="",
                          start=0.0, end=60.0, text="some transcript", score=0.9, lo_idx=0, hi_idx=1)
    return [rag.EpisodeHit(episode_id="ep1", title="Placebo", published="2007-05-17", page_url="",
                           score=0.9, passages=[passage])]


class OutageTests(unittest.TestCase):
    def setUp(self) -> None:
        # raise_server_exceptions=False: an unhandled error shows up as the 500 a browser would get.
        self.client = TestClient(app_module.app, raise_server_exceptions=False)

    def test_search_and_ask_are_503_with_the_reason_when_retrieval_is_down(self) -> None:
        with mock.patch.object(rag, "retrieve", side_effect=REFUSED):
            search = self.client.get("/api/search", params={"q": "hello"})
            ask = self.client.post("/api/ask", json={"question": "hello"})

        for resp in (search, ask):
            with self.subTest(url=str(resp.url)):
                self.assertEqual(resp.status_code, 503)
                self.assertIn("ConnectionError: [Errno 111] Connection refused", resp.json()["detail"])

    def test_stream_ends_with_error_then_done_when_retrieval_is_down(self) -> None:
        with mock.patch.object(rag, "retrieve", side_effect=REFUSED):
            resp = self.client.post("/api/ask/stream", json={"question": "hello"})

        self.assertEqual(resp.status_code, 200)
        events = sse_events(resp.text)
        self.assertEqual([e["type"] for e in events], ["error", "done"])
        self.assertIn("ConnectionError", events[0]["message"])

    def test_stream_ends_with_error_then_done_when_the_llm_fails_mid_answer(self) -> None:
        def stream_complete(prompt):
            yield mock.Mock(delta="Hel")
            raise ConnectionError("ollama went away")

        fake_settings = mock.Mock()
        fake_settings.llm.stream_complete.side_effect = stream_complete
        with mock.patch.object(rag, "retrieve", return_value=one_hit()), \
                mock.patch.object(rag, "Settings", fake_settings):
            resp = self.client.post("/api/ask/stream", json={"question": "hello"})

        self.assertEqual(resp.status_code, 200)
        events = sse_events(resp.text)
        self.assertEqual([e["type"] for e in events], ["sources", "token", "error", "done"])
        self.assertEqual(events[1]["text"], "Hel")
        self.assertIn("ollama went away", events[2]["message"])

    def test_validation_errors_are_still_422(self) -> None:
        with mock.patch.object(rag, "retrieve", side_effect=AssertionError("must not be called")):
            short = self.client.get("/api/search", params={"q": "h"})
            missing = self.client.post("/api/ask", json={})

        self.assertEqual(short.status_code, 422)
        self.assertEqual(missing.status_code, 422)

    def test_a_bug_is_still_a_500_not_a_503(self) -> None:
        with mock.patch.object(rag, "retrieve", side_effect=KeyError("idx")):
            resp = self.client.get("/api/search", params={"q": "hello"})

        self.assertEqual(resp.status_code, 500)


if __name__ == "__main__":
    unittest.main()
