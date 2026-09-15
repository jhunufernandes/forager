# Copyright (c) 2026 Jhunu Fernandes. All rights reserved.
"""Tests for forager.Llm with a stubbed OpenAI-compatible client."""

import unittest
from unittest import mock

import forager


def stub_response(content: str) -> mock.Mock:
    """Return a chat completion response stub."""
    return mock.Mock(choices=[mock.Mock(message=mock.Mock(content=content))])


def make_llm(*, max_attempts: int = 3) -> forager.Llm:
    """Return an Llm instance with test defaults."""
    return forager.Llm(
        base_url="https://example.invalid/v1",
        model="test-model",
        max_attempts=max_attempts,
    )


class LlmTests(unittest.TestCase):
    """Tests for :meth:`forager.Llm.ask`."""

    def setUp(self) -> None:
        """Disable the retry sleep during tests."""
        sleep_patcher = mock.patch("forager.time.sleep")
        sleep_patcher.start()
        self.addCleanup(sleep_patcher.stop)

    def patch_client(self, side_effect: object) -> mock.Mock:
        """Patch the OpenAI client factory and return the stubbed client."""
        client = mock.Mock()
        client.chat.completions.create.side_effect = side_effect
        patcher = mock.patch("forager.openai.OpenAI", mock.Mock(return_value=client))
        self.addCleanup(patcher.stop)
        patcher.start()
        return client

    def test_returns_reply_text(self) -> None:
        """A successful call returns the model reply text."""
        client = self.patch_client([stub_response("answer")])
        llm = make_llm()
        self.assertEqual(llm.ask("prompt"), "answer")
        self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_request_carries_model_and_messages(self) -> None:
        """The request carries the configured model and system + user messages."""
        client = self.patch_client([stub_response("answer")])
        make_llm().ask("USER")
        call = client.chat.completions.create.call_args
        self.assertEqual(call.kwargs["model"], "test-model")
        self.assertEqual(call.kwargs["messages"][1], {"role": "user", "content": "USER"})

    def test_retries_then_succeeds(self) -> None:
        """Failing calls are retried until one succeeds."""
        client = self.patch_client([ValueError("boom"), ValueError("boom"), stub_response("recovered")])
        llm = make_llm()
        self.assertEqual(llm.ask("prompt"), "recovered")
        self.assertEqual(client.chat.completions.create.call_count, 3)

    def test_raises_after_exhausting_attempts(self) -> None:
        """Exhausting all attempts raises RuntimeError."""
        client = self.patch_client([ValueError("boom")] * 3)
        llm = make_llm()
        with self.assertRaises(RuntimeError):
            llm.ask("prompt")
        self.assertEqual(client.chat.completions.create.call_count, 3)

    def test_raises_configured_attempt_count(self) -> None:
        """The number of attempts matches the configured max_attempts."""
        client = self.patch_client([ValueError("boom")] * 5)
        llm = make_llm(max_attempts=5)
        with self.assertRaises(RuntimeError):
            llm.ask("prompt")
        self.assertEqual(client.chat.completions.create.call_count, 5)


if __name__ == "__main__":
    unittest.main()
