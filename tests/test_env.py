# Copyright (c) 2026 Jhunu Fernandes. All rights reserved.
"""Tests for forager.env: validated environment loading."""

import os
import typing
import unittest
from unittest import mock

from pydantic import HttpUrl, ValidationError

import forager
from forager import env

if typing.TYPE_CHECKING:
    import collections.abc

_VARS = (
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_API_KEY",
    "LLM_MAX_ATTEMPTS",
    "LLM_TIMEOUT_SECONDS",
    "LLM_USER_PROMPT",
    "INPUT_DIR",
    "OUTPUT_DIR",
)


class EnvTestCase(unittest.TestCase):
    """Base test case that isolates the forager environment variables."""

    def setUp(self) -> None:
        """Remove every forager variable so tests see only what they set."""
        patcher = mock.patch.dict("os.environ", {}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        for key in _VARS:
            os.environ.pop(key, None)

    def set_env(self, **values: str) -> None:
        """Set forager environment variables for the current test."""
        os.environ.update(values)

    def assert_load_error(
        self,
        loader: collections.abc.Callable[[], object],
        env: dict[str, str],
        location: str,
        kind: str,
    ) -> None:
        """Assert that loading with *env* raises a ValidationError for *location* and *kind*."""
        self.set_env(**env)
        with self.assertRaises(ValidationError) as ctx:
            loader()
        errors = ctx.exception.errors()
        self.assertEqual(errors[0]["loc"][0], location)
        self.assertEqual(errors[0]["type"], kind)


class LoadLlmTests(EnvTestCase):
    """Tests for :func:`forager.load_llm`."""

    def test_defaults_when_unset(self) -> None:
        """Unset variables fall back to documented defaults."""
        settings = forager.load_llm()
        self.assertEqual(settings.base_url, HttpUrl(env.DEFAULT_LLM_BASE_URL))
        self.assertEqual(settings.model, env.DEFAULT_LLM_MODEL)
        self.assertEqual(settings.api_key, env.DEFAULT_LLM_API_KEY)
        self.assertEqual(settings.max_attempts, env.DEFAULT_LLM_MAX_ATTEMPTS)
        self.assertEqual(settings.timeout_seconds, env.DEFAULT_LLM_TIMEOUT_SECONDS)
        self.assertEqual(settings.user_prompt, env.DEFAULT_LLM_USER_PROMPT)

    def test_overrides_are_applied(self) -> None:
        """Set variables override the defaults."""
        self.set_env(
            LLM_BASE_URL="https://example.com/v1",
            LLM_MODEL="gpt",
            LLM_MAX_ATTEMPTS="5",
            LLM_TIMEOUT_SECONDS="1.5",
        )
        settings = forager.load_llm()
        self.assertEqual(settings.base_url, HttpUrl("https://example.com/v1"))
        self.assertEqual(settings.model, "gpt")
        self.assertEqual(settings.max_attempts, 5)
        self.assertEqual(settings.timeout_seconds, 1.5)

    def test_kwargs_override_the_environment(self) -> None:
        """Keyword arguments take precedence over both the environment and the defaults."""
        self.set_env(LLM_MODEL="env-model", LLM_MAX_ATTEMPTS="5")
        settings = forager.load_llm(model="flag-model", max_attempts=7)
        self.assertEqual(settings.model, "flag-model")
        self.assertEqual(settings.max_attempts, 7)
        self.assertEqual(settings.base_url, HttpUrl(env.DEFAULT_LLM_BASE_URL))

    def test_kwargs_beat_env_and_env_beats_defaults(self) -> None:
        """Full precedence chain: kwargs, then environment, then defaults."""
        self.set_env(LLM_MAX_ATTEMPTS="5")
        settings = forager.load_llm(model="flag-model")
        self.assertEqual(settings.model, "flag-model")
        self.assertEqual(settings.max_attempts, 5)
        self.assertEqual(settings.timeout_seconds, env.DEFAULT_LLM_TIMEOUT_SECONDS)

    def test_invalid_max_attempts_raises(self) -> None:
        """A non-integer LLM_MAX_ATTEMPTS raises a ValidationError for max_attempts."""
        self.assert_load_error(
            forager.load_llm,
            {"LLM_MAX_ATTEMPTS": "abc"},
            "max_attempts",
            "int_parsing",
        )

    def test_zero_max_attempts_raises(self) -> None:
        """LLM_MAX_ATTEMPTS below 1 raises a ValidationError for max_attempts."""
        self.assert_load_error(
            forager.load_llm,
            {"LLM_MAX_ATTEMPTS": "0"},
            "max_attempts",
            "greater_than_equal",
        )

    def test_invalid_timeout_raises(self) -> None:
        """A non-numeric LLM_TIMEOUT_SECONDS raises a ValidationError for timeout_seconds."""
        self.assert_load_error(
            forager.load_llm,
            {"LLM_TIMEOUT_SECONDS": "soon"},
            "timeout_seconds",
            "float_parsing",
        )

    def test_non_positive_timeout_raises(self) -> None:
        """LLM_TIMEOUT_SECONDS <= 0 raises a ValidationError for timeout_seconds."""
        self.assert_load_error(
            forager.load_llm,
            {"LLM_TIMEOUT_SECONDS": "0"},
            "timeout_seconds",
            "greater_than",
        )

    def test_bad_base_url_raises(self) -> None:
        """A base URL without an http(s) scheme raises a ValidationError for base_url."""
        self.assert_load_error(
            forager.load_llm,
            {"LLM_BASE_URL": "example.com/v1"},
            "base_url",
            "url_parsing",
        )

    def test_empty_string_value_raises(self) -> None:
        """An explicitly empty variable raises a ValidationError for model."""
        self.assert_load_error(
            forager.load_llm,
            {"LLM_MODEL": ""},
            "model",
            "string_too_short",
        )

    def test_whitespace_only_value_raises(self) -> None:
        """A whitespace-only variable raises a ValidationError for model."""
        self.assert_load_error(
            forager.load_llm,
            {"LLM_MODEL": "   "},
            "model",
            "string_too_short",
        )

    def test_values_are_stripped(self) -> None:
        """Surrounding whitespace is stripped from values."""
        self.set_env(LLM_MODEL="  gpt  ")
        self.assertEqual(forager.load_llm().model, "gpt")


class LoadForagerTests(EnvTestCase):
    """Tests for :func:`forager.load_forager`."""

    def test_defaults_when_unset(self) -> None:
        """Unset variables fall back to documented defaults."""
        settings = forager.load_forager()
        self.assertEqual(str(settings.input_dir), env.DEFAULT_INPUT_DIR)
        self.assertEqual(str(settings.output_dir), env.DEFAULT_OUTPUT_DIR)

    def test_overrides_are_applied(self) -> None:
        """Set variables override the defaults."""
        self.set_env(INPUT_DIR="custom-in", OUTPUT_DIR="custom-out")
        settings = forager.load_forager()
        self.assertEqual(str(settings.input_dir), "custom-in")
        self.assertEqual(str(settings.output_dir), "custom-out")

    def test_kwargs_override_the_environment(self) -> None:
        """Keyword arguments take precedence over both the environment and the defaults."""
        self.set_env(INPUT_DIR="env-in")
        settings = forager.load_forager(input_dir="flag-in", output_dir="flag-out")
        self.assertEqual(str(settings.input_dir), "flag-in")
        self.assertEqual(str(settings.output_dir), "flag-out")

    def test_empty_input_dir_raises(self) -> None:
        """An empty INPUT_DIR raises a ValidationError for input_dir."""
        self.assert_load_error(
            forager.load_forager,
            {"INPUT_DIR": ""},
            "input_dir",
            "string_too_short",
        )

    def test_empty_output_dir_raises(self) -> None:
        """An empty OUTPUT_DIR raises a ValidationError for output_dir."""
        self.assert_load_error(
            forager.load_forager,
            {"OUTPUT_DIR": ""},
            "output_dir",
            "string_too_short",
        )

    def test_whitespace_only_dir_raises(self) -> None:
        """A whitespace-only INPUT_DIR raises a ValidationError for input_dir."""
        self.assert_load_error(
            forager.load_forager,
            {"INPUT_DIR": "   "},
            "input_dir",
            "string_too_short",
        )


if __name__ == "__main__":
    unittest.main()
