# Copyright (c) 2026 Jhunu Fernandes. All rights reserved.
"""Environment-based configuration loader."""

import pathlib
import typing

from pydantic import AfterValidator, Field, HttpUrl, StringConstraints
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_LLM_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_LLM_MODEL = "llama3.2"
DEFAULT_LLM_API_KEY = ""
DEFAULT_LLM_MAX_ATTEMPTS = 3
DEFAULT_LLM_TIMEOUT_SECONDS = 180.0
DEFAULT_LLM_USER_PROMPT = "Describe this file in one sentence."
DEFAULT_INPUT_DIR = "input"
DEFAULT_OUTPUT_DIR = "output"


def _to_path(value: str) -> pathlib.Path:
    """Coerce a validated non-empty string to a path."""
    return pathlib.Path(value)


NonEmptyPath = typing.Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
    AfterValidator(_to_path),
]


class LlmSettings(BaseSettings):
    """LLM section loaded from the environment.

    Reads ``LLM_BASE_URL``, ``LLM_MODEL``, ``LLM_API_KEY``,
    ``LLM_MAX_ATTEMPTS``, ``LLM_TIMEOUT_SECONDS`` and ``LLM_USER_PROMPT``.
    Invalid values fail fast with a :class:`pydantic.ValidationError`.
    """

    model_config = SettingsConfigDict(extra="ignore", env_prefix="LLM_")

    base_url: HttpUrl = HttpUrl(DEFAULT_LLM_BASE_URL)
    model: typing.Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)] = DEFAULT_LLM_MODEL
    api_key: str = DEFAULT_LLM_API_KEY
    max_attempts: int = Field(default=DEFAULT_LLM_MAX_ATTEMPTS, ge=1)
    timeout_seconds: float = Field(default=DEFAULT_LLM_TIMEOUT_SECONDS, gt=0)
    user_prompt: typing.Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)] = (
        DEFAULT_LLM_USER_PROMPT
    )


class ForagerSettings(BaseSettings):
    """Forager section loaded from the environment.

    Reads ``INPUT_DIR`` and ``OUTPUT_DIR``. Invalid values fail fast with a
    :class:`pydantic.ValidationError`.
    """

    model_config = SettingsConfigDict(extra="ignore")

    input_dir: NonEmptyPath = DEFAULT_INPUT_DIR
    output_dir: NonEmptyPath = DEFAULT_OUTPUT_DIR


def load_llm(**overrides: typing.Any) -> LlmSettings:
    """Read the LLM section from the environment.

    Args:
        **overrides: Keyword arguments taking precedence over the environment,
            which in turn takes precedence over the defaults.

    Returns:
        An :class:`LlmSettings` built from ``LLM_BASE_URL``, ``LLM_MODEL``,
        ``LLM_API_KEY``, ``LLM_MAX_ATTEMPTS``, ``LLM_TIMEOUT_SECONDS`` and
        ``LLM_USER_PROMPT``.
    """
    return LlmSettings(**overrides)


def load_forager(**overrides: typing.Any) -> ForagerSettings:
    """Read the forager section from the environment.

    Args:
        **overrides: Keyword arguments taking precedence over the environment,
            which in turn takes precedence over the defaults.

    Returns:
        A :class:`ForagerSettings` built from ``INPUT_DIR`` and ``OUTPUT_DIR``.
    """
    return ForagerSettings(**overrides)
