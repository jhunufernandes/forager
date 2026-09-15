# Copyright (c) 2026 Jhunu Fernandes. All rights reserved.
"""Environment-based configuration loader."""

import dataclasses
import os
import pathlib

DEFAULT_LLM_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_LLM_MODEL = "llama3.2"
DEFAULT_LLM_API_KEY = ""
DEFAULT_LLM_MAX_ATTEMPTS = 3
DEFAULT_LLM_TIMEOUT_SECONDS = 180.0
DEFAULT_LLM_USER_PROMPT = "Describe this file in one sentence."
DEFAULT_INPUT_DIR = "input"
DEFAULT_OUTPUT_DIR = "output"

_LLM_MAX_ATTEMPTS = "LLM_MAX_ATTEMPTS"
_LLM_TIMEOUT_SECONDS = "LLM_TIMEOUT_SECONDS"


@dataclasses.dataclass(frozen=True)
class LlmSettings:
    """LLM section loaded from the environment."""

    base_url: str = DEFAULT_LLM_BASE_URL
    model: str = DEFAULT_LLM_MODEL
    api_key: str = DEFAULT_LLM_API_KEY
    max_attempts: int = DEFAULT_LLM_MAX_ATTEMPTS
    timeout_seconds: float = DEFAULT_LLM_TIMEOUT_SECONDS
    user_prompt: str = DEFAULT_LLM_USER_PROMPT


@dataclasses.dataclass(frozen=True)
class ForagerSettings:
    """Forager section loaded from the environment."""

    input_dir: pathlib.Path = pathlib.Path(DEFAULT_INPUT_DIR)
    output_dir: pathlib.Path = pathlib.Path(DEFAULT_OUTPUT_DIR)


def load_llm() -> LlmSettings:
    """Read the LLM section from the environment.

    Returns:
        An :class:`LlmSettings` built from ``LLM_BASE_URL``, ``LLM_MODEL``,
        ``LLM_API_KEY``, ``LLM_MAX_ATTEMPTS``, ``LLM_TIMEOUT_SECONDS`` and
        ``LLM_USER_PROMPT``.
    """
    return LlmSettings(
        base_url=os.environ.get("LLM_BASE_URL", DEFAULT_LLM_BASE_URL),
        model=os.environ.get("LLM_MODEL", DEFAULT_LLM_MODEL),
        api_key=os.environ.get("LLM_API_KEY", DEFAULT_LLM_API_KEY),
        max_attempts=int(os.environ.get(_LLM_MAX_ATTEMPTS, DEFAULT_LLM_MAX_ATTEMPTS)),
        timeout_seconds=float(os.environ.get(_LLM_TIMEOUT_SECONDS, DEFAULT_LLM_TIMEOUT_SECONDS)),
        user_prompt=os.environ.get("LLM_USER_PROMPT", DEFAULT_LLM_USER_PROMPT),
    )


def load_forager() -> ForagerSettings:
    """Read the forager section from the environment.

    Returns:
        A :class:`ForagerSettings` built from ``INPUT_DIR`` and ``OUTPUT_DIR``.
    """
    return ForagerSettings(
        input_dir=pathlib.Path(os.environ.get("INPUT_DIR", DEFAULT_INPUT_DIR)),
        output_dir=pathlib.Path(os.environ.get("OUTPUT_DIR", DEFAULT_OUTPUT_DIR)),
    )
