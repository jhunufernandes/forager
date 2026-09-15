# Copyright (c) 2026 Jhunu Fernandes. All rights reserved.
"""Forager: one loop over a folder, four callables — filter, ask, filter, act."""

__all__ = ["Forager", "ForagerSettings", "Llm", "LlmSettings", "load_forager", "load_llm"]

import logging
import time
import typing

import openai

from forager.env import ForagerSettings, LlmSettings, load_forager, load_llm

if typing.TYPE_CHECKING:
    import collections.abc
    import pathlib

_LOGGER = logging.getLogger(__name__)


class Llm:
    """OpenAI-compatible chat client with retries."""

    _SYSTEM_PROMPT = "You read documents and reply with exactly one answer and nothing else."
    _BACKOFF_SECONDS = 3
    _EXHAUSTED_ATTEMPTS = "Failed to query the LLM after {attempts} attempts"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        *,
        max_attempts: int = 3,
        timeout_seconds: float = 180.0,
    ) -> None:
        """Build the client.

        Args:
            base_url: Root URL of the chat completions API.
            model: Model name to request.
            api_key: Bearer token; falls back to a placeholder when empty.
            max_attempts: How many times to retry a failing call.
            timeout_seconds: Per-request timeout.
        """
        self._model = model
        self._max_attempts = max_attempts
        self._client = openai.OpenAI(base_url=base_url, api_key=api_key or "not-needed", timeout=timeout_seconds)

    def ask(self, prompt: str) -> str | None:
        """Ask the model and return its reply, retrying on failure.

        Args:
            prompt: The user prompt to send.

        Returns:
            The model's reply (possibly empty).

        Raises:
            RuntimeError: When every attempt fails.
        """
        last_exception: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": self._SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )
                return response.choices[0].message.content
            except Exception as exception:
                last_exception = exception
                _LOGGER.exception("Attempt %s/%s failed", attempt, self._max_attempts)
                if attempt < self._max_attempts:
                    time.sleep(self._BACKOFF_SECONDS * attempt)
        raise RuntimeError(self._EXHAUSTED_ATTEMPTS.format(attempts=self._max_attempts)) from last_exception


class Forager:
    """Pipeline over an input folder: filter, ask, filter, act."""

    def __init__(
        self,
        file_filter: collections.abc.Callable[[pathlib.Path], bool],
        llm: collections.abc.Callable[[pathlib.Path], str | None],
        result_filter: collections.abc.Callable[[str | None], bool],
        action: collections.abc.Callable[[pathlib.Path, str | None, pathlib.Path], pathlib.Path | None],
        *,
        recursive: bool = False,
    ) -> None:
        """Store the pipeline configuration; :meth:`run` performs the walk.

        Args:
            file_filter: Decides whether a file enters the pipeline.
            llm: Receives the file and returns the model reply (builds the prompt).
            result_filter: Decides whether a response proceeds to *action*.
            action: Executes on the file inside *output_dir*; returns a destination.
            recursive: When true, walk subdirectories as well.
        """
        self._file_filter = file_filter
        self._llm = llm
        self._result_filter = result_filter
        self._action = action
        self._recursive = recursive

    def run(self, input_dir: pathlib.Path, output_dir: pathlib.Path) -> None:
        """Process every file that passes the file filter.

        For each file, in sorted order:

        1. Drop it when ``file_filter(source)`` is false.
        2. Ask the LLM: ``response = llm(source)``.
        3. Drop it when ``result_filter(response)`` is false.
        4. Apply ``destination = action(source, response, output_dir)``.

        A failure on one file never interrupts the run.

        Args:
            input_dir: Folder whose files enter the pipeline.
            output_dir: Folder handed to *action* for outputs; created if missing.

        Raises:
            FileNotFoundError: When *input_dir* does not exist.
            NotADirectoryError: When *input_dir* is not a directory.
        """
        if not input_dir.exists():
            message = f"Input folder '{input_dir}' does not exist"
            raise FileNotFoundError(message)
        if not input_dir.is_dir():
            message = f"Input folder '{input_dir}' is not a directory"
            raise NotADirectoryError(message)
        output_dir.mkdir(parents=True, exist_ok=True)

        files = self._collect(input_dir)
        _LOGGER.info("Found %s file(s) in %s.", len(files), input_dir)
        for source in files:
            self._process(source, output_dir)

    def _collect(self, input_dir: pathlib.Path) -> list[pathlib.Path]:
        """Return the files inside *input_dir*, sorted by path."""
        pattern = "**/*" if self._recursive else "*"
        return sorted((path for path in input_dir.glob(pattern) if path.is_file()), key=str)

    def _process(self, source: pathlib.Path, output_dir: pathlib.Path) -> None:
        """Run the filter/llm/filter/action chain for one file, swallowing failures."""
        try:
            if not self._file_filter(source):
                return
            response = self._llm(source)
            if not self._result_filter(response):
                return
            self._action(source, response, output_dir)
            _LOGGER.info("Processed %s.", source.name)
        except Exception:
            _LOGGER.exception("Processing failed for %s", source.name)
