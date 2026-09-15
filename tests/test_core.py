# Copyright (c) 2026 Jhunu Fernandes. All rights reserved.
"""Tests for forager: the filter/llm/filter/action loop."""

import pathlib
import tempfile
import typing
import unittest

import forager

if typing.TYPE_CHECKING:
    import collections.abc


def touch(folder: pathlib.Path, name: str) -> pathlib.Path:
    """Create an empty file in *folder* and return its path."""
    path = folder / name
    path.touch()
    return path


class RunTests(unittest.TestCase):
    """Tests for :meth:`forager.Forager.run`."""

    def setUp(self) -> None:
        """Create a fresh temporary input folder and output path."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.input_dir = pathlib.Path(tmp.name) / "input"
        self.output_dir = pathlib.Path(tmp.name) / "output"
        self.input_dir.mkdir()

    def make_pipeline(
        self,
        *,
        file_filter: collections.abc.Callable[[pathlib.Path], bool] | None = None,
        llm: collections.abc.Callable[[pathlib.Path], str | None] | None = None,
        result_filter: collections.abc.Callable[[str | None], bool] | None = None,
        action: (
            collections.abc.Callable[[pathlib.Path, str | None, pathlib.Path], pathlib.Path | None] | None
        ) = None,
        recursive: bool = False,
    ) -> forager.Forager:
        """Return a pipeline with passing filters and a rename action, unless overridden."""
        return forager.Forager(
            file_filter=file_filter or (lambda _source: True),
            llm=llm or (lambda source: f"answer-{source.name}"),
            result_filter=result_filter or (lambda _response: True),
            action=action or (lambda _source, response, output_dir: output_dir / str(response)),
            recursive=recursive,
        )

    def test_processes_every_file_end_to_end(self) -> None:
        """Every file is handed to the action in sorted order."""
        first = touch(self.input_dir, "a.txt")
        second = touch(self.input_dir, "b.txt")
        processed: list[pathlib.Path] = []

        def action(source: pathlib.Path, _response: str | None, _output_dir: pathlib.Path) -> None:
            processed.append(source)

        self.make_pipeline(action=action).run(self.input_dir, self.output_dir)
        self.assertEqual(processed, [first, second])
        self.assertTrue(self.output_dir.is_dir())

    def test_file_filter_skips_file_before_llm(self) -> None:
        """A file rejected by the file filter never reaches the LLM."""
        touch(self.input_dir, "a.txt")
        calls: list[str] = []
        self.make_pipeline(
            file_filter=lambda source: source.name.startswith("x"),
            llm=lambda source: calls.append(source.name) or "answer",
        ).run(self.input_dir, self.output_dir)
        self.assertEqual(calls, [])

    def test_result_filter_skips_file_before_action(self) -> None:
        """A response rejected by the result filter never reaches the action."""
        touch(self.input_dir, "a.txt")
        applied: list[str] = []
        self.make_pipeline(
            result_filter=lambda _response: False,
            action=lambda source, _response, _output_dir: applied.append(source.name),
        ).run(self.input_dir, self.output_dir)
        self.assertEqual(applied, [])

    def test_action_receives_response_and_output_dir(self) -> None:
        """The action receives the source file, the LLM response, and the output dir."""
        source = touch(self.input_dir, "a.txt")
        seen: dict[str, object] = {}
        self.make_pipeline(
            action=lambda source, response, output_dir: (
                seen.update(source=source, response=response, output_dir=output_dir) or None
            ),
        ).run(self.input_dir, self.output_dir)
        self.assertEqual(seen["source"], source)
        self.assertEqual(seen["response"], "answer-a.txt")
        self.assertEqual(seen["output_dir"], self.output_dir)

    def test_failure_on_one_file_does_not_stop_the_run(self) -> None:
        """A failing action on one file lets the remaining files proceed."""
        touch(self.input_dir, "a.txt")
        touch(self.input_dir, "b.txt")
        processed: list[str] = []

        def action(source: pathlib.Path, _response: str | None, output_dir: pathlib.Path) -> pathlib.Path:
            if source.name == "a.txt":
                message = "boom"
                raise ValueError(message)
            processed.append(source.name)
            return output_dir / "ok"

        self.make_pipeline(action=action).run(self.input_dir, self.output_dir)
        self.assertEqual(processed, ["b.txt"])

    def test_llm_failure_does_not_stop_the_run(self) -> None:
        """A failing LLM call is swallowed instead of aborting the run."""
        touch(self.input_dir, "a.txt")

        def failing_llm(_source: pathlib.Path) -> str | None:
            message = "no llm"
            raise ValueError(message)

        self.make_pipeline(llm=failing_llm).run(self.input_dir, self.output_dir)

    def test_recursive_walks_subdirectories(self) -> None:
        """With recursive enabled, files in subdirectories are processed too."""
        (self.input_dir / "sub").mkdir()
        nested = touch(self.input_dir / "sub", "n.txt")
        processed: list[pathlib.Path] = []
        self.make_pipeline(
            recursive=True,
            action=lambda source, _response, _output_dir: processed.append(source),
        ).run(self.input_dir, self.output_dir)
        self.assertEqual(processed, [nested])

    def test_missing_input_dir_raises(self) -> None:
        """A missing input folder raises FileNotFoundError."""
        self.input_dir.rmdir()
        with self.assertRaises(FileNotFoundError):
            self.make_pipeline().run(self.input_dir, self.output_dir)

    def test_input_dir_not_a_directory_raises(self) -> None:
        """An input folder that is a file raises NotADirectoryError."""
        with tempfile.TemporaryDirectory() as name:
            parent = pathlib.Path(name)
            file = parent / "file.txt"
            file.touch()
            with self.assertRaises(NotADirectoryError):
                forager.Forager(
                    file_filter=lambda _: True,
                    llm=lambda _: None,
                    result_filter=lambda _: True,
                    action=lambda *_: None,
                ).run(file, parent / "out")


if __name__ == "__main__":
    unittest.main()
