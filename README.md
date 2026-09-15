# forager

One loop over a folder: filter, ask an LLM, filter, act.

`forager` is a tiny pipeline library for document/file processing projects.
It walks an input folder and, for each file, runs your four callables:

1. `file_filter(source)` — keep or drop the file.
2. `llm(source)` — your LLM call (builds the prompt, returns the reply).
3. `result_filter(response)` — keep or drop based on the reply.
4. `action(source, response, output_dir)` — rename, copy, create a file, anything.

## Install

```bash
pip install forager
```

## Quick start

```python
import pathlib
import forager

def file_filter(path):
    return path.suffix.lower() == ".pdf"

def llm(path):
    llm = forager.Llm(base_url="https://openrouter.ai/api/v1", model="...", api_key="...")
    return llm.ask(f"Suggest a new file name for '{path.name}'. Reply with only the name.")

def result_filter(response):
    return bool(response)

def action(source, response, output_dir):
    destination = output_dir / f"{response}{source.suffix}"
    source.rename(destination)
    return destination

pipeline = forager.Forager(
    file_filter=file_filter,
    llm=llm,
    result_filter=result_filter,
    action=action,
)
pipeline.run(pathlib.Path("input"), pathlib.Path("output"))
```

`run` returns nothing — files are processed one by one, and a failure on
one file never interrupts the run.

## API

- `Forager(file_filter, llm, result_filter, action, *, recursive=False)` — the pipeline; `.run(input_dir, output_dir)` performs the loop.
- `Llm(base_url, model, api_key="", *, max_attempts=3, timeout_seconds=180.0)` — `.ask(prompt) -> str | None` for any OpenAI-compatible API, with retries.
