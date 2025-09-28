# Repository Guidelines

## Project Structure & Module Organization
The repository centers on `main.py`, which drives the synthetic data pipeline by wiring helpers in `src/`. Key modules include `src/funs.py` for assembling the pipeline, model adapters in `src/vllm_inf.py` and `src/api_*_inf.py`, embedding helpers in `src/e5.py`, and reusable utilities in `src/util.py`. Prompt templates live in `prompts/`, generated artifacts default to `data/`, and quick demos sit in `test_run.py` and `test.py`. Configuration defaults sit in `settings.yaml` with environment-specific overrides (`settings_api.yaml`, `settings_ollama.yaml`, etc.), while `requirements.txt` is UTF-16 encoded—use `pip install -r requirements.txt` and a UTF-16 aware editor if you need to modify it.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: create and enter the project virtual environment.
- `pip install -r requirements.txt`: install runtime and tooling dependencies.
- `python test_run.py`: smoke-test vLLM inference with the bundled sample prompts.
- `python main.py --config settings.yaml`: run the full synthetic data generation flow; swap `settings.yaml` for another preset when experimenting.
- `python test.py`: execute the end-to-end pipeline that writes staged outputs under `data/`.

## Coding Style & Naming Conventions
Follow PEP 8 with four-space indentation and snake_case identifiers; reserve PascalCase for configuration namespaces (see `settings.yaml`). Preserve concise Japanese comments where they explain domain context, and add short English notes only when logic is non-obvious. Extend existing type hints and docstrings in `src/util.py`/`src/funs.py` when exposing new helpers. Keep modules importable (no side effects at import time) so scripts like `test_run.py` stay lightweight.

## Testing Guidelines
Keep sample payloads small and deterministic; check generated JSONL into `data/` only when they are fixtures. Run `python test_run.py` after touching any inference backend, and `python test.py` before shipping broader pipeline changes. If you introduce pytest-based suites, place them under `tests/` with files named `test_<feature>.py` and ensure they can run headlessly without GPU access.

## Commit & Pull Request Guidelines
Recent history favors short, descriptive subjects (English or Japanese) such as `OpenAI-API対応` or `use sentence-transformer...`; follow that pattern and start with a verb. Reference related issues (`#123`) and note config or dependency shifts in the commit body. Pull requests should include a concise summary, list affected YAML keys, attach sample CLI output or JSONL paths, and mention any new hardware assumptions.
