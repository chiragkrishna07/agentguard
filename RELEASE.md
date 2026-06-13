# Release checklist

Steps to cut a new AgentGuard release. Targets PyPI via Hatchling.

## Pre-flight

- [ ] `pip install -e ".[dev]"` in a clean venv
- [ ] `ruff check agentguard/ tests/ examples/` — clean
- [ ] `mypy agentguard/ --ignore-missing-imports` — clean
- [ ] `pytest tests/unit/` — all green
- [ ] `python -m tests.benchmarks.bench_detection` — strict mode at 100% recall /
      100% precision / 0% FPR on the corpus (no regressions)
- [ ] `CHANGELOG.md` has a dated entry for the new version
- [ ] Version bumped in **both** `pyproject.toml` and `agentguard/__init__.py`
      (they must match)

## Build & verify

- [ ] `python -m build` (produces `dist/*.whl` and `dist/*.tar.gz`)
- [ ] Wheel includes `agentguard/py.typed`:
      `python -c "import zipfile,glob; w=sorted(glob.glob('dist/*.whl'))[-1]; print([n for n in zipfile.ZipFile(w).namelist() if n.endswith('py.typed')])"`
- [ ] `pip install dist/agentguard-*.whl` in a fresh venv, then
      `python -c "import agentguard; print(agentguard.__version__)"`
- [ ] Smoke test the import surface:
      `python -c "from agentguard import Guard, PromptShield, SecretsShield, PIIRedactor, SizeLimit; from agentguard.presets import recommended; recommended(max_usd=None, audit=False)"`

## Publish

- [ ] `twine check dist/*`
- [ ] `twine upload dist/*` (or push a tag if CI publishes)
- [ ] Tag the commit: `git tag vX.Y.Z && git push --tags`
- [ ] Verify the project page renders the README on PyPI

## Notes

- Core install must work with only `pydantic`, `httpx`, `tiktoken`. Presidio and
  ML (`transformers`/`torch`) are optional extras and must stay lazy-imported —
  `python -c "import agentguard"` has to succeed without them installed.
- `dist/` is git-ignored; don't commit build artifacts.
