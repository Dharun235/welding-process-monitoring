# Code guidelines

Project code is research software. Keep changes easy to trace to a processing stage or experiment.

## Structure

- Keep image-analysis code under `image_analysis_app/scripts/`.
- Keep data preparation, model training, and evaluation under `predictive_modeling/scripts/`.
- Keep generated data, plots, checkpoints, and local datasets outside commits.
- Put shared defaults in the relevant `config.py`; avoid machine-specific absolute paths.

## Python

- Use Python 3.10+ type hints where practical.
- Add a module docstring and docstrings to public functions, classes, and CLI entry points.
- Keep private helpers small and name them with a leading underscore.
- Prefer explicit `Path` objects and deterministic sorting for file discovery.
- Validate input columns, shapes, ranges, and missing files at pipeline boundaries.
- Preserve units and timestamp meaning in feature names and output columns.
- Do not silently change anomaly handling, split logic, or model defaults; record such changes in the commit message and report.

## Research integrity

- Do not commit ESAB data, production images, credentials, or confidential material.
- Do not claim results unless the corresponding experiment artifact or thesis result exists.
- Record random seeds and resolved configuration for model comparisons.
- Keep preprocessing and evaluation separate to avoid leakage between splits.
- Describe limitations when a pipeline requires unavailable data or modules.

## Before sharing a change

```bash
python -m compileall -q .
git diff --check
```

For data or model changes, also run the affected command on an authorized sample and inspect its output schema. This repository has no automated test suite.
