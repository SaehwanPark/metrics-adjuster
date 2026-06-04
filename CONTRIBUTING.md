# Contributing

Thank you for contributing to `metrics-adjuster`.

## Development Setup

```bash
uv sync --extra dev
```

## Checks

Run these before opening a pull request:

```bash
uv run python -m pytest
uv run ruff check .
uv run python -m mypy src/metrics_adjuster
```

## Project Boundaries

Reusable package behavior belongs in `src/metrics_adjuster/`. CLI and
serialization concerns should stay at package boundaries. Tests should read like
contracts for validation, output schemas, deterministic examples, and CLI
behavior.

The public v1 package does not include historical compatibility modules from
the private development repository.
