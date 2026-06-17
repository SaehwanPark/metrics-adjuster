# CHANGELOG

All notable changes to this project will be documented in this file.

## Unreleased

## [1.0.1] - 2026-06-17

### Changed

- Simplified PyPI-first installation docs across `README.md`,
  `deployment/public_docs/README.md`, and `deployment/public_docs/QUICKSTART.md`.

## [1.0.0] - 2026-06-17

### Added

- Added GPLv3 licensing metadata and public-release deployment tooling.
- Added public documentation templates under `deployment/public_docs/`.
- Added machine-readable public repository curation config and publish scripts.
- Added root operational project docs: `SPEC.md`, `ARCHITECTURE.md`, and `CHANGELOG.md` for spec-driven development workflow.
- Added tracked compliance artifact bundle under `docs/artifacts/*/spec-fp-compliance-20260520.yaml`.
- Added `ReportConfig`, `adjusted_metrics_report(...)`, and report component APIs for self-contained HTML reporting.
- Added CLI `--report`, `--report-title`, and `--report-max-cutoff-lines` options for `run` and `demo`.
- Added `scripts/prepare_sampled_report_input.py` for bounded real-data report review runs that prepare ignored sampled inputs for the public CLI.
- Added integration coverage for parquet-backed `run` flows and sampled report workflows.
- Added YAML-configurable report labels and plot scale through `--report-config-yaml` and `ReportConfig`.
- Added `scripts/va_can_report_config.yml` for the sampled VA CAN report review.

### Changed

- Moved unsupported historical scripts and import shims under `legacy/`.
- Updated sampled evaluation result READMEs to include copy-pasteable reproduction commands.
- Changed CLI metric defaults for `run` and `demo` to `aTPR`; explicit `--metrics aTPR,aPPV,aNB,aHR` still computes all supported metrics.
- Reworked sampled report reproduction so metric CSVs, `bootstrap.csv`, and `report.html` are produced through `metrics-adjuster run --report`, with sampled row-level outputs constrained to `data/generated/`.
- Updated architecture documentation to explicitly describe module ownership, data flow, and constraints.
- Extended bootstrap records with original and adjusted values while preserving the existing adjusted `value` field.
- Expanded report HTML with run-summary metadata and optional subtitles to improve report review quality.
- Redesigned report HTML around numbered Table 1/Figure 1/Figure 2 sections, compact per-metric subtables, consistent plot colors, and optional log-odds plot axes.

### Fixed

- Resolved strict mypy return typing in `compute_logit` (`src/metrics_adjuster/core.py`) without changing runtime behavior.
- Declared parquet runtime support with `pyarrow` and fixed CLI reference-group coercion so numeric values like `99` work against numeric group columns.
- Corrected Figure 2 to plot each group density normalized by the reference-group density, including the reference group at 1.0.
