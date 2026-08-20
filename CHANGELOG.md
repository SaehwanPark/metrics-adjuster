# CHANGELOG

All notable changes to this project will be documented in this file.

## Unreleased

## [1.1.1] - 2026-08-20

### Added

- Added opt-in legacy Xiaoyi-style adjusted metrics: `aFPR`, `aNPV`, `aBSP`,
  `aBSN`, and `aSP`.
- Added fixed-threshold evaluation through `MetricConfig.thresholds` and CLI
  `--thresholds`.
- Added optional pairwise reference-vs-comparison delta outputs through
  `MetricConfig.pairwise` and CLI `--pairwise-deltas`.
- Added opt-in calibrated-unweighted `c*` metric outputs through
  `MetricConfig.include_calibrated_metrics` and CLI
  `--include-calibrated-metrics`, including report tables and DCA curves.
- Added a Xiaoyi-style simulation data generator for alignment tests.
- Added decision curve analysis to HTML reports, with standalone
  `figure_3_standard_subgroup_dca` and
  `figure_4_comparative_model_utility` exports when `--report --report-figures`
  is used.
- Added default `decision_curve_table.csv` export for report DCA, with
  `DecisionCurveConfig.write_csv_artifact` / report YAML opt-out support.
- Added ANB/DCA design documentation that defines threshold probability,
  treat-all prevalence usage, and the `NB`/`cNB`/`aNB` decomposition.

### Changed

- `aNB` is now the plug-in of estimated `aTPR`, `aFPR`, and observed
  reference-group prevalence rather than a separately normalized weighted
  net-benefit mean. Report DCA thresholds original model risk `g(X)` and uses
  one adjusted treat-all curve for every subgroup.
  `decision_curve_table` requires `ref_group`.
- Documented the research and software provenance from Jinbo Chen and Sarah E.
  Hegarty's original work and Hegarty's `fairRisk` R package through the
  internally validated VA Python rewrite that became `metrics-adjuster`, with
  a comparison of the projects' documented interfaces and workflow scope.
- Marked superseded decision-curve artifact notes as historical snapshots so
  current DCA semantics are taken from `docs/adjusted-dca.md` and
  `docs/adjusted-net-benefit.md`.

## [1.1.0] - 2026-06-17

### Added

- Added CLI `--save-artifacts` for `run` and `demo` to write
  `calibration.parquet` and `weights.parquet`.
- Added `OutputConfig.include_intermediates` and optional `MetricFrames.calibrated`
  / `MetricFrames.weighted` for in-memory pipeline inspection.
- Added CLI `--report-figures` and `--report-figure-format` to export standalone
  report figure files alongside `report.html`.
- Added `write_report_figures(...)` for API callers exporting Matplotlib figures.
- Added `metrics-adjuster generate-synthetic` and
  `scripts/prepare_synthetic_demo_data.py` for reproducible synthetic data under
  `data/generated/`.
- Added `docs/R_CLI_GUIDE.md` for calling the CLI from R.
- Added `results/synthetic-metrics-demo/README.md` reproduction guide.

## [1.0.2] - 2026-06-17

### Changed

- PyPI publishing now curates and builds from the public repository snapshot
  instead of the development repository.
- PyPI long descriptions use the public `README.md`, with relative links
  rewritten to the public GitHub repository during packaging.

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
