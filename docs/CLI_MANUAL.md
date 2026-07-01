# CLI Manual

This manual covers the `metrics-adjuster` command-line interface. For a short path through installation and first use, see `QUICKSTART.md`.

## Installation

From a checkout, install the project environment with uv:

```bash
uv sync --extra dev
```

Then run CLI commands through the project environment:

```bash
uv run metrics-adjuster --help
```

Without uv, install the package and run `metrics-adjuster` directly:

```bash
python -m pip install -e '.[dev]'
metrics-adjuster --help
```

## Command overview

```text
metrics-adjuster
├── demo               # generate synthetic data and metric outputs
├── generate-synthetic # write reproducible synthetic data under data/generated/
└── run                # compute metrics for a caller-provided CSV or Parquet table
```

## `metrics-adjuster demo`

The demo command creates synthetic data and computes `aTPR` by default.

```bash
uv run metrics-adjuster demo \
  --output-dir demo_outputs \
  --n 600 \
  --seed 2026 \
  --report
```

Arguments:

| Argument | Required | Default | Meaning |
| --- | --- | --- | --- |
| `--output-dir` | yes | none | Directory where generated data and metric CSVs are written. |
| `--n` | no | `600` | Number of synthetic rows to generate. |
| `--seed` | no | `2026` | Random seed for reproducibility. |
| `--metrics` | no | `aTPR` | Comma-separated adjusted metrics to compute. |
| `--report` | no | disabled | Write `report.html` with report tables and plots. |
| `--report-title` | no | `Synthetic Adjusted Metrics Report` | HTML report title. |
| `--report-max-cutoff-lines` | no | `8` | Maximum cutoff reference lines per plot. |
| `--report-figures` | no | disabled | Write standalone figure files (requires `--report`). |
| `--report-figure-format` | no | `svg` | Figure format for `--report-figures` (`svg` or `png`). |
| `--save-artifacts` | no | disabled | Write `calibration.parquet` and `weights.parquet`. |

Outputs:

```text
demo_outputs/
├── synthetic_metrics_data.csv
├── aTPR.csv
├── calibration.parquet      # with --save-artifacts
├── weights.parquet          # with --save-artifacts
├── figure_1_calibrated_density.svg  # with --report --report-figures
├── figure_2_weight_ratio.svg        # with --report --report-figures
└── report.html              # with --report
```

Each adjusted metric CSV also includes the matching conventional metric column.
Additional metric CSVs are written when requested with `--metrics`.

## `metrics-adjuster run`

The run command computes metrics for a CSV or Parquet table.

```bash
uv run metrics-adjuster run \
  --input path/to/input.csv \
  --output-dir adjusted_metric_outputs \
  --group-col group \
  --ref-group ref \
  --response-col outcome \
  --risk-col risk \
  --id-col patient_id \
  --quantiles 0.2,0.4,0.6,0.8 \
  --metrics aTPR,aPPV,aNB,aHR \
  --cal-degree 2 \
  --dr-degree 1 \
  --seed 2026 \
  --report
```

### Required arguments

| Argument | Meaning |
| --- | --- |
| `--input` | Input `.csv`, `.parquet`, or `.pq` file. |
| `--output-dir` | Directory where one CSV per metric is written. |
| `--group-col` | Column containing group membership. |
| `--ref-group` | Reference group value present in `--group-col`. |
| `--response-col` | Binary observed outcome column. |
| `--risk-col` | Predicted risk score column. |

### Optional arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `--id-col` | omitted | Optional stable row identifier. |
| `--quantiles` | `0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9` | Comma-separated quantiles used to select thresholds. |
| `--thresholds` | omitted | Comma-separated fixed risk thresholds evaluated in addition to quantiles. |
| `--metrics` | `aTPR` | Comma-separated adjusted metrics to compute. |
| `--pairwise-deltas` | disabled | Write `pairwise.csv` with reference-vs-comparison deltas. |
| `--cal-degree` | `2` | Polynomial degree for calibration. |
| `--dr-degree` | `1` | Polynomial degree for density-ratio estimation. |
| `--cv` | disabled | Use group-wise cross-fitting for calibration and density-ratio models. |
| `--bootstrap` | disabled | Add raw bootstrap records and bootstrap summaries. |
| `--n-boot` | `100` | Number of bootstrap iterations when `--bootstrap` is used. |
| `--seed` | omitted | Random seed. |
| `--report` | disabled | Write a self-contained HTML report to the output directory. |
| `--report-title` | `Adjusted Metrics Report` | HTML report title. |
| `--report-max-cutoff-lines` | `8` | Maximum cutoff reference lines per plot. |
| `--report-config-yaml` | omitted | YAML file for report title, labels, and plot x-axis scale. |
| `--report-figures` | disabled | Write standalone figure files (requires `--report`). |
| `--report-figure-format` | `svg` | Figure format for `--report-figures` (`svg` or `png`). |
| `--save-artifacts` | disabled | Write `calibration.parquet` and `weights.parquet`. |

## Input requirements

The input table must include:

| Role | Example | Notes |
| --- | --- | --- |
| group | `group` | Must include the selected reference group. |
| response | `outcome` | Must contain only `0`/`1` or `False`/`True`. |
| risk | `risk` | Numeric predicted risk score. |
| id | `patient_id` | Optional. |

The CLI drops rows with missing group, response, or risk values before running metrics.

## Output files and schemas

The CLI writes one CSV per requested adjusted metric. The file name is the adjusted metric name.

| File | Columns |
| --- | --- |
| `aTPR.csv` | group column, `quantile`, `tau`, `TPR`, `aTPR` |
| `aFPR.csv` | group column, `quantile`, `tau`, `FPR`, `aFPR` |
| `aPPV.csv` | group column, `quantile`, `tau`, `PPV`, `aPPV` |
| `aNPV.csv` | group column, `quantile`, `tau`, `NPV`, `aNPV` |
| `aBSP.csv` | group column, `quantile`, `tau`, `BSP`, `aBSP` |
| `aBSN.csv` | group column, `quantile`, `tau`, `BSN`, `aBSN` |
| `aSP.csv` | group column, `quantile`, `tau`, `SP`, `aSP` |
| `aNB.csv` | group column, `quantile`, `tau`, `NB`, `aNB` |
| `aHR.csv` | group column, `quantile`, `tau`, `HR`, `aHR` |

The group column keeps the same name that you pass to `--group-col`.

For fixed-threshold Xiaoyi-style alignment runs, pass an empty quantile list and
one or more thresholds:

```bash
uv run metrics-adjuster run \
  --input path/to/input.csv \
  --output-dir adjusted_metric_outputs \
  --group-col group \
  --ref-group 0 \
  --response-col outcome \
  --risk-col risk \
  --quantiles "" \
  --thresholds 0.3 \
  --metrics aTPR,aFPR,aPPV,aNPV,aBSP,aBSN,aSP \
  --pairwise-deltas
```

When `--pairwise-deltas` is used, `pairwise.csv` contains `metric`, the group
column, `reference_group`, `quantile`, `tau`, `reference_value`,
`comparison_value`, `adjusted_comparison_value`, `delta`, and `adjusted_delta`.

When `--bootstrap` is enabled, metric CSVs also include adjusted-metric bootstrap summary columns:

```text
bootmean,bootse,boot_lower,boot_upper,n
```

The raw long-form bootstrap records are written to:

```text
bootstrap.csv
```

When `--report` is used, the CLI also writes `report.html`. The report contains:

- Table 1 with compact per-metric subtables of group, threshold, original value,
  and adjusted value with interval when bootstrap is enabled.
- Calibrated probability density plots for the reference and comparison groups.
- Density plots normalized to the reference-group density.

When `--report-figures` is also used, standalone files are written:

```text
figure_1_calibrated_density.svg
figure_2_weight_ratio.svg
```

When `--save-artifacts` is used, row-level parquet artifacts are written:

| File | Columns |
| --- | --- |
| `calibration.parquet` | `cal_risk` plus `id` when `--id-col` is set |
| `weights.parquet` | `dens_ratio` plus `id` when `--id-col` is set |

Confidence interval fields use bootstrap summaries when `--bootstrap` is also
enabled. Without bootstrap, those fields are shown as unavailable.

Use `--report-config-yaml` to customize display labels and plot scale:

```yaml
title: Adjusted metric report
x_scale: log_odds  # probability | log_odds
labels:
  columns:
    Prior1245: Veteran Priority Group
  groups:
    Prior1245:
      "0": Default Priority
      "99": Priority Group 5
  metrics:
    aTPR: True positive rate
```

## Metric selection

`--metrics` accepts adjusted names only:

```bash
--metrics aTPR
--metrics aTPR,aPPV
--metrics aTPR,aPPV,aNB,aHR
```

The conventional companions are automatically included in each output. You do not need to request `TPR`, `PPV`, `NB`, or `HR` separately.

## Examples

### All metrics with default quantiles

```bash
uv run metrics-adjuster run \
  --input cohort.csv \
  --output-dir outputs \
  --group-col race \
  --ref-group White \
  --response-col readmitted \
  --risk-col predicted_risk
```

### One metric at custom quantiles

```bash
uv run metrics-adjuster run \
  --input cohort.parquet \
  --output-dir outputs_atpr \
  --group-col sex \
  --ref-group Female \
  --response-col outcome \
  --risk-col risk \
  --metrics aTPR \
  --quantiles 0.25,0.5,0.75
```

### Cross-fitted estimates

```bash
uv run metrics-adjuster run \
  --input cohort.csv \
  --output-dir outputs_cv \
  --group-col group \
  --ref-group ref \
  --response-col outcome \
  --risk-col risk \
  --cv \
  --seed 2026
```

### Bootstrap uncertainty summaries

```bash
uv run metrics-adjuster run \
  --input cohort.csv \
  --output-dir outputs_boot \
  --group-col group \
  --ref-group ref \
  --response-col outcome \
  --risk-col risk \
  --bootstrap \
  --n-boot 500 \
  --seed 2026
```

### HTML report

```bash
uv run metrics-adjuster run \
  --input cohort.csv \
  --output-dir outputs_report \
  --group-col race \
  --ref-group White \
  --response-col outcome \
  --risk-col risk \
  --metrics aTPR,aPPV \
  --quantiles 0.2,0.4,0.6,0.8 \
  --bootstrap \
  --n-boot 500 \
  --report \
  --report-title "Adjusted metric report"
```

## `metrics-adjuster generate-synthetic`

Write reproducible synthetic row-level data under `data/generated/` without
running metrics:

```bash
uv run metrics-adjuster generate-synthetic \
  --output-dir data/generated/synthetic-metrics-demo \
  --n 600 \
  --seed 2026
```

| Argument | Required | Default | Meaning |
| --- | --- | --- | --- |
| `--output-dir` | yes | none | Must be under `data/generated/`. |
| `--n` | no | `600` | Number of synthetic rows. |
| `--seed` | no | `2026` | Random seed. |

Outputs:

```text
data/generated/synthetic-metrics-demo/
├── synthetic_metrics_data.csv
└── sample.parquet
```

See `results/synthetic-metrics-demo/README.md` for a tracked reproduction guide.
R users can follow `docs/R_CLI_GUIDE.md`.

## Standalone integration script

The repository also includes a demonstration script outside the package source tree:

```bash
uv run python scripts/run_synthetic_integration.py \
  --output-dir demo_outputs \
  --n 600 \
  --seed 2026
```

This is useful for smoke testing the package, CLI-adjacent workflow, and output schemas with synthetic data.

## Sampled report preparation

Prepare row-level sampled data under an ignored generated-data directory:

```bash
uv run python scripts/prepare_sampled_report_input.py \
  --input /path/to/input.parquet \
  --output-dir data/generated/va-can-2019-prior1245-hosp1y-sample50k \
  --group-col Prior1245 \
  --ref-group 0 \
  --response-col Hosp_1y \
  --risk-col pHosp_1y \
  --sample-size 50000 \
  --bootstrap-iterations 25 \
  --seed 20260521 \
  --report-config-yaml scripts/va_can_report_config.yml
```

Then use the public CLI report path:

```bash
uv run metrics-adjuster run \
  --input data/generated/va-can-2019-prior1245-hosp1y-sample50k/sample.parquet \
  --output-dir results/va-can-2019-prior1245-hosp1y-sample50k \
  --group-col Prior1245 \
  --ref-group 0 \
  --response-col Hosp_1y \
  --risk-col pHosp_1y \
  --quantiles 0.1,0.3,0.5,0.7,0.9 \
  --metrics aTPR,aPPV,aNB,aHR \
  --bootstrap \
  --n-boot 25 \
  --seed 20260521 \
  --report \
  --report-config-yaml scripts/va_can_report_config.yml
```

The preparation command rejects sampled-data output paths outside
`data/generated/`. Its generated README mirrors that generated-data path under
`results/` for report output unless `--report-output-dir` is supplied.

## Troubleshooting

### `ModuleNotFoundError: No module named 'metrics_adjuster'`

Run commands through the project environment:

```bash
uv sync --extra dev
uv run metrics-adjuster --help
```

For tests, use `uv run pytest`, not `uvx pytest`, because tests need the local package installed in the project environment.

### `missing required columns`

Check that the column names passed to the CLI match the input table exactly.

### `reference group ... is absent`

Check the spelling and type of `--ref-group`. The value must appear in the group column after the table is read.

### Empty or `NaN` metric values

Some ratios are undefined when a group has no positive outcomes or no high-risk rows at a threshold. Undefined ratios are returned as `NaN`.
