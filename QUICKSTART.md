# Quickstart

## Install

From [PyPI](https://pypi.org/project/metrics-adjuster/) (Python 3.11+):

```bash
python -m pip install metrics-adjuster
```

In a uv project: `uv add metrics-adjuster`. For CLI-only use:
`pipx install metrics-adjuster`.

## Run the Demo

```bash
metrics-adjuster demo --output-dir demo_outputs --report
```

Expected outputs include synthetic data, adjusted metric CSVs, and
`report.html`.

Add optional artifact and figure export:

```bash
metrics-adjuster demo \
  --output-dir demo_outputs \
  --report \
  --save-artifacts \
  --report-figures
```

This also writes:

```text
demo_outputs/calibration.parquet
demo_outputs/weights.parquet
demo_outputs/figure_1_calibrated_density.svg
demo_outputs/figure_2_weight_ratio.svg
demo_outputs/figure_3_decision_curve.svg
```

Use `--report-figure-format png` when PNG files are preferred.

## Run on Your Own Data

```bash
metrics-adjuster run \
  --input cohort.csv \
  --output-dir adjusted_metric_outputs \
  --group-col group \
  --ref-group ref \
  --response-col outcome \
  --risk-col risk \
  --id-col patient_id \
  --quantiles 0.2,0.4,0.6,0.8 \
  --metrics aTPR,aPPV,aNB,aHR \
  --report \
  --save-artifacts
```

Input data must include a group column, binary response column, predicted risk
column, and the selected reference group. CSV and Parquet inputs are supported.

When `--id-col` is set, artifact parquet files include that identifier column
alongside `cal_risk` or `dens_ratio`.

## Generate Reproducible Synthetic Input

To write synthetic row-level data without running metrics:

```bash
mkdir -p data/generated/synthetic-metrics-demo

metrics-adjuster generate-synthetic \
  --output-dir data/generated/synthetic-metrics-demo \
  --n 600 \
  --seed 2026
```

`--output-dir` must be under `data/generated/`. The command writes
`synthetic_metrics_data.csv` and `sample.parquet`.

Then run metrics on the persisted sample:

```bash
metrics-adjuster run \
  --input data/generated/synthetic-metrics-demo/sample.parquet \
  --output-dir demo_outputs \
  --group-col group \
  --ref-group ref \
  --response-col outcome \
  --risk-col risk \
  --id-col patient_id \
  --report
```

## Use from R

R users can call the CLI with `system2()` or `processx::run()` and read CSV or
parquet outputs back into R. See [R CLI guide](docs/R_CLI_GUIDE.md).

## Read the Full Manuals

- [API Manual](docs/API_MANUAL.md)
- [CLI Manual](docs/CLI_MANUAL.md)
- [R CLI Guide](docs/R_CLI_GUIDE.md)
