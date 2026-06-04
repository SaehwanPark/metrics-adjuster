# API Manual

This manual covers the public Python API for `metrics-adjuster`. For a minimal example, see `QUICKSTART.md`.

## Mental model

`metrics-adjuster` separates reusable computation from IO:

- Build a typed `MetricConfig`.
- Pass a `pandas.DataFrame` and the config to `adjusted_metrics(...)`.
- Receive a `MetricFrames` object containing one result frame per requested adjusted metric.

Each metric output includes both the conventional unadjusted metric and its adjusted companion. For example, the `aTPR` output includes both `TPR` and `aTPR` columns.

## Installation for API users

From a checkout:

```bash
uv sync --extra dev
```

From GitHub inside another project:

```bash
uv add "metrics-adjuster @ git+https://github.com/SaehwanPark/metrics-adjuster.git"
```

With pip:

```bash
python -m pip install metrics-adjuster
```

## Main imports

```python
from metrics_adjuster import (
  BootstrapConfig,
  CalibrationConfig,
  ColumnSpec,
  DensityRatioConfig,
  MetricConfig,
  MetricName,
  ReportConfig,
  adjusted_metrics,
  adjusted_metrics_report,
  compute_adjusted_metrics,
)
```

Synthetic data for examples is available from:

```python
from metrics_adjuster.synthetic import generate_synthetic_metrics_data
```

## Input data contract

The input frame must contain:

| Concept | Example column | Requirements |
| --- | --- | --- |
| group | `group` | Categorical group membership. The configured reference group must be present. |
| response | `outcome` | Binary outcome encoded as `0`/`1` or `False`/`True`. |
| risk | `risk` | Numeric predicted risk score. Calibration assumes probability-like scores when logit transforms are enabled. |
| id | `patient_id` | Optional stable row identifier, used only for optional persisted intermediate outputs. |

Rows with missing values in the configured group, response, or risk columns are dropped before computation.

## Configuration objects

### `ColumnSpec`

Maps semantic roles to column names:

```python
columns = ColumnSpec(
  group="group",
  response="outcome",
  risk="risk",
  id="patient_id",
)
```

`id` may be omitted or set to `None`.

### `MetricName`

Supported adjusted metric names are:

| Enum | Value | Conventional companion |
| --- | --- | --- |
| `MetricName.ATPR` | `aTPR` | `TPR` |
| `MetricName.APPV` | `aPPV` | `PPV` |
| `MetricName.ANB` | `aNB` | `NB` |
| `MetricName.AHR` | `aHR` | `HR` |

### `CalibrationConfig`

Controls within-group calibration of risk scores:

```python
calibration = CalibrationConfig(
  degree=2,
  transform=True,
  cv=False,
  k_folds=5,
)
```

Fields:

| Field | Meaning |
| --- | --- |
| `degree` | Polynomial degree for the internal logistic calibration model. |
| `transform` | Whether to apply a logit transform to risk before fitting. |
| `cv` | Whether to use group-wise cross-fitted predictions. |
| `k_folds` | Number of folds when `cv=True`. |

### `DensityRatioConfig`

Controls density-ratio estimation against the reference group:

```python
density_ratio = DensityRatioConfig(
  degree=1,
  transform=False,
  cv=False,
  k_folds=5,
)
```

Fields mirror `CalibrationConfig`, but apply to the binary classifier used to estimate reference-vs-group density ratios.

### `BootstrapConfig`

Controls optional bootstrap records and summaries:

```python
bootstrap = BootstrapConfig(
  enabled=True,
  iterations=500,
  alpha=0.05,
)
```

When enabled, each metric frame receives bootstrap summary columns for the adjusted metric estimate:

- `bootmean`
- `bootse`
- `boot_lower`
- `boot_upper`
- `n`

The raw long-form bootstrap records are also available as `result.bootstrap`.

### `OutputConfig`

Optional paths for intermediate calibrated risk and density-ratio outputs:

```python
from pathlib import Path
from metrics_adjuster import OutputConfig

output = OutputConfig(
  calibration_path=Path("artifacts/calibration.parquet"),
  density_ratio_path=Path("artifacts/density_ratio.parquet"),
)
```

These outputs are written at the API boundary. The core metric outputs are still returned in memory.

### `MetricConfig`

Top-level config:

```python
config = MetricConfig(
  columns=columns,
  ref_group="ref",
  quantiles=(0.2, 0.4, 0.6, 0.8),
  metrics=(MetricName.ATPR, MetricName.APPV, MetricName.ANB, MetricName.AHR),
  calibration=calibration,
  density_ratio=density_ratio,
  bootstrap=bootstrap,
  output=output,
  random_state=2026,
)
```

`quantiles` must be strictly between 0 and 1. A quantile determines the risk threshold `tau` used to flag high-risk rows with `risk > tau`.

## Main API: `adjusted_metrics(...)`

```python
from metrics_adjuster import (
  CalibrationConfig,
  ColumnSpec,
  DensityRatioConfig,
  MetricConfig,
  MetricName,
  adjusted_metrics,
)
from metrics_adjuster.synthetic import generate_synthetic_metrics_data

frame = generate_synthetic_metrics_data(n=600, seed=2026)
config = MetricConfig(
  columns=ColumnSpec(group="group", response="outcome", risk="risk", id="patient_id"),
  ref_group="ref",
  quantiles=(0.2, 0.4, 0.6, 0.8),
  metrics=(MetricName.ATPR, MetricName.APPV),
  calibration=CalibrationConfig(degree=2),
  density_ratio=DensityRatioConfig(degree=1),
  random_state=2026,
)

result = adjusted_metrics(frame, config)
atpr = result.metrics["aTPR"]
appv = result.metrics["aPPV"]
```

`result` is a `MetricFrames` object:

```python
result.metrics     # dict[str, pandas.DataFrame]
result.bootstrap   # pandas.DataFrame | None
result.as_dict()   # compatibility-shaped dictionary
```

## Report API

Use `adjusted_metrics_report(...)` when you want a polished end-user artifact
instead of only machine-readable metric tables:

```python
from metrics_adjuster import ReportConfig, ReportLabelConfig, adjusted_metrics_report

bundle = adjusted_metrics_report(
  frame,
  config,
  ReportConfig(
    title="Adjusted metrics report",
    x_scale="log_odds",
    labels=ReportLabelConfig(
      columns={"group": "Cohort group"},
      groups={"group": {"ref": "Reference group"}},
      metrics={"aTPR": "True positive rate"},
    ),
    max_cutoff_lines=8,
  ),
)

bundle.html                 # self-contained HTML string
bundle.metric_table         # pandas.DataFrame
bundle.density_figure       # matplotlib.figure.Figure
bundle.weight_ratio_figure  # matplotlib.figure.Figure
bundle.metrics              # MetricFrames used to build the report
```

The reusable `bundle.metric_table` is long-form and includes:

```text
metric,<group column>,quantile,tau,original_metric,original_value,
original_ci_lower,original_ci_upper,adjusted_metric,adjusted_value,
adjusted_ci_lower,adjusted_ci_upper,n_boot
```

Confidence interval fields are populated from bootstrap records when
`BootstrapConfig(enabled=True, ...)` is used. Without bootstrap, the CI fields
remain present and render as unavailable in HTML.

Individual component functions are also exported for custom workflows:

- `metric_comparison_table(...)`
- `calibrated_density_figure(...)`
- `weight_ratio_figure(...)`
- `render_report_html(...)`

The rendered HTML shows compact per-metric subtables under Table 1. Report plots
can use calibrated probability or calibrated log-odds on the x-axis through
`ReportConfig.x_scale`. Figure 2 shows each group density divided by the
reference-group density, so the reference group appears at 1.0. When many
quantiles are requested, cutoff reference lines are capped by
`ReportConfig.max_cutoff_lines`.

## Output frame schema

Each requested adjusted metric produces one frame. The common columns are:

| Column | Meaning |
| --- | --- |
| configured group column | Group value. For example, `group`, `sex`, or `race`. |
| `quantile` | Requested quantile used to choose the risk threshold. |
| `tau` | Risk threshold corresponding to that quantile. |
| conventional metric | One of `TPR`, `PPV`, `NB`, or `HR`. |
| adjusted metric | One of `aTPR`, `aPPV`, `aNB`, or `aHR`. |

Example `aTPR` columns:

```text
group,quantile,tau,TPR,aTPR
```

Example `aNB` columns:

```text
group,quantile,tau,NB,aNB
```

With bootstrap enabled, adjusted-metric bootstrap summary columns are appended.

## Metric definitions

For threshold `tau`, define `high_risk = risk > tau`.

The conventional metrics use observed outcomes and no density-ratio weights:

| Metric | Definition |
| --- | --- |
| `TPR` | `mean(outcome * high_risk) / mean(outcome)` |
| `PPV` | `mean(outcome * high_risk) / mean(high_risk)` |
| `NB` | `mean(high_risk * outcome) - mean(high_risk * (1 - outcome)) * tau / (1 - tau)` |
| `HR` | `mean(high_risk)` |

The adjusted metrics use calibrated risk (`cal_risk`) and density ratios (`dens_ratio`):

| Metric | Definition |
| --- | --- |
| `aTPR` | `mean(cal_risk * high_risk * dens_ratio) / mean(cal_risk * dens_ratio)` |
| `aPPV` | `mean(cal_risk * high_risk * dens_ratio) / mean(high_risk * dens_ratio)` |
| `aNB` | `(mean(high_risk * cal_risk * dens_ratio) - mean(high_risk * (1 - cal_risk) * dens_ratio) * tau / (1 - tau)) / mean(dens_ratio)` |
| `aHR` | `mean(high_risk * dens_ratio) / mean(dens_ratio)` |

Undefined ratios return `NaN` rather than raising during metric computation.

## Quantile source masks

By default, thresholds are computed from the full cleaned data frame. You can compute thresholds from a subset by passing `quantiles_from`.

As a boolean `Series`:

```python
mask = frame["group"].eq("ref")
result = adjusted_metrics(frame, config, quantiles_from=mask)
```

As a callable:

```python
result = adjusted_metrics(
  frame,
  config,
  quantiles_from=lambda df: df["group"].eq("ref"),
)
```

A provided mask must have boolean dtype and exactly match the frame index. This avoids silent reindexing.

## Compatibility API

`compute_adjusted_metrics(...)` is retained for callers using the older parameter-based interface:

```python
result = compute_adjusted_metrics(
  df=frame,
  group_col="group",
  ref_group="ref",
  response_col="outcome",
  orig_risk_col="risk",
  idvar="patient_id",
  metrics=["aTPR", "aPPV"],
  quantiles=[0.2, 0.4, 0.6, 0.8],
  cal_degree=2,
  dr_degree=1,
  cv=False,
  se_boot=False,
  random_state=2026,
)

atpr = result["metrics"]["aTPR"]
```

Historical `fair_metrics.compute_metrics` and `compute_metrics` import shims
are retained only under `legacy/` in the private development repository for
migration research. They are not part of the supported v1 package.

## Error handling

Common validation errors include:

| Error | Cause |
| --- | --- |
| `missing required columns` | The configured group, response, risk, or id column is absent. |
| `must contain only binary 0/1 values` | Response includes values outside `0`, `1`, `False`, and `True`. |
| `reference group ... is absent` | `ref_group` does not appear in the configured group column. |
| `quantiles must be strictly between 0 and 1` | At least one quantile is `<= 0` or `>= 1`. |
| `quantiles_from mask index must exactly match df.index` | Caller provided a misaligned mask. |

## Reproducibility

Set `random_state` on `MetricConfig` to make fold assignment, density-ratio estimation, and bootstrap sampling reproducible.

## Testing your integration

A contract-style test for your project can assert output shape and column names:

```python
def test_metric_output_contract(input_frame):
  result = adjusted_metrics(input_frame, config)
  atpr = result.metrics["aTPR"]
  assert {"group", "quantile", "tau", "TPR", "aTPR"}.issubset(atpr.columns)
  assert set(atpr["quantile"]) == {0.2, 0.4, 0.6, 0.8}
```
