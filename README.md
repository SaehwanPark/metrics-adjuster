# metrics-adjuster

`metrics-adjuster` computes conventional and adjusted group-aware metrics for
binary prediction models. It provides a typed Python API, a command-line
interface, deterministic synthetic demos, self-contained HTML reports, and
optional parquet artifacts for downstream inspection.

## Research and software provenance

The underlying adjusted-risk-distribution research was originally led by Jinbo
Chen and Sarah E. Hegarty. Hegarty developed the original R implementation,
[`fairRisk`](https://github.com/sarahhegarty/fairRisk), for the aTPR methodology.

Sae-Hwan Park later joined the research team and performed additional VA
analyses with his Python rewrite of that workflow. The rewrite was validated
internally and reviewed and approved by Hegarty for the team's analysis. It
subsequently became the foundation of `metrics-adjuster`, which is maintained as
a separate Python project and does not replace or supersede `fairRisk`.

For scientific provenance, users of the aTPR methodology should cite the
[underlying research article](https://pmc.ncbi.nlm.nih.gov/articles/PMC11838655/)
and acknowledge `fairRisk` as the original R implementation. Cite
`metrics-adjuster` separately when using this Python software.

## How `metrics-adjuster` extends the workflow

`fairRisk` remains the original native R implementation and the methodological
reference for the adjusted error-rate work. Compared with the currently
documented `fairRisk` v1.1.1.1 interface, `metrics-adjuster` exposes additional
capabilities for Python-based analysis and reproducible delivery:

| Area | `fairRisk` | Additional `metrics-adjuster` capability |
| --- | --- | --- |
| Interface | Native R package functions | Typed Python API plus a shell CLI that can also be invoked from R |
| Adjusted metrics | aTPR, aTNR, aPPV, and aNPV | Nine named adjusted outputs: aTPR, aFPR, aPPV, aNPV, aBSP, aBSN, aSP, aNB, and aHR |
| Adjustment decomposition | Calibration and density-ratio estimation support the adjusted estimators | Optional calibrated-unweighted `c*` metrics separate recalibration effects from density-ratio standardization |
| Cutoffs and comparisons | Function-level R analysis | Fixed thresholds or risk-score quantiles, with optional reference-versus-comparison pairwise deltas |
| Uncertainty and intermediates | Original estimator-focused R workflow | Integrated bootstrap summaries plus optional calibrated-risk and density-ratio parquet artifacts |
| Reporting | No report or decision-curve surface is exported by the current package namespace | Self-contained HTML reports, metric tables, density and weight-ratio plots, adjusted net-benefit decision curves, standalone figures, and CSV artifacts |
| Reproducible execution | Installable from the `fairRisk` GitHub repository | PyPI distribution, validated configuration models, CSV/Parquet input, deterministic synthetic demos, and contract-tested CLI workflows |

These additions broaden the operational workflow; they do not make
`metrics-adjuster` a scientific replacement for `fairRisk`. Shared aTPR analyses
should align the calibration model, density-ratio model, thresholds, reference
group, and other estimator settings before comparing results across languages.
Intentional differences in supported metrics, defaults, or output schemas should
be documented rather than treated as implementation disagreement.

## Installation

Install from [PyPI](https://pypi.org/project/metrics-adjuster/) (Python 3.11+):

```bash
python -m pip install metrics-adjuster
```

That one command installs both the `metrics-adjuster` CLI and the Python API.
Parquet input and artifact output work out of the box (`pyarrow` is included).

**Other setups:**

- CLI only, without a project environment: `pipx install metrics-adjuster`
- uv project: `uv add metrics-adjuster`
- conda or mamba env: run the pip command above inside your activated environment
- pinned release: `python -m pip install "metrics-adjuster==1.1.2"`
- source checkout: see [Contributing](CONTRIBUTING.md)

## Quick Example

```bash
metrics-adjuster demo \
  --output-dir demo_outputs \
  --report \
  --save-artifacts \
  --report-figures
```

For caller-provided CSV or Parquet data:

```bash
metrics-adjuster run \
  --input cohort.csv \
  --output-dir adjusted_metric_outputs \
  --group-col group \
  --ref-group ref \
  --response-col outcome \
  --risk-col risk \
  --id-col patient_id \
  --metrics aTPR,aPPV,aNB,aHR \
  --report \
  --save-artifacts \
  --report-figures
```

The CLI writes one CSV per adjusted metric. Optional outputs include:

- `report.html` when `--report` is enabled
- `figure_1_calibrated_density.svg`, `figure_2_weight_ratio.svg`, and
  `figure_3_decision_curve.svg` when
  `--report-figures` is also used
- `calibration.parquet` and `weights.parquet` when `--save-artifacts` is used

## Python API

```python
from pathlib import Path

from metrics_adjuster import ColumnSpec, MetricConfig, OutputConfig, adjusted_metrics
from metrics_adjuster.synthetic import generate_synthetic_metrics_data

frame = generate_synthetic_metrics_data(n=600, seed=2026)
config = MetricConfig(
  columns=ColumnSpec(group="group", response="outcome", risk="risk", id="patient_id"),
  ref_group="ref",
  output=OutputConfig(
    calibration_path=Path("artifacts/calibration.parquet"),
    density_ratio_path=Path("artifacts/weights.parquet"),
    include_intermediates=True,
  ),
  random_state=2026,
)

result = adjusted_metrics(frame, config)
print(result.metrics["aTPR"])
print(result.weighted[["cal_risk", "dens_ratio"]].head())
```

For reports and standalone figures, use `adjusted_metrics_report(...)` and
`write_report_figures(...)`. See the [API manual](docs/API_MANUAL.md).

## Documentation

- [Quickstart](QUICKSTART.md)
- [API manual](docs/API_MANUAL.md)
- [CLI manual](docs/CLI_MANUAL.md)
- [R CLI guide](docs/R_CLI_GUIDE.md)
- [Contributing](CONTRIBUTING.md)

## Citation

If you use the aTPR methodology, cite the
[underlying research article](https://pmc.ncbi.nlm.nih.gov/articles/PMC11838655/)
and acknowledge Sarah E. Hegarty's
[`fairRisk`](https://github.com/sarahhegarty/fairRisk) as the original R
implementation. If you use `metrics-adjuster`, cite this Python software
separately as:

APA:

Park, S. (2026). _metrics-adjuster_ (Version 1.1.2) [Computer software]. GitHub. https://github.com/SaehwanPark/metrics-adjuster

AMA:

Park S. metrics-adjuster [computer program]. Version 1.1.2. Published 2026. Accessed August 20, 2026. https://github.com/SaehwanPark/metrics-adjuster

## License

`metrics-adjuster` is distributed under the GNU General Public License v3.0.
