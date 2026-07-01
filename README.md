# metrics-adjuster

`metrics-adjuster` computes conventional and adjusted group-aware metrics for
binary prediction models. It provides a typed Python API, a command-line
interface, deterministic synthetic demos, self-contained HTML reports, and
optional parquet artifacts for downstream inspection.

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
- pinned release: `python -m pip install "metrics-adjuster==1.1.0"`
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
- `figure_1_calibrated_density.svg` and `figure_2_weight_ratio.svg` when
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

If you use `metrics-adjuster` in published work, cite it as:

APA:

Park, S. (2026). _metrics-adjuster_ (Version 1.1.0) [Computer software]. GitHub. https://github.com/SaehwanPark/metrics-adjuster

AMA:

Park S. metrics-adjuster [computer program]. Version 1.1.0. Published 2026. Accessed June 24, 2026. https://github.com/SaehwanPark/metrics-adjuster

## License

`metrics-adjuster` is distributed under the GNU General Public License v3.0.
