# metrics-adjuster

`metrics-adjuster` computes conventional and adjusted group-aware metrics for
binary prediction models. It provides a typed Python API, a command-line
interface, deterministic synthetic demos, and self-contained HTML reports.

## Installation

```bash
python -m pip install metrics-adjuster
```

For uv-managed projects:

```bash
uv add metrics-adjuster
```

## Quick Example

```bash
metrics-adjuster demo --output-dir demo_outputs --report
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
  --metrics aTPR,aPPV,aNB,aHR \
  --report
```

The CLI writes one CSV per adjusted metric and, when `--report` is enabled, a
self-contained `report.html`.

## Python API

```python
from metrics_adjuster import ColumnSpec, MetricConfig, adjusted_metrics
from metrics_adjuster.synthetic import generate_synthetic_metrics_data

frame = generate_synthetic_metrics_data(n=600, seed=2026)
config = MetricConfig(
  columns=ColumnSpec(group="group", response="outcome", risk="risk"),
  ref_group="ref",
  random_state=2026,
)

result = adjusted_metrics(frame, config)
print(result.metrics["aTPR"])
```

## Documentation

- [Quickstart](QUICKSTART.md)
- [API manual](docs/API_MANUAL.md)
- [CLI manual](docs/CLI_MANUAL.md)
- [Contributing](CONTRIBUTING.md)

## License

`metrics-adjuster` is distributed under the GNU General Public License v3.0.
