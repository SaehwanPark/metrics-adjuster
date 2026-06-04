"""Run an end-to-end demonstration with generated data.

This script is intentionally outside ``src`` so it remains a user-facing demo
rather than package code.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from metrics_adjuster import (
  CalibrationConfig,
  ColumnSpec,
  DensityRatioConfig,
  MetricConfig,
  MetricName,
  adjusted_metrics,
)
from metrics_adjuster.synthetic import generate_synthetic_metrics_data


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Run synthetic adjusted-metrics integration demo",
  )
  parser.add_argument("--output-dir", type=Path, default=Path("demo_outputs"))
  parser.add_argument("--n", type=int, default=600)
  parser.add_argument("--seed", type=int, default=2026)
  return parser


def main() -> None:
  args = build_parser().parse_args()
  args.output_dir.mkdir(parents=True, exist_ok=True)

  data = generate_synthetic_metrics_data(n=args.n, seed=args.seed)
  data.to_csv(args.output_dir / "synthetic_metrics_data.csv", index=False)

  config = MetricConfig(
    columns=ColumnSpec(group="group", response="outcome", risk="risk", id="patient_id"),
    ref_group="ref",
    quantiles=(0.2, 0.4, 0.6, 0.8),
    metrics=tuple(MetricName),
    calibration=CalibrationConfig(degree=2, cv=False),
    density_ratio=DensityRatioConfig(degree=1, cv=False),
    random_state=args.seed,
  )
  result = adjusted_metrics(data, config)
  for metric_name, frame in result.metrics.items():
    frame.to_csv(args.output_dir / f"{metric_name}.csv", index=False)
    print(f"\n{metric_name}")
    print(frame.head().to_string(index=False))


if __name__ == "__main__":
  main()
