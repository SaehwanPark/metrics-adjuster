"""Command line interface for adjusted metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from metrics_adjuster.api import adjusted_metrics, adjusted_metrics_report
from metrics_adjuster.synthetic import generate_synthetic_metrics_data
from metrics_adjuster.types import (
  BootstrapConfig,
  CalibrationConfig,
  ColumnSpec,
  DensityRatioConfig,
  MetricConfig,
  MetricName,
  ReportConfig,
)


def parse_quantiles(value: str) -> tuple[float, ...]:
  """Parse comma-separated quantiles."""
  return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def parse_metrics(value: str) -> tuple[MetricName, ...]:
  """Parse comma-separated metric names."""
  return tuple(MetricName(item.strip()) for item in value.split(",") if item.strip())


def normalize_scalar(value: Any) -> Any:
  """Convert pandas or NumPy scalars into plain Python values."""
  return value.item() if hasattr(value, "item") else value


def resolve_reference_group(raw_value: str, groups: pd.Series) -> Any:
  """Match a CLI reference-group string to the input column's observed values."""
  observed = [normalize_scalar(value) for value in groups.dropna().unique()]
  if any(value == raw_value for value in observed):
    return raw_value
  matches = [value for value in observed if str(value) == raw_value]
  if len(matches) == 1:
    return matches[0]
  return raw_value


def read_table(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
  """Read CSV or parquet input by extension."""
  if path.suffix.lower() == ".csv":
    return pd.read_csv(path, usecols=columns)
  if path.suffix.lower() in {".parquet", ".pq"}:
    return pd.read_parquet(path, columns=columns)
  raise ValueError("input path must end in .csv, .parquet, or .pq")


def write_metric_outputs(metrics: dict[str, pd.DataFrame], output_dir: Path) -> None:
  """Write one CSV file per metric."""
  output_dir.mkdir(parents=True, exist_ok=True)
  for metric_name, frame in metrics.items():
    frame.to_csv(output_dir / f"{metric_name}.csv", index=False)


def write_report_output(html: str, output_dir: Path) -> None:
  """Write the self-contained HTML report."""
  output_dir.mkdir(parents=True, exist_ok=True)
  (output_dir / "report.html").write_text(html, encoding="utf-8")


def read_report_config_yaml(path: Path) -> dict[str, Any]:
  """Read a YAML report configuration file."""
  loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
  if loaded is None:
    return {}
  if not isinstance(loaded, dict):
    raise ValueError("report YAML must contain a mapping at the top level")
  return loaded


def build_report_config(args: argparse.Namespace) -> ReportConfig:
  """Create report rendering config from CLI defaults plus optional YAML."""
  base = ReportConfig(
    title=args.report_title,
    max_cutoff_lines=args.report_max_cutoff_lines,
  ).model_dump()
  if args.report_config_yaml is not None:
    base.update(read_report_config_yaml(args.report_config_yaml))
  return ReportConfig.model_validate(base)


def build_run_config(args: argparse.Namespace, df: pd.DataFrame) -> MetricConfig:
  """Create validated config for the public ``run`` command."""
  return MetricConfig(
    columns=ColumnSpec(
      group=args.group_col,
      response=args.response_col,
      risk=args.risk_col,
      id=args.id_col,
    ),
    ref_group=resolve_reference_group(args.ref_group, df[args.group_col]),
    quantiles=parse_quantiles(args.quantiles),
    metrics=parse_metrics(args.metrics),
    calibration=CalibrationConfig(degree=args.cal_degree, cv=args.cv),
    density_ratio=DensityRatioConfig(degree=args.dr_degree, cv=args.cv),
    bootstrap=BootstrapConfig(enabled=args.bootstrap, iterations=args.n_boot),
    random_state=args.seed,
  )


def build_parser() -> argparse.ArgumentParser:
  """Build the CLI parser."""
  parser = argparse.ArgumentParser(prog="metrics-adjuster")
  subcommands = parser.add_subparsers(dest="command", required=True)

  run = subcommands.add_parser("run", help="compute adjusted metrics for a table")
  run.add_argument("--input", type=Path, required=True)
  run.add_argument("--output-dir", type=Path, required=True)
  run.add_argument("--group-col", required=True)
  run.add_argument("--ref-group", required=True)
  run.add_argument("--response-col", required=True)
  run.add_argument("--risk-col", required=True)
  run.add_argument("--id-col")
  run.add_argument("--quantiles", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
  run.add_argument("--metrics", default="aTPR")
  run.add_argument("--cal-degree", type=int, default=2)
  run.add_argument("--dr-degree", type=int, default=1)
  run.add_argument("--cv", action="store_true")
  run.add_argument("--bootstrap", action="store_true")
  run.add_argument("--n-boot", type=int, default=100)
  run.add_argument("--seed", type=int, default=None)
  run.add_argument("--report", action="store_true", help="write a self-contained HTML report")
  run.add_argument("--report-title", default="Adjusted Metrics Report")
  run.add_argument("--report-max-cutoff-lines", type=int, default=8)
  run.add_argument("--report-config-yaml", type=Path)

  demo = subcommands.add_parser("demo", help="run on synthetic data")
  demo.add_argument("--output-dir", type=Path, required=True)
  demo.add_argument("--n", type=int, default=600)
  demo.add_argument("--seed", type=int, default=2026)
  demo.add_argument("--metrics", default="aTPR")
  demo.add_argument("--report", action="store_true", help="write a self-contained HTML report")
  demo.add_argument("--report-title", default="Synthetic Adjusted Metrics Report")
  demo.add_argument("--report-max-cutoff-lines", type=int, default=8)
  demo.add_argument("--report-config-yaml", type=Path)
  return parser


def run_command(args: argparse.Namespace) -> None:
  """Execute the run subcommand."""
  columns = [args.group_col, args.response_col, args.risk_col]
  if args.id_col is not None:
    columns.append(args.id_col)
  df = read_table(args.input, columns)
  config = build_run_config(args, df)
  if args.report:
    bundle = adjusted_metrics_report(
      df,
      config,
      build_report_config(args),
    )
    result = bundle.metrics
    write_report_output(bundle.html, args.output_dir)
  else:
    result = adjusted_metrics(df, config)
  write_metric_outputs(result.metrics, args.output_dir)
  if result.bootstrap is not None:
    result.bootstrap.to_csv(args.output_dir / "bootstrap.csv", index=False)


def demo_command(args: argparse.Namespace) -> None:
  """Execute the synthetic demo subcommand."""
  args.output_dir.mkdir(parents=True, exist_ok=True)
  data = generate_synthetic_metrics_data(n=args.n, seed=args.seed)
  data_path = args.output_dir / "synthetic_metrics_data.csv"
  data.to_csv(data_path, index=False)
  config = MetricConfig(
    columns=ColumnSpec(group="group", response="outcome", risk="risk", id="patient_id"),
    ref_group="ref",
    quantiles=(0.2, 0.4, 0.6, 0.8),
    metrics=parse_metrics(args.metrics),
    calibration=CalibrationConfig(degree=2, cv=False),
    density_ratio=DensityRatioConfig(degree=1, cv=False),
    random_state=args.seed,
  )
  if args.report:
    bundle = adjusted_metrics_report(
      data,
      config,
      build_report_config(args),
    )
    result = bundle.metrics
    write_report_output(bundle.html, args.output_dir)
  else:
    result = adjusted_metrics(data, config)
  write_metric_outputs(result.metrics, args.output_dir)


def main() -> None:
  """CLI entry point."""
  parser = build_parser()
  args = parser.parse_args()
  if args.command == "run":
    run_command(args)
  elif args.command == "demo":
    demo_command(args)


if __name__ == "__main__":
  main()
