"""Command line interface for adjusted metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml

from metrics_adjuster.api import adjusted_metrics, adjusted_metrics_report
from metrics_adjuster.reporting import (
  ReportBundle,
  write_decision_curve_csv,
  write_decision_curve_figures,
  write_report_figures,
)
from metrics_adjuster.synthetic import generate_synthetic_metrics_data
from metrics_adjuster.types import (
  BootstrapConfig,
  CalibrationConfig,
  ColumnSpec,
  DensityRatioConfig,
  MetricConfig,
  MetricName,
  OutputConfig,
  ReportConfig,
)


def parse_quantiles(value: str) -> tuple[float, ...]:
  """Parse comma-separated quantiles."""
  return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def parse_thresholds(value: str | None) -> tuple[float, ...]:
  """Parse optional comma-separated fixed thresholds."""
  if value is None:
    return ()
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


def is_generated_data_path(path: Path) -> bool:
  """Return whether a resolved path is under a data/generated directory."""
  parts = path.expanduser().resolve().parts
  return any(
    current == "data" and next_part == "generated"
    for current, next_part in zip(parts, parts[1:], strict=False)
  )


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


def build_output_config(args: argparse.Namespace, output_dir: Path) -> OutputConfig:
  """Create optional artifact persistence config from CLI flags."""
  if not args.save_artifacts:
    return OutputConfig()
  return OutputConfig(
    calibration_path=output_dir / "calibration.parquet",
    density_ratio_path=output_dir / "weights.parquet",
  )


def build_run_config(
  args: argparse.Namespace,
  df: pd.DataFrame,
  output_dir: Path,
) -> MetricConfig:
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
    thresholds=parse_thresholds(args.thresholds),
    metrics=parse_metrics(args.metrics),
    pairwise=args.pairwise_deltas,
    include_calibrated_metrics=args.include_calibrated_metrics,
    calibration=CalibrationConfig(degree=args.cal_degree, cv=args.cv),
    density_ratio=DensityRatioConfig(degree=args.dr_degree, cv=args.cv),
    bootstrap=BootstrapConfig(enabled=args.bootstrap, iterations=args.n_boot),
    output=build_output_config(args, output_dir),
    random_state=args.seed,
  )


def require_report_for_figures(args: argparse.Namespace) -> None:
  """Reject standalone figure export without an HTML report."""
  if args.report_figures and not args.report:
    raise SystemExit("--report-figures requires --report")


def write_report_bundle_outputs(
  bundle: ReportBundle,
  output_dir: Path,
  *,
  report_config: ReportConfig,
  report_figures: bool,
  report_figure_format: Literal["svg", "png"],
) -> None:
  """Write HTML and optional standalone report figures."""
  write_report_output(bundle.html, output_dir)
  if report_figures:
    write_report_figures(
      bundle,
      output_dir,
      figure_format=report_figure_format,
    )
    write_decision_curve_figures(
      bundle,
      output_dir,
      figure_format=report_figure_format,
    )
  if (
    report_config.decision_curve.write_csv_artifact and bundle.decision_curve_table is not None
  ):
    write_decision_curve_csv(bundle, output_dir)


def add_report_arguments(
  parser: argparse.ArgumentParser,
  *,
  default_report_title: str = "Adjusted Metrics Report",
) -> None:
  """Add shared report flags to a subcommand parser."""
  parser.add_argument(
    "--report",
    action="store_true",
    help="write a self-contained HTML report",
  )
  parser.add_argument("--report-title", default=default_report_title)
  parser.add_argument("--report-max-cutoff-lines", type=int, default=8)
  parser.add_argument("--report-config-yaml", type=Path)
  parser.add_argument(
    "--report-figures",
    action="store_true",
    help="write standalone report figure files (requires --report)",
  )
  parser.add_argument(
    "--report-figure-format",
    choices=("svg", "png"),
    default="svg",
    help="file format for --report-figures",
  )


def add_artifact_arguments(parser: argparse.ArgumentParser) -> None:
  """Add shared artifact persistence flags to a subcommand parser."""
  parser.add_argument(
    "--save-artifacts",
    action="store_true",
    help="write calibration.parquet and weights.parquet under --output-dir",
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
  run.add_argument(
    "--thresholds",
    default=None,
    help="comma-separated fixed risk thresholds to evaluate in addition to quantiles",
  )
  run.add_argument("--metrics", default="aTPR")
  run.add_argument(
    "--pairwise-deltas",
    action="store_true",
    help="write pairwise.csv with reference-vs-comparison deltas",
  )
  run.add_argument(
    "--include-calibrated-metrics",
    action="store_true",
    help="include calibrated-unweighted c* metric columns and report curves",
  )
  run.add_argument("--cal-degree", type=int, default=2)
  run.add_argument("--dr-degree", type=int, default=1)
  run.add_argument("--cv", action="store_true")
  run.add_argument("--bootstrap", action="store_true")
  run.add_argument("--n-boot", type=int, default=100)
  run.add_argument("--seed", type=int, default=None)
  add_report_arguments(run)
  add_artifact_arguments(run)

  demo = subcommands.add_parser("demo", help="run on synthetic data")
  demo.add_argument("--output-dir", type=Path, required=True)
  demo.add_argument("--n", type=int, default=600)
  demo.add_argument("--seed", type=int, default=2026)
  demo.add_argument("--metrics", default="aTPR")
  demo.add_argument(
    "--include-calibrated-metrics",
    action="store_true",
    help="include calibrated-unweighted c* metric columns and report curves",
  )
  add_report_arguments(demo, default_report_title="Synthetic Adjusted Metrics Report")
  add_artifact_arguments(demo)

  generate = subcommands.add_parser(
    "generate-synthetic",
    help="write reproducible synthetic data under data/generated/",
  )
  generate.add_argument("--output-dir", type=Path, required=True)
  generate.add_argument("--n", type=int, default=600)
  generate.add_argument("--seed", type=int, default=2026)
  return parser


def run_command(args: argparse.Namespace) -> None:
  """Execute the run subcommand."""
  require_report_for_figures(args)
  args.output_dir.mkdir(parents=True, exist_ok=True)
  columns = [args.group_col, args.response_col, args.risk_col]
  if args.id_col is not None:
    columns.append(args.id_col)
  df = read_table(args.input, columns)
  config = build_run_config(args, df, args.output_dir)
  if args.report:
    report_config = build_report_config(args)
    bundle = adjusted_metrics_report(
      df,
      config,
      report_config,
    )
    result = bundle.metrics
    write_report_bundle_outputs(
      bundle,
      args.output_dir,
      report_config=report_config,
      report_figures=args.report_figures,
      report_figure_format=args.report_figure_format,
    )
  else:
    result = adjusted_metrics(df, config)
  write_metric_outputs(result.metrics, args.output_dir)
  if result.pairwise is not None:
    result.pairwise.to_csv(args.output_dir / "pairwise.csv", index=False)
  if result.bootstrap is not None:
    result.bootstrap.to_csv(args.output_dir / "bootstrap.csv", index=False)


def demo_command(args: argparse.Namespace) -> None:
  """Execute the synthetic demo subcommand."""
  require_report_for_figures(args)
  args.output_dir.mkdir(parents=True, exist_ok=True)
  data = generate_synthetic_metrics_data(n=args.n, seed=args.seed)
  data_path = args.output_dir / "synthetic_metrics_data.csv"
  data.to_csv(data_path, index=False)
  data.to_csv(args.output_dir / "metrics-adjuster-synthetic-data.csv", index=False)
  config = MetricConfig(
    columns=ColumnSpec(group="group", response="outcome", risk="risk", id="patient_id"),
    ref_group="ref",
    quantiles=(0.2, 0.4, 0.6, 0.8),
    metrics=parse_metrics(args.metrics),
    include_calibrated_metrics=args.include_calibrated_metrics,
    calibration=CalibrationConfig(degree=2, cv=False),
    density_ratio=DensityRatioConfig(degree=1, cv=False),
    output=build_output_config(args, args.output_dir),
    random_state=args.seed,
  )
  if args.report:
    report_config = build_report_config(args)
    bundle = adjusted_metrics_report(
      data,
      config,
      report_config,
    )
    result = bundle.metrics
    write_report_bundle_outputs(
      bundle,
      args.output_dir,
      report_config=report_config,
      report_figures=args.report_figures,
      report_figure_format=args.report_figure_format,
    )
  else:
    result = adjusted_metrics(data, config)
  write_metric_outputs(result.metrics, args.output_dir)


def generate_synthetic_command(args: argparse.Namespace) -> None:
  """Write reproducible synthetic data without running metrics."""
  if not is_generated_data_path(args.output_dir):
    raise SystemExit("--output-dir must be under an ignored data/generated directory")
  if args.n <= 0:
    raise SystemExit("--n must be greater than 0")
  args.output_dir.mkdir(parents=True, exist_ok=True)
  data = generate_synthetic_metrics_data(n=args.n, seed=args.seed)
  data.to_csv(args.output_dir / "metrics-adjuster-synthetic-data.csv", index=False)
  data.to_csv(args.output_dir / "synthetic_metrics_data.csv", index=False)
  data.to_parquet(args.output_dir / "sample.parquet", index=False)
  print(f"Wrote synthetic data to {args.output_dir}")


def main() -> None:
  """CLI entry point."""
  parser = build_parser()
  args = parser.parse_args()
  if args.command == "run":
    run_command(args)
  elif args.command == "demo":
    demo_command(args)
  elif args.command == "generate-synthetic":
    generate_synthetic_command(args)


if __name__ == "__main__":
  main()
