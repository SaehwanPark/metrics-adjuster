"""Prepare a deterministic sampled input for public CLI report reproduction."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

import pandas as pd

from metrics_adjuster.cli import read_table, resolve_reference_group

DEFAULT_SAMPLE_FILENAME = "sample.parquet"


def build_parser() -> argparse.ArgumentParser:
  """Build the preparation script parser."""
  parser = argparse.ArgumentParser(
    description="Prepare sampled row-level input for metrics-adjuster CLI reports",
  )
  parser.add_argument("--input", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--group-col", required=True)
  parser.add_argument("--ref-group", required=True)
  parser.add_argument("--response-col", required=True)
  parser.add_argument("--risk-col", required=True)
  parser.add_argument("--sample-size", type=int, default=50_000)
  parser.add_argument("--bootstrap-iterations", type=int, default=25)
  parser.add_argument("--seed", type=int, default=20260521)
  parser.add_argument("--report-config-yaml", type=Path)
  parser.add_argument("--report-output-dir", type=Path)
  return parser


def is_generated_data_path(path: Path) -> bool:
  """Return whether a resolved path is under a data/generated directory."""
  parts = path.expanduser().resolve().parts
  return any(
    current == "data" and next_part == "generated"
    for current, next_part in zip(parts, parts[1:], strict=False)
  )


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
  """Validate preparation arguments before writing row-level data."""
  if args.sample_size <= 0:
    parser.error("--sample-size must be greater than 0")
  if args.bootstrap_iterations <= 0:
    parser.error("--bootstrap-iterations must be greater than 0")
  if not is_generated_data_path(args.output_dir):
    parser.error("--output-dir must be under an ignored data/generated directory")


def resolve_existing_reference_group(raw_value: str, groups: pd.Series) -> object:
  """Resolve a reference group and fail if it is absent from observed groups."""
  resolved = resolve_reference_group(raw_value, groups)
  observed = groups.dropna().unique()
  if any(value == resolved for value in observed):
    return resolved
  raise ValueError(f"reference group {raw_value!r} is absent from {groups.name!r}")


def sample_input_frame(args: argparse.Namespace) -> pd.DataFrame:
  """Read only needed columns and sample rows deterministically."""
  columns = [args.group_col, args.response_col, args.risk_col]
  frame = read_table(args.input, columns)
  sample_size = min(args.sample_size, len(frame))
  return frame.sample(n=sample_size, random_state=args.seed).reset_index(drop=True)


def _format_command_line(parts: list[str]) -> str:
  """Render a shell command as a copy-pasteable multiline Bash command."""
  if not parts:
    raise ValueError("command parts must not be empty")
  option_start = next(
    (index for index, part in enumerate(parts) if part.startswith("--")),
    len(parts),
  )
  prefix = " ".join(shlex.quote(part) for part in parts[:option_start])
  quoted = [prefix, *(shlex.quote(part) for part in parts[option_start:])]
  continuation = " \\\n  "
  return continuation.join(quoted)


def build_preparation_command(args: argparse.Namespace) -> str:
  """Return the command that recreates the sampled input."""
  command = [
    "uv",
    "run",
    "python",
    "scripts/prepare_sampled_report_input.py",
    "--input",
    str(args.input),
    "--output-dir",
    str(args.output_dir),
    "--group-col",
    args.group_col,
    "--ref-group",
    args.ref_group,
    "--response-col",
    args.response_col,
    "--risk-col",
    args.risk_col,
    "--sample-size",
    str(args.sample_size),
    "--bootstrap-iterations",
    str(args.bootstrap_iterations),
    "--seed",
    str(args.seed),
  ]
  if args.report_config_yaml is not None:
    command.extend(["--report-config-yaml", str(args.report_config_yaml)])
  if args.report_output_dir is not None:
    command.extend(["--report-output-dir", str(args.report_output_dir)])
  return _format_command_line(command)


def default_report_output_dir(sample_output_dir: Path) -> Path:
  """Derive a matching report output path for a generated sample directory."""
  parts = sample_output_dir.parts
  generated_index = next(
    index
    for index, (current, next_part) in enumerate(zip(parts, parts[1:], strict=False))
    if current == "data" and next_part == "generated"
  )
  suffix = parts[generated_index + 2 :]
  return Path("results").joinpath(*suffix) if suffix else Path("results")


def build_cli_report_command(args: argparse.Namespace, sample_path: Path) -> str:
  """Return the public CLI report command for the prepared sample."""
  report_output_dir = args.report_output_dir or default_report_output_dir(args.output_dir)
  command = [
    "uv",
    "run",
    "metrics-adjuster",
    "run",
    "--input",
    str(sample_path),
    "--output-dir",
    str(report_output_dir),
    "--group-col",
    args.group_col,
    "--ref-group",
    args.ref_group,
    "--response-col",
    args.response_col,
    "--risk-col",
    args.risk_col,
    "--quantiles",
    "0.1,0.3,0.5,0.7,0.9",
    "--metrics",
    "aTPR,aPPV,aNB,aHR",
    "--bootstrap",
    "--n-boot",
    str(args.bootstrap_iterations),
    "--seed",
    str(args.seed),
    "--report",
  ]
  if args.report_config_yaml is not None:
    command.extend(["--report-config-yaml", str(args.report_config_yaml)])
  return _format_command_line(command)


def write_preparation_readme(
  output_dir: Path,
  args: argparse.Namespace,
  analyzed_rows: int,
  resolved_ref_group: object,
) -> None:
  """Document the sampled input and public CLI reproduction commands."""
  sample_path = output_dir / DEFAULT_SAMPLE_FILENAME
  readme = f"""# Prepared sampled report input

- Source input file: `{args.input}`
- Sampled row-level file: `{sample_path}`
- Group column: `{args.group_col}`
- Response column: `{args.response_col}`
- Risk column: `{args.risk_col}`
- Requested reference group: `{args.ref_group}`
- Resolved reference group: `{resolved_ref_group}`
- Sampled size {analyzed_rows}
- Sample seed: `{args.seed}`
- Report config YAML: `{args.report_config_yaml}`
- Privacy note: this directory contains row-level sampled records and should
  remain under the ignored `data/generated/` tree.

## Preparation command

```bash
{build_preparation_command(args)}
```

## Public CLI report command

```bash
{build_cli_report_command(args, sample_path)}
```
"""
  output_dir.mkdir(parents=True, exist_ok=True)
  (output_dir / "README.md").write_text(readme, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
  """Prepare sampled input and reproduction instructions."""
  parser = build_parser()
  args = parser.parse_args(argv)
  validate_args(parser, args)
  frame = sample_input_frame(args)
  resolved_ref_group = resolve_existing_reference_group(args.ref_group, frame[args.group_col])
  args.output_dir.mkdir(parents=True, exist_ok=True)
  sample_path = args.output_dir / DEFAULT_SAMPLE_FILENAME
  frame.to_parquet(sample_path, index=False)
  write_preparation_readme(args.output_dir, args, len(frame), resolved_ref_group)


if __name__ == "__main__":
  main()
