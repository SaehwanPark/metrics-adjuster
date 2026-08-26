"""Prepare reproducible synthetic data for public CLI workflows."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from metrics_adjuster.cli import is_generated_data_path
from metrics_adjuster.synthetic import generate_synthetic_metrics_data

DEFAULT_SAMPLE_FILENAME = "sample.parquet"


def build_parser() -> argparse.ArgumentParser:
  """Build the synthetic data preparation parser."""
  parser = argparse.ArgumentParser(
    description="Prepare reproducible synthetic input for metrics-adjuster CLI workflows",
  )
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--n", type=int, default=600)
  parser.add_argument("--seed", type=int, default=2026)
  return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
  """Validate preparation arguments before writing row-level data."""
  if args.n <= 0:
    parser.error("--n must be greater than 0")
  if not is_generated_data_path(args.output_dir):
    parser.error("--output-dir must be under an ignored data/generated directory")


def render_readme(args: argparse.Namespace, output_dir: Path) -> str:
  """Render a copy-pasteable README for the generated synthetic bundle."""
  sample_path = output_dir / DEFAULT_SAMPLE_FILENAME
  csv_path = output_dir / "synthetic_metrics_data.csv"
  demo_cmd = " ".join(
    shlex.quote(part)
    for part in [
      "uv",
      "run",
      "metrics-adjuster",
      "demo",
      "--output-dir",
      "demo_outputs",
      "--n",
      str(args.n),
      "--seed",
      str(args.seed),
      "--report",
    ]
  )
  run_cmd = " ".join(
    shlex.quote(part)
    for part in [
      "uv",
      "run",
      "metrics-adjuster",
      "run",
      "--input",
      str(sample_path),
      "--output-dir",
      "demo_outputs",
      "--group-col",
      "group",
      "--ref-group",
      "ref",
      "--response-col",
      "outcome",
      "--risk-col",
      "risk",
      "--id-col",
      "patient_id",
      "--report",
      "--save-artifacts",
      "--report-figures",
    ]
  )
  generate_cmd = " ".join(
    shlex.quote(part)
    for part in [
      "uv",
      "run",
      "metrics-adjuster",
      "generate-synthetic",
      "--output-dir",
      str(output_dir),
      "--n",
      str(args.n),
      "--seed",
      str(args.seed),
    ]
  )
  return f"""# Synthetic metrics-adjuster demo data

This directory stores reproducible synthetic row-level input for CLI workflows.
Row-level data is gitignored under `data/generated/`.

## Regenerate

```bash
{generate_cmd}
```

## Files

- `{sample_path.name}`: parquet input for `metrics-adjuster run`
- `metrics-adjuster-synthetic-data.csv`: canonical synthetic dataset
- `{csv_path.name}`: CSV copy of the same synthetic frame

## Example commands

Run the built-in demo with the same seed:

```bash
{demo_cmd}
```

Run metrics on the persisted parquet sample:

```bash
{run_cmd}
```
"""


def main() -> None:
  """Generate synthetic data and a reproducibility README."""
  parser = build_parser()
  args = parser.parse_args()
  validate_args(parser, args)

  output_dir = args.output_dir
  output_dir.mkdir(parents=True, exist_ok=True)
  data = generate_synthetic_metrics_data(n=args.n, seed=args.seed)
  data.to_csv(output_dir / "metrics-adjuster-synthetic-data.csv", index=False)
  data.to_csv(output_dir / "synthetic_metrics_data.csv", index=False)
  data.to_parquet(output_dir / DEFAULT_SAMPLE_FILENAME, index=False)
  (output_dir / "README.md").write_text(render_readme(args, output_dir), encoding="utf-8")
  print(f"Wrote synthetic data to {output_dir}")


if __name__ == "__main__":
  main()
