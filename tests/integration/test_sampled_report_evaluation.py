from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from metrics_adjuster.cli import main as cli_main


def load_prepare_script() -> ModuleType:
  script_path = Path(__file__).parents[2] / "scripts" / "prepare_sampled_report_input.py"
  spec = importlib.util.spec_from_file_location(
    "prepare_sampled_report_input",
    script_path,
  )
  assert spec is not None
  assert spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def test_prepare_sampled_report_input_writes_ignored_sample_without_metrics(
  tmp_path,
) -> None:
  input_path = tmp_path / "input.parquet"
  output_dir = tmp_path / "data" / "generated" / "sampled-report"
  yaml_path = tmp_path / "report.yml"
  df = pd.DataFrame(
    {
      "Prior1245": [0, 0, 99, 99, 1, 1, 2, 2] * 8,
      "Hosp_1y": [1, 0, 1, 0, 1, 0, 1, 0] * 8,
      "pHosp_1y": [0.95, 0.2, 0.85, 0.3, 0.75, 0.15, 0.65, 0.05] * 8,
    }
  )
  df.to_parquet(input_path, index=False)
  yaml_path.write_text(
    """
title: Sampled report
labels:
  columns:
    Prior1245: Veteran Priority Group
  groups:
    Prior1245:
      "0": Default Priority
      "99": Priority Group 5
  metrics:
    aTPR: True positive rate
""".strip(),
    encoding="utf-8",
  )

  prepare_script = load_prepare_script()

  prepare_script.main(
    [
      "--input",
      str(input_path),
      "--output-dir",
      str(output_dir),
      "--group-col",
      "Prior1245",
      "--ref-group",
      "0",
      "--response-col",
      "Hosp_1y",
      "--risk-col",
      "pHosp_1y",
      "--sample-size",
      "32",
      "--bootstrap-iterations",
      "4",
      "--seed",
      "11",
      "--report-config-yaml",
      str(yaml_path),
    ]
  )

  expected = {"sample.parquet", "README.md"}
  written = {path.name for path in Path(output_dir).iterdir()}

  assert written == expected
  sample = pd.read_parquet(output_dir / "sample.parquet")
  assert list(sample.columns) == ["Prior1245", "Hosp_1y", "pHosp_1y"]
  assert len(sample) == 32
  readme = (output_dir / "README.md").read_text(encoding="utf-8")
  assert "sampled size 32" in readme.lower()
  assert "uv run python scripts/prepare_sampled_report_input.py" in readme
  assert "uv run metrics-adjuster run" in readme
  assert f"--input \\\n  {input_path} \\" in readme
  assert f"--input \\\n  {output_dir / 'sample.parquet'} \\" in readme
  assert "--output-dir \\\n  results/sampled-report \\" in readme
  assert "--sample-size \\\n  32 \\" in readme
  assert "--bootstrap \\" in readme
  assert "--n-boot \\\n  4 \\" in readme
  assert "--metrics \\\n  aTPR,aPPV,aNB,aHR \\" in readme
  assert f"--report-config-yaml \\\n  {yaml_path}\n```" in readme
  assert not (output_dir / "aTPR.csv").exists()
  assert not (output_dir / "bootstrap.csv").exists()
  assert not (output_dir / "report.html").exists()


def test_prepare_sampled_report_input_subprocess_entrypoint(tmp_path) -> None:
  input_path = tmp_path / "input.parquet"
  output_dir = tmp_path / "data" / "generated" / "subprocess-sample"
  df = pd.DataFrame(
    {
      "Prior1245": [0, 0, 99, 99] * 4,
      "Hosp_1y": [1, 0, 1, 0] * 4,
      "pHosp_1y": [0.95, 0.2, 0.85, 0.3] * 4,
    }
  )
  df.to_parquet(input_path, index=False)

  result = subprocess.run(
    [
      sys.executable,
      "scripts/prepare_sampled_report_input.py",
      "--input",
      str(input_path),
      "--output-dir",
      str(output_dir),
      "--group-col",
      "Prior1245",
      "--ref-group",
      "0",
      "--response-col",
      "Hosp_1y",
      "--risk-col",
      "pHosp_1y",
      "--sample-size",
      "8",
      "--bootstrap-iterations",
      "2",
      "--seed",
      "11",
    ],
    cwd=Path(__file__).parents[2],
    check=False,
    text=True,
    capture_output=True,
  )

  assert result.returncode == 0, result.stderr
  assert (output_dir / "sample.parquet").exists()
  assert (output_dir / "README.md").exists()


@pytest.mark.parametrize(
  ("option", "value", "message"),
  [
    ("--sample-size", "0", "--sample-size must be greater than 0"),
    ("--bootstrap-iterations", "-1", "--bootstrap-iterations must be greater than 0"),
  ],
)
def test_prepare_sampled_report_input_rejects_invalid_positive_options(
  tmp_path,
  option,
  value,
  message,
) -> None:
  input_path = tmp_path / "input.parquet"
  output_dir = tmp_path / "data" / "generated" / "sampled-report"
  df = pd.DataFrame(
    {
      "Prior1245": [0, 0, 99, 99],
      "Hosp_1y": [1, 0, 1, 0],
      "pHosp_1y": [0.95, 0.2, 0.85, 0.3],
    }
  )
  df.to_parquet(input_path, index=False)
  prepare_script = load_prepare_script()

  args = [
    "--input",
    str(input_path),
    "--output-dir",
    str(output_dir),
    "--group-col",
    "Prior1245",
    "--ref-group",
    "0",
    "--response-col",
    "Hosp_1y",
    "--risk-col",
    "pHosp_1y",
    option,
    value,
  ]

  with pytest.raises(SystemExit):
    prepare_script.main(args)

  assert not output_dir.exists()


def test_prepare_sampled_report_input_rejects_non_generated_output_dir(
  tmp_path,
) -> None:
  input_path = tmp_path / "input.parquet"
  output_dir = tmp_path / "results" / "sampled-report"
  df = pd.DataFrame(
    {
      "Prior1245": [0, 0, 99, 99],
      "Hosp_1y": [1, 0, 1, 0],
      "pHosp_1y": [0.95, 0.2, 0.85, 0.3],
    }
  )
  df.to_parquet(input_path, index=False)
  prepare_script = load_prepare_script()

  with pytest.raises(SystemExit):
    prepare_script.main(
      [
        "--input",
        str(input_path),
        "--output-dir",
        str(output_dir),
        "--group-col",
        "Prior1245",
        "--ref-group",
        "0",
        "--response-col",
        "Hosp_1y",
        "--risk-col",
        "pHosp_1y",
      ]
    )

  assert not output_dir.exists()


def test_prepare_sampled_report_input_validates_reference_group_before_write(
  tmp_path,
) -> None:
  input_path = tmp_path / "input.parquet"
  output_dir = tmp_path / "data" / "generated" / "sampled-report"
  df = pd.DataFrame(
    {
      "Prior1245": [0, 0, 99, 99],
      "Hosp_1y": [1, 0, 1, 0],
      "pHosp_1y": [0.95, 0.2, 0.85, 0.3],
    }
  )
  df.to_parquet(input_path, index=False)
  prepare_script = load_prepare_script()

  with pytest.raises(ValueError, match="reference group 'missing' is absent"):
    prepare_script.main(
      [
        "--input",
        str(input_path),
        "--output-dir",
        str(output_dir),
        "--group-col",
        "Prior1245",
        "--ref-group",
        "missing",
        "--response-col",
        "Hosp_1y",
        "--risk-col",
        "pHosp_1y",
      ]
    )

  assert not output_dir.exists()


def test_cli_run_reproduces_report_from_prepared_sample(
  tmp_path,
  monkeypatch,
) -> None:
  input_path = tmp_path / "input.parquet"
  sample_dir = tmp_path / "data" / "generated" / "sampled-report"
  report_dir = tmp_path / "results"
  yaml_path = tmp_path / "report.yml"
  df = pd.DataFrame(
    {
      "Prior1245": [0, 0, 99, 99, 1, 1, 2, 2] * 8,
      "Hosp_1y": [1, 0, 1, 0, 1, 0, 1, 0] * 8,
      "pHosp_1y": [0.95, 0.2, 0.85, 0.3, 0.75, 0.15, 0.65, 0.05] * 8,
    }
  )
  df.to_parquet(input_path, index=False)
  yaml_path.write_text(
    """
title: Sampled report
labels:
  columns:
    Prior1245: Veteran Priority Group
  groups:
    Prior1245:
      "0": Default Priority
      "99": Priority Group 5
  metrics:
    aTPR: True positive rate
""".strip(),
    encoding="utf-8",
  )
  prepare_script = load_prepare_script()

  prepare_script.main(
    [
      "--input",
      str(input_path),
      "--output-dir",
      str(sample_dir),
      "--group-col",
      "Prior1245",
      "--ref-group",
      "0",
      "--response-col",
      "Hosp_1y",
      "--risk-col",
      "pHosp_1y",
      "--sample-size",
      "32",
      "--bootstrap-iterations",
      "4",
      "--seed",
      "11",
      "--report-config-yaml",
      str(yaml_path),
    ]
  )

  monkeypatch.setattr(
    sys,
    "argv",
    [
      "metrics-adjuster",
      "run",
      "--input",
      str(sample_dir / "sample.parquet"),
      "--output-dir",
      str(report_dir),
      "--group-col",
      "Prior1245",
      "--ref-group",
      "0",
      "--response-col",
      "Hosp_1y",
      "--risk-col",
      "pHosp_1y",
      "--quantiles",
      "0.1,0.3,0.5,0.7,0.9",
      "--metrics",
      "aTPR,aPPV,aNB,aHR",
      "--bootstrap",
      "--n-boot",
      "4",
      "--seed",
      "11",
      "--report",
      "--report-config-yaml",
      str(yaml_path),
    ],
  )

  cli_main()

  written = {path.name for path in Path(report_dir).iterdir()}
  assert {
    "aTPR.csv",
    "aPPV.csv",
    "aNB.csv",
    "aHR.csv",
    "bootstrap.csv",
    "report.html",
  }.issubset(written)
  html = (report_dir / "report.html").read_text(encoding="utf-8")
  assert "Veteran Priority Group" in html
  assert "Default Priority" in html
  assert "Priority Group 5" in html
  assert "True positive rate" in html
