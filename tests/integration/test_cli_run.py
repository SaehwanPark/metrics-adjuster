from __future__ import annotations

import sys

import pandas as pd

from metrics_adjuster.cli import main


def test_cli_run_report_defaults_to_atpr_metric_from_parquet(
  tmp_path,
  monkeypatch,
) -> None:
  input_path = tmp_path / "input.parquet"
  df = pd.DataFrame(
    {
      "Prior1245": [99, 99, 1, 1, 2, 2] * 6,
      "Hosp_1y": [1, 0, 1, 0, 1, 0] * 6,
      "pHosp_1y": [0.92, 0.21, 0.81, 0.32, 0.71, 0.11] * 6,
    }
  )
  df.to_parquet(input_path, index=False)

  monkeypatch.setattr(
    sys,
    "argv",
    [
      "metrics-adjuster",
      "run",
      "--input",
      str(input_path),
      "--output-dir",
      str(tmp_path),
      "--group-col",
      "Prior1245",
      "--ref-group",
      "99",
      "--response-col",
      "Hosp_1y",
      "--risk-col",
      "pHosp_1y",
      "--quantiles",
      "0.5",
      "--report",
    ],
  )

  main()

  assert (tmp_path / "aTPR.csv").exists()
  assert not (tmp_path / "aPPV.csv").exists()
  assert not (tmp_path / "aNB.csv").exists()
  assert not (tmp_path / "aHR.csv").exists()
  assert (tmp_path / "report.html").exists()


def test_cli_run_accepts_numeric_reference_group_from_parquet(
  tmp_path,
  monkeypatch,
) -> None:
  input_path = tmp_path / "input.parquet"
  yaml_path = tmp_path / "report.yml"
  df = pd.DataFrame(
    {
      "Prior1245": [99, 99, 1, 1, 2, 2],
      "Hosp_1y": [1, 0, 1, 0, 1, 0],
      "pHosp_1y": [0.92, 0.21, 0.81, 0.32, 0.71, 0.11],
    }
  )
  df.to_parquet(input_path, index=False)
  yaml_path.write_text(
    """
title: Numeric ref-group report
x_scale: log_odds
labels:
  columns:
    Prior1245: Veteran Priority Group
    Hosp_1y: 1-year hospitalization
    pHosp_1y: Predicted hospitalization risk
  groups:
    Prior1245:
      "99": Reference group
      "1": Priority group 1
      "2": Priority group 2
  metrics:
    aTPR: True positive rate
""".strip(),
    encoding="utf-8",
  )

  monkeypatch.setattr(
    sys,
    "argv",
    [
      "metrics-adjuster",
      "run",
      "--input",
      str(input_path),
      "--output-dir",
      str(tmp_path),
      "--group-col",
      "Prior1245",
      "--ref-group",
      "99",
      "--response-col",
      "Hosp_1y",
      "--risk-col",
      "pHosp_1y",
      "--quantiles",
      "0.5",
      "--metrics",
      "aTPR",
      "--bootstrap",
      "--n-boot",
      "4",
      "--seed",
      "7",
      "--report",
      "--report-title",
      "Fallback title",
      "--report-config-yaml",
      str(yaml_path),
    ],
  )

  main()

  atpr = pd.read_csv(tmp_path / "aTPR.csv")
  bootstrap = pd.read_csv(tmp_path / "bootstrap.csv")
  report = (tmp_path / "report.html").read_text(encoding="utf-8")

  assert {"Prior1245", "quantile", "tau", "TPR", "aTPR"}.issubset(atpr.columns)
  assert set(atpr["Prior1245"]) == {1, 2, 99}
  assert {"original_value", "adjusted_value"}.issubset(bootstrap.columns)
  assert "Numeric ref-group report" in report
  assert "Veteran Priority Group" in report
  assert "Reference group" in report
  assert "True positive rate" in report
  assert "Table 1. Metric Estimates" in report
