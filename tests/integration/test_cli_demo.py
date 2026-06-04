from __future__ import annotations

import sys

import pandas as pd

from metrics_adjuster.cli import main


def test_cli_demo_defaults_to_atpr_metric_file(tmp_path, monkeypatch) -> None:
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "metrics-adjuster",
      "demo",
      "--output-dir",
      str(tmp_path),
      "--n",
      "120",
      "--seed",
      "42",
    ],
  )
  main()
  expected = [
    "synthetic_metrics_data.csv",
    "aTPR.csv",
  ]
  for filename in expected:
    assert (tmp_path / filename).exists()
  for filename in ["aPPV.csv", "aNB.csv", "aHR.csv"]:
    assert not (tmp_path / filename).exists()
  atpr = pd.read_csv(tmp_path / "aTPR.csv")
  assert {"group", "quantile", "tau", "TPR", "aTPR"}.issubset(atpr.columns)
  assert len(atpr) == 8


def test_cli_demo_preserves_explicit_multi_metric_selection(
  tmp_path,
  monkeypatch,
) -> None:
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "metrics-adjuster",
      "demo",
      "--output-dir",
      str(tmp_path),
      "--n",
      "120",
      "--seed",
      "42",
      "--metrics",
      "aTPR,aPPV,aNB,aHR",
    ],
  )
  main()

  for filename in ["aTPR.csv", "aPPV.csv", "aNB.csv", "aHR.csv"]:
    assert (tmp_path / filename).exists()


def test_cli_demo_report_writes_self_contained_html(tmp_path, monkeypatch) -> None:
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "metrics-adjuster",
      "demo",
      "--output-dir",
      str(tmp_path),
      "--n",
      "120",
      "--seed",
      "42",
      "--report",
      "--report-title",
      "Demo report",
      "--report-max-cutoff-lines",
      "2",
    ],
  )
  main()

  report = tmp_path / "report.html"
  assert report.exists()
  html = report.read_text(encoding="utf-8")
  assert "Demo report" in html
  assert "Table 1. Metric Estimates" in html
  assert "data:image/svg+xml;base64" in html
