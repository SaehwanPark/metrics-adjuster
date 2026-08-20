from __future__ import annotations

import sys

import pandas as pd
import pytest

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


def test_cli_demo_can_include_calibrated_metric_family(tmp_path, monkeypatch) -> None:
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
      "--include-calibrated-metrics",
      "--report",
    ],
  )

  main()

  atpr = pd.read_csv(tmp_path / "aTPR.csv")
  dca = pd.read_csv(tmp_path / "decision_curve_table.csv")
  html = (tmp_path / "report.html").read_text(encoding="utf-8")
  assert {"group", "quantile", "tau", "TPR", "cTPR", "aTPR"}.issubset(atpr.columns)
  assert "calibrated" in set(dca["curve_family"])
  assert "Calibrated" in html


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
  assert (tmp_path / "decision_curve_table.csv").exists()
  html = report.read_text(encoding="utf-8")
  assert "Demo report" in html
  assert "Table 1. Metric Estimates" in html
  assert "data:image/svg+xml;base64" in html


def test_cli_demo_save_artifacts_and_report_figures(tmp_path, monkeypatch) -> None:
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
      "--save-artifacts",
      "--report-figures",
    ],
  )
  main()

  assert (tmp_path / "calibration.parquet").exists()
  assert (tmp_path / "weights.parquet").exists()
  assert (tmp_path / "figure_1_calibrated_density.svg").exists()
  assert (tmp_path / "figure_2_weight_ratio.svg").exists()
  assert (tmp_path / "figure_3_standard_subgroup_dca.svg").exists()
  assert (tmp_path / "figure_4_comparative_model_utility.svg").exists()
  assert (tmp_path / "report.html").exists()
  assert (tmp_path / "decision_curve_table.csv").exists()


def test_cli_demo_report_figures_requires_report(tmp_path, monkeypatch) -> None:
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "metrics-adjuster",
      "demo",
      "--output-dir",
      str(tmp_path),
      "--report-figures",
    ],
  )
  with pytest.raises(SystemExit, match="--report-figures requires --report"):
    main()


def test_cli_demo_can_disable_one_dca_plot_via_yaml(tmp_path, monkeypatch) -> None:
  report_yaml = tmp_path / "report.yml"
  report_yaml.write_text(
    "\n".join(
      [
        "decision_curve:",
        "  plots:",
        "    standard_subgroup: false",
        "    comparative_model_utility: true",
      ]
    ),
    encoding="utf-8",
  )
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
      "--report-figures",
      "--report-config-yaml",
      str(report_yaml),
    ],
  )

  main()

  html = (tmp_path / "report.html").read_text(encoding="utf-8")
  assert "Figure 3. Model Net Benefit by Subgroup" in html
  assert "Figure 3. Decision Curves by Subgroup" not in html
  assert (tmp_path / "figure_3_standard_subgroup_dca.svg").exists() is False
  assert (tmp_path / "figure_4_comparative_model_utility.svg").exists()
  assert (tmp_path / "decision_curve_table.csv").exists()


def test_cli_demo_can_disable_dca_csv_artifact_via_yaml(tmp_path, monkeypatch) -> None:
  report_yaml = tmp_path / "report.yml"
  report_yaml.write_text(
    "\n".join(
      [
        "decision_curve:",
        "  write_csv_artifact: false",
      ]
    ),
    encoding="utf-8",
  )
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
      "--report-config-yaml",
      str(report_yaml),
    ],
  )

  main()

  assert (tmp_path / "report.html").exists()
  assert not (tmp_path / "decision_curve_table.csv").exists()
