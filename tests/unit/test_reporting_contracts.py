from __future__ import annotations

import numpy as np
import pandas as pd

from metrics_adjuster import (
  BootstrapConfig,
  CalibrationConfig,
  ColumnSpec,
  DensityRatioConfig,
  MetricConfig,
  MetricName,
  ReportConfig,
  ReportLabelConfig,
  adjusted_metrics_report,
  metric_comparison_table,
  write_report_figures,
)
from metrics_adjuster.core import run_metric_pipeline
from metrics_adjuster.reporting import cutoff_reference_lines
from metrics_adjuster.synthetic import generate_synthetic_metrics_data


def report_config(
  *,
  bootstrap: BootstrapConfig | None = None,
  quantiles: tuple[float, ...] = (0.25, 0.5, 0.75),
) -> MetricConfig:
  return MetricConfig(
    columns=ColumnSpec(group="group", response="outcome", risk="risk", id="patient_id"),
    ref_group="ref",
    quantiles=quantiles,
    metrics=(MetricName.ATPR,),
    calibration=CalibrationConfig(degree=1, cv=False),
    density_ratio=DensityRatioConfig(degree=1, cv=False),
    bootstrap=bootstrap or BootstrapConfig(),
    random_state=7,
  )


def test_metric_comparison_table_marks_ci_unavailable_without_bootstrap() -> None:
  df = generate_synthetic_metrics_data(n=140, seed=7)
  pipeline = run_metric_pipeline(df, report_config())

  table = metric_comparison_table(
    pipeline.metrics.metrics,
    "group",
    pipeline.metrics.bootstrap,
  )

  assert {
    "metric",
    "group",
    "quantile",
    "tau",
    "original_metric",
    "original_value",
    "original_ci_lower",
    "original_ci_upper",
    "adjusted_metric",
    "adjusted_value",
    "adjusted_ci_lower",
    "adjusted_ci_upper",
    "n_boot",
  } == set(table.columns)
  assert table["original_ci_lower"].isna().all()
  assert table["adjusted_ci_upper"].isna().all()
  assert table["n_boot"].isna().all()


def test_metric_comparison_table_returns_stable_empty_schema_for_no_metrics() -> None:
  table = metric_comparison_table({}, "group")

  assert table.empty
  assert list(table.columns) == [
    "metric",
    "group",
    "quantile",
    "tau",
    "original_metric",
    "original_value",
    "original_ci_lower",
    "original_ci_upper",
    "adjusted_metric",
    "adjusted_value",
    "adjusted_ci_lower",
    "adjusted_ci_upper",
    "n_boot",
  ]


def test_metric_comparison_table_includes_original_and_adjusted_bootstrap_intervals() -> None:
  df = generate_synthetic_metrics_data(n=120, seed=8)
  pipeline = run_metric_pipeline(
    df,
    report_config(bootstrap=BootstrapConfig(enabled=True, iterations=4, alpha=0.1)),
  )

  assert pipeline.metrics.bootstrap is not None
  assert {"original_value", "adjusted_value", "value"}.issubset(
    pipeline.metrics.bootstrap.columns
  )
  table = metric_comparison_table(
    pipeline.metrics.metrics,
    "group",
    pipeline.metrics.bootstrap,
    alpha=0.1,
  )

  assert table["original_ci_lower"].notna().all()
  assert table["adjusted_ci_upper"].notna().all()
  assert set(table["n_boot"]) == {4}


def test_cutoff_reference_lines_caps_many_cutoffs_with_endpoints() -> None:
  table = pd.DataFrame(
    {
      "quantile": [0.1, 0.2, 0.3, 0.4, 0.5],
      "tau": [0.11, 0.22, 0.33, 0.44, 0.55],
    }
  )

  cutoffs = cutoff_reference_lines(table, max_cutoff_lines=3)

  assert cutoffs[0] == (0.1, 0.11)
  assert cutoffs[-1] == (0.5, 0.55)
  assert len(cutoffs) == 3


def test_adjusted_metrics_report_returns_html_and_component_figures() -> None:
  df = generate_synthetic_metrics_data(n=140, seed=9)
  bundle = adjusted_metrics_report(
    df,
    report_config(quantiles=(0.25, 0.5)),
    ReportConfig(
      title="Synthetic fairness report",
      subtitle="Contract subtitle",
      max_cutoff_lines=2,
    ),
  )

  assert not bundle.metric_table.empty
  assert len(bundle.density_figure.axes) == 1
  assert len(bundle.weight_ratio_figure.axes) == 1
  assert "Synthetic fairness report" in bundle.html
  assert "Contract subtitle" in bundle.html
  assert "Run Summary" in bundle.html
  assert "Table 1. Metric Estimates" in bundle.html
  assert "Figure 1. Calibrated Probability Densities" in bundle.html
  assert "Figure 2. Densities Normalized to Reference Group" in bundle.html
  assert "Rows analyzed" in bundle.html
  assert "Threshold" in bundle.html
  assert "original_ci_lower" not in bundle.html
  assert "data:image/svg+xml;base64" in bundle.html
  assert "unavailable" in bundle.html


def test_report_labels_apply_to_summary_table_and_figures() -> None:
  df = generate_synthetic_metrics_data(n=140, seed=11)
  bundle = adjusted_metrics_report(
    df,
    report_config(quantiles=(0.25,)),
    ReportConfig(
      labels=ReportLabelConfig(
        columns={
          "group": "Veteran Priority Group",
          "outcome": "1-year hospitalization",
          "risk": "Predicted hospitalization risk",
        },
        groups={"group": {"ref": "Reference group", "minority": "Comparison group"}},
        metrics={"aTPR": "True positive rate"},
      )
    ),
  )

  density_labels = {
    line.get_label()
    for line in bundle.density_figure.axes[0].lines
    if not line.get_label().startswith("_")
  }
  ratio_labels = {
    line.get_label()
    for line in bundle.weight_ratio_figure.axes[0].lines
    if not line.get_label().startswith("_")
  }

  assert "Veteran Priority Group" in bundle.html
  assert "1-year hospitalization" in bundle.html
  assert "True positive rate" in bundle.html
  assert "Reference group" in bundle.html
  assert {"Reference group", "Comparison group", "cutoff"}.issubset(density_labels)
  assert {"Reference group", "Comparison group", "cutoff"}.issubset(ratio_labels)


def test_log_odds_report_scale_and_reference_normalization() -> None:
  df = generate_synthetic_metrics_data(n=140, seed=12)
  bundle = adjusted_metrics_report(
    df,
    report_config(quantiles=(0.25,)),
    ReportConfig(
      x_scale="log_odds",
      labels=ReportLabelConfig(groups={"group": {"ref": "Reference group"}}),
    ),
  )

  density_ax = bundle.density_figure.axes[0]
  ratio_ax = bundle.weight_ratio_figure.axes[0]
  ref_line = next(line for line in ratio_ax.lines if line.get_label() == "Reference group")

  assert density_ax.get_xlabel() == "Calibrated log-odds"
  assert ratio_ax.get_xlabel() == "Calibrated log-odds"
  assert ratio_ax.get_ylabel() == "Density relative to reference"
  assert np.allclose(ref_line.get_ydata(), 1.0)


def test_adjusted_metrics_report_accepts_empty_metric_selection() -> None:
  df = generate_synthetic_metrics_data(n=120, seed=10)
  config = MetricConfig(
    columns=ColumnSpec(group="group", response="outcome", risk="risk", id="patient_id"),
    ref_group="ref",
    quantiles=(0.25, 0.5),
    metrics=(),
    calibration=CalibrationConfig(degree=1, cv=False),
    density_ratio=DensityRatioConfig(degree=1, cv=False),
    random_state=10,
  )

  bundle = adjusted_metrics_report(df, config)

  assert bundle.metric_table.empty
  assert len(bundle.density_figure.axes) == 1
  assert len(bundle.weight_ratio_figure.axes) == 1
  assert "Table 1. Metric Estimates" in bundle.html


def test_write_report_figures_writes_svg_and_png(tmp_path) -> None:
  df = generate_synthetic_metrics_data(n=120, seed=13)
  bundle = adjusted_metrics_report(df, report_config(quantiles=(0.5,)))

  svg_density, svg_weight = write_report_figures(bundle, tmp_path / "svg", figure_format="svg")
  png_density, png_weight = write_report_figures(bundle, tmp_path / "png", figure_format="png")

  assert svg_density.name == "figure_1_calibrated_density.svg"
  assert svg_weight.name == "figure_2_weight_ratio.svg"
  assert png_density.suffix == ".png"
  assert png_weight.suffix == ".png"
  assert svg_density.exists()
  assert png_weight.exists()
