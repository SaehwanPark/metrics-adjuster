from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metrics_adjuster import (
  BootstrapConfig,
  CalibrationConfig,
  ColumnSpec,
  DecisionCurveConfig,
  DensityRatioConfig,
  MetricConfig,
  MetricName,
  ReportConfig,
  ReportLabelConfig,
  adjusted_metrics_report,
  decision_curve_table,
  metric_comparison_table,
  write_decision_curve_csv,
  write_decision_curve_figure,
  write_decision_curve_figures,
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


def test_metric_comparison_table_includes_calibrated_values_when_requested() -> None:
  df = generate_synthetic_metrics_data(n=120, seed=18)
  config = report_config(
    bootstrap=BootstrapConfig(enabled=True, iterations=4, alpha=0.1),
    quantiles=(0.5,),
  ).model_copy(update={"include_calibrated_metrics": True})
  pipeline = run_metric_pipeline(df, config)

  assert pipeline.metrics.bootstrap is not None
  table = metric_comparison_table(
    pipeline.metrics.metrics,
    "group",
    pipeline.metrics.bootstrap,
    alpha=0.1,
  )

  assert {"calibrated_metric", "calibrated_value", "calibrated_ci_lower"}.issubset(
    table.columns
  )
  assert table["calibrated_metric"].eq("cTPR").all()
  assert table["calibrated_ci_lower"].notna().all()


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


def test_decision_curve_config_rejects_invalid_threshold_range() -> None:
  with pytest.raises(ValueError, match="threshold_min must be less than threshold_max"):
    DecisionCurveConfig(threshold_min=0.4, threshold_max=0.4)


def test_decision_curve_table_matches_hand_calculated_values() -> None:
  weighted = pd.DataFrame(
    {
      "group": ["ref", "ref", "alt", "alt"],
      "outcome": [1, 0, 1, 0],
      "risk": [0.8, 0.4, 0.7, 0.1],
      "cal_risk": [0.9, 0.2, 0.6, 0.3],
      "dens_ratio": [1.0, 1.0, 2.0, 2.0],
    }
  )
  table = decision_curve_table(
    weighted,
    group_col="group",
    response_col="outcome",
    risk_col="risk",
    thresholds=np.array([0.5]),
    ref_group="ref",
  )

  ref_conventional = table[
    (table["group"] == "ref")
    & (table["curve_family"] == "conventional")
    & (table["strategy"] == "model")
  ]["net_benefit"].item()
  alt_adjusted = table[
    (table["group"] == "alt")
    & (table["curve_family"] == "adjusted")
    & (table["strategy"] == "model")
  ]["net_benefit"].item()
  treat_none = table[
    (table["group"] == "ref")
    & (table["curve_family"] == "conventional")
    & (table["strategy"] == "treat_none")
  ]["net_benefit"].item()
  adjusted_treat_all = table[
    (table["curve_family"] == "adjusted") & (table["strategy"] == "treat_all")
  ]["net_benefit"]

  mu0 = 0.5
  atpr = (0.6 * 2.0) / (0.6 * 2.0 + 0.3 * 2.0)
  afpr = (0.4 * 2.0) / (0.4 * 2.0 + 0.7 * 2.0)
  expected_anb = atpr * mu0 - afpr * (1.0 - mu0) * 1.0
  old_weighted_mean_estimator = 0.1

  assert ref_conventional == pytest.approx(0.5)
  assert alt_adjusted == pytest.approx(expected_anb)
  assert alt_adjusted != pytest.approx(old_weighted_mean_estimator)
  assert treat_none == 0.0
  assert adjusted_treat_all.nunique() == 1
  assert adjusted_treat_all.iloc[0] == pytest.approx(mu0 - (1.0 - mu0) * 1.0)


def test_decision_curve_table_thresholds_on_model_score_not_calibrated_risk() -> None:
  weighted = pd.DataFrame(
    {
      "group": ["ref", "ref", "alt", "alt"],
      "outcome": [1, 0, 1, 0],
      "risk": [0.8, 0.4, 0.9, 0.8],
      "cal_risk": [0.4, 0.3, 0.8, 0.1],
      "dens_ratio": [1.0, 1.0, 2.0, 2.0],
    }
  )

  table = decision_curve_table(
    weighted,
    group_col="group",
    response_col="outcome",
    risk_col="risk",
    thresholds=np.array([0.5]),
    ref_group="ref",
  )

  ref_conventional = table[
    (table["group"] == "ref")
    & (table["curve_family"] == "conventional")
    & (table["strategy"] == "model")
  ]["net_benefit"].item()
  alt_adjusted = table[
    (table["group"] == "alt")
    & (table["curve_family"] == "adjusted")
    & (table["strategy"] == "model")
  ]["net_benefit"].item()

  # Original scores classify the first reference row as high-risk; calibrated
  # risks would classify nobody in the reference group as high-risk.
  assert ref_conventional == pytest.approx(0.5)
  assert alt_adjusted == pytest.approx(0.0)


def test_decision_curve_table_can_include_calibrated_family() -> None:
  weighted = pd.DataFrame(
    {
      "group": ["ref", "ref", "alt", "alt"],
      "outcome": [1, 0, 1, 0],
      "risk": [0.8, 0.4, 0.7, 0.1],
      "cal_risk": [0.9, 0.2, 0.6, 0.3],
      "dens_ratio": [1.0, 1.0, 2.0, 2.0],
    }
  )

  table = decision_curve_table(
    weighted,
    group_col="group",
    response_col="outcome",
    risk_col="risk",
    thresholds=np.array([0.5]),
    include_calibrated_metrics=True,
    ref_group="ref",
  )

  alt_calibrated_model = table[
    (table["group"] == "alt")
    & (table["curve_family"] == "calibrated")
    & (table["strategy"] == "model")
  ]["net_benefit"].item()
  alt_calibrated_treat_all = table[
    (table["group"] == "alt")
    & (table["curve_family"] == "calibrated")
    & (table["strategy"] == "treat_all")
  ]["net_benefit"].item()

  assert set(table["curve_family"]) == {"conventional", "calibrated", "adjusted"}
  assert alt_calibrated_model == pytest.approx(0.1)
  assert alt_calibrated_treat_all == pytest.approx(-0.1)


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
  assert bundle.decision_curve_table is not None
  assert not bundle.decision_curve_table.empty
  assert len(bundle.density_figure.axes) == 1
  assert len(bundle.weight_ratio_figure.axes) == 1
  assert bundle.decision_curve_standard_subgroup_figure is not None
  assert bundle.decision_curve_comparative_model_utility_figure is not None
  assert len(bundle.decision_curve_standard_subgroup_figure.axes) == 2
  assert len(bundle.decision_curve_comparative_model_utility_figure.axes) == 2
  assert "Synthetic fairness report" in bundle.html
  assert "Contract subtitle" in bundle.html
  assert "Run Summary" in bundle.html
  assert "Table 1. Metric Estimates" in bundle.html


def test_adjusted_metrics_report_exposes_calibrated_family_when_requested() -> None:
  df = generate_synthetic_metrics_data(n=140, seed=19)
  config = report_config(quantiles=(0.5,)).model_copy(
    update={"include_calibrated_metrics": True}
  )

  bundle = adjusted_metrics_report(df, config)

  assert "calibrated_value" in bundle.metric_table.columns
  assert bundle.decision_curve_table is not None
  assert "calibrated" in set(bundle.decision_curve_table["curve_family"])
  assert "Calibrated" in bundle.html
  assert bundle.decision_curve_comparative_model_utility_figure is not None
  assert len(bundle.decision_curve_comparative_model_utility_figure.axes) == 3
  assert "Figure 1. Calibrated Probability Densities" in bundle.html
  assert "Figure 2. Densities Normalized to Reference Group" in bundle.html
  assert "Figure 3. Decision Curves by Subgroup" in bundle.html
  assert "Figure 4. Model Net Benefit by Subgroup" in bundle.html
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
  assert bundle.decision_curve_standard_subgroup_figure is not None
  assert bundle.decision_curve_comparative_model_utility_figure is not None
  assert "Table 1. Metric Estimates" in bundle.html


def test_adjusted_metrics_report_can_disable_decision_curve() -> None:
  df = generate_synthetic_metrics_data(n=120, seed=14)
  bundle = adjusted_metrics_report(
    df,
    report_config(quantiles=(0.5,)),
    ReportConfig(decision_curve=DecisionCurveConfig(enabled=False)),
  )

  assert bundle.decision_curve_table is None
  assert bundle.decision_curve_standard_subgroup_figure is None
  assert bundle.decision_curve_comparative_model_utility_figure is None
  assert "Figure 3. Decision Curves by Subgroup" not in bundle.html
  assert "Figure 3. Model Net Benefit by Subgroup" not in bundle.html


def test_adjusted_metrics_report_can_select_only_standard_subgroup_dca() -> None:
  df = generate_synthetic_metrics_data(n=120, seed=16)
  bundle = adjusted_metrics_report(
    df,
    report_config(quantiles=(0.5,)),
    ReportConfig(
      decision_curve=DecisionCurveConfig(
        plots={"comparative_model_utility": False},
      )
    ),
  )

  assert bundle.decision_curve_standard_subgroup_figure is not None
  assert bundle.decision_curve_comparative_model_utility_figure is None
  assert "Figure 3. Decision Curves by Subgroup" in bundle.html
  assert "Figure 4. Model Net Benefit by Subgroup" not in bundle.html


def test_adjusted_metrics_report_can_select_only_comparative_utility_dca() -> None:
  df = generate_synthetic_metrics_data(n=120, seed=17)
  bundle = adjusted_metrics_report(
    df,
    report_config(quantiles=(0.5,)),
    ReportConfig(
      decision_curve=DecisionCurveConfig(
        plots={"standard_subgroup": False},
      )
    ),
  )

  assert bundle.decision_curve_standard_subgroup_figure is None
  assert bundle.decision_curve_comparative_model_utility_figure is not None
  assert "Figure 3. Model Net Benefit by Subgroup" in bundle.html
  assert "Figure 3. Decision Curves by Subgroup" not in bundle.html


def test_decision_curve_log_odds_axis_applies_to_both_dca_plots() -> None:
  df = generate_synthetic_metrics_data(n=120, seed=18)
  bundle = adjusted_metrics_report(
    df,
    report_config(quantiles=(0.5,)),
    ReportConfig(x_scale="log_odds"),
  )

  assert bundle.decision_curve_standard_subgroup_figure is not None
  assert bundle.decision_curve_comparative_model_utility_figure is not None
  subgroup_xlabel = bundle.decision_curve_standard_subgroup_figure.axes[0].get_xlabel()
  comparative_xlabel = bundle.decision_curve_comparative_model_utility_figure.axes[
    0
  ].get_xlabel()
  assert subgroup_xlabel == "Calibrated log-odds"
  assert comparative_xlabel == "Calibrated log-odds"


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


def test_decision_curve_subgroup_figure_legend_uses_net_benefit_labels() -> None:
  df = generate_synthetic_metrics_data(n=120, seed=20)
  bundle = adjusted_metrics_report(df, report_config(quantiles=(0.5,)))

  assert bundle.decision_curve_standard_subgroup_figure is not None
  legend_labels = {
    line.get_label()
    for line in bundle.decision_curve_standard_subgroup_figure.axes[0].lines
    if not line.get_label().startswith("_")
  }
  assert {"original net benefit", "adjusted net benefit", "treat all"}.issubset(legend_labels)


def test_write_decision_curve_figure_writes_configured_format(tmp_path) -> None:
  df = generate_synthetic_metrics_data(n=120, seed=15)
  bundle = adjusted_metrics_report(df, report_config(quantiles=(0.5,)))

  svg_path = write_decision_curve_figure(bundle, tmp_path / "svg", figure_format="svg")
  png_path = write_decision_curve_figure(bundle, tmp_path / "png", figure_format="png")

  assert svg_path.name == "figure_4_comparative_model_utility.svg"
  assert png_path.name == "figure_4_comparative_model_utility.png"
  assert svg_path.exists()
  assert png_path.exists()


def test_write_decision_curve_figures_writes_only_enabled_plot_files(tmp_path) -> None:
  df = generate_synthetic_metrics_data(n=120, seed=19)
  standard_only = adjusted_metrics_report(
    df,
    report_config(quantiles=(0.5,)),
    ReportConfig(
      decision_curve=DecisionCurveConfig(
        plots={"comparative_model_utility": False},
      )
    ),
  )
  both_enabled = adjusted_metrics_report(df, report_config(quantiles=(0.5,)))

  standard_paths = write_decision_curve_figures(
    standard_only,
    tmp_path / "standard_only",
    figure_format="svg",
  )
  both_paths = write_decision_curve_figures(
    both_enabled,
    tmp_path / "both",
    figure_format="svg",
  )

  assert [path.name for path in standard_paths] == ["figure_3_standard_subgroup_dca.svg"]
  assert [path.name for path in both_paths] == [
    "figure_3_standard_subgroup_dca.svg",
    "figure_4_comparative_model_utility.svg",
  ]


def test_write_decision_curve_csv_writes_shared_artifact(tmp_path) -> None:
  df = generate_synthetic_metrics_data(n=120, seed=21)
  bundle = adjusted_metrics_report(df, report_config(quantiles=(0.5,)))

  csv_path = write_decision_curve_csv(bundle, tmp_path)

  written = pd.read_csv(csv_path)
  assert csv_path.name == "decision_curve_table.csv"
  assert csv_path.exists()
  assert set(["curve_family", "group", "strategy", "threshold", "net_benefit"]).issubset(
    written.columns
  )


def test_write_decision_curve_csv_rejects_bundle_without_dca_table(tmp_path) -> None:
  df = generate_synthetic_metrics_data(n=120, seed=22)
  bundle = adjusted_metrics_report(
    df,
    report_config(quantiles=(0.5,)),
    ReportConfig(decision_curve=DecisionCurveConfig(enabled=False)),
  )

  with pytest.raises(ValueError, match="decision-curve table"):
    write_decision_curve_csv(bundle, tmp_path)
