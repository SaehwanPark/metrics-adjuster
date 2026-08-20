from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from metrics_adjuster import (
  BootstrapConfig,
  CalibrationConfig,
  ColumnSpec,
  DensityRatioConfig,
  MetricConfig,
  MetricName,
  OutputConfig,
  adjusted_metrics,
  compute_adjusted_metrics,
)
from metrics_adjuster.core import (
  calibrated_metric_name,
  compute_logit,
  high_risk_indicator,
  metric_frame_at_threshold,
  plug_in_adjusted_net_benefit,
  resolve_quantile_mask,
  safe_divide,
)
from metrics_adjuster.synthetic import (
  generate_synthetic_metrics_data,
  generate_xiaoyi_simulation_data,
)


def small_metric_frame() -> pd.DataFrame:
  return pd.DataFrame(
    {
      "group": ["ref", "ref", "alt", "alt"],
      "risk": [0.1, 0.8, 0.4, 0.9],
      "cal_risk": [0.2, 0.7, 0.5, 0.6],
      "dens_ratio": [1.0, 1.0, 2.0, 2.0],
      "outcome": [0, 1, 1, 0],
    }
  )


def test_safe_divide_contract() -> None:
  assert safe_divide(6.0, 3.0) == 2.0
  assert math.isnan(safe_divide(1.0, 0.0))


def test_logit_clips_boundary_probabilities() -> None:
  values = compute_logit(np.array([0.0, 0.5, 1.0]))
  assert np.isfinite(values).all()
  assert values[0] < 0
  assert values[1] == pytest.approx(0.0)
  assert values[2] > 0


def test_high_risk_indicator_is_strictly_greater_than_threshold() -> None:
  indicator = high_risk_indicator(np.array([0.2, 0.5, 0.7]), 0.5)
  assert indicator.tolist() == [0, 0, 1]


def test_metric_frame_at_threshold_matches_hand_calculated_atpr() -> None:
  result = metric_frame_at_threshold(
    small_metric_frame(),
    group_col="group",
    risk_col="risk",
    response_col="outcome",
    metric=MetricName.ATPR,
    quantile=0.5,
    tau=0.5,
  )
  ref_value = result.loc[result["group"] == "ref", "aTPR"].item()
  alt_value = result.loc[result["group"] == "alt", "aTPR"].item()
  ref_traditional = result.loc[result["group"] == "ref", "TPR"].item()
  alt_traditional = result.loc[result["group"] == "alt", "TPR"].item()
  assert ref_traditional == pytest.approx(1.0)
  assert alt_traditional == pytest.approx(0.0)
  assert ref_value == pytest.approx(0.7 / (0.2 + 0.7))
  assert alt_value == pytest.approx((0.6 * 2.0) / ((0.5 * 2.0) + (0.6 * 2.0)))


@pytest.mark.parametrize(
  ("metric", "calibrated_name", "ref_calibrated", "alt_calibrated"),
  [
    (MetricName.ATPR, "cTPR", 0.7 / 0.9, 0.6 / 1.1),
    (MetricName.APPV, "cPPV", 0.7, 0.6),
    (MetricName.ANB, "cNB", 0.35 - 0.15, 0.3 - 0.2),
    (MetricName.AHR, "cHR", 0.5, 0.5),
  ],
)
def test_metric_frame_can_include_calibrated_unweighted_values(
  metric: MetricName,
  calibrated_name: str,
  ref_calibrated: float,
  alt_calibrated: float,
) -> None:
  result = metric_frame_at_threshold(
    small_metric_frame(),
    group_col="group",
    risk_col="risk",
    response_col="outcome",
    metric=metric,
    quantile=float("nan"),
    tau=0.5,
    include_calibrated_metrics=True,
    ref_group="ref",
  )
  ref_row = result[result["group"] == "ref"].iloc[0]
  alt_row = result[result["group"] == "alt"].iloc[0]

  assert calibrated_metric_name(metric) == calibrated_name
  assert calibrated_name in result.columns
  assert ref_row[calibrated_name] == pytest.approx(ref_calibrated)
  assert alt_row[calibrated_name] == pytest.approx(alt_calibrated)


def test_plug_in_adjusted_net_benefit_matches_atpr_afpr_identity() -> None:
  tau = 0.5
  mu0 = 0.5
  atpr = 0.6 / 1.1
  afpr = 0.8 / 1.8
  expected = atpr * mu0 - afpr * (1.0 - mu0) * tau / (1.0 - tau)
  assert plug_in_adjusted_net_benefit(atpr, afpr, mu0, tau) == pytest.approx(expected)


def test_metric_frame_anb_uses_atpr_afpr_and_reference_prevalence() -> None:
  result = metric_frame_at_threshold(
    small_metric_frame(),
    group_col="group",
    risk_col="risk",
    response_col="outcome",
    metric=MetricName.ANB,
    quantile=float("nan"),
    tau=0.5,
    ref_group="ref",
  )
  alt_row = result[result["group"] == "alt"].iloc[0]
  atpr = 0.6 / 1.1
  afpr = 0.8 / 1.8
  mu0 = 0.5
  expected = plug_in_adjusted_net_benefit(atpr, afpr, mu0, 0.5)
  old_weighted_mean_estimator = 0.1

  assert alt_row["NB"] == pytest.approx(-0.5)
  assert alt_row["aNB"] == pytest.approx(expected)
  assert alt_row["aNB"] != pytest.approx(old_weighted_mean_estimator)


def test_metric_frame_anb_requires_reference_group() -> None:
  with pytest.raises(ValueError, match="ref_group is required for aNB"):
    metric_frame_at_threshold(
      small_metric_frame(),
      group_col="group",
      risk_col="risk",
      response_col="outcome",
      metric=MetricName.ANB,
      quantile=float("nan"),
      tau=0.5,
    )


@pytest.mark.parametrize(
  ("metric", "original_name", "ref_original", "alt_original", "alt_adjusted"),
  [
    (MetricName.AFPR, "FPR", 0.0, 1.0, 0.8 / 1.8),
    (MetricName.ANPV, "NPV", 1.0, 0.0, 1.0 / 2.0),
    (MetricName.ABSP, "BSP", 0.8, 0.4, 1.48 / 2.2),
    (MetricName.ABSN, "BSN", 0.1, 0.9, 1.12 / 1.8),
    (MetricName.ASP, "SP", 0.5, 0.5, 0.5),
  ],
)
def test_legacy_metric_formulas_match_hand_calculations(
  metric: MetricName,
  original_name: str,
  ref_original: float,
  alt_original: float,
  alt_adjusted: float,
) -> None:
  result = metric_frame_at_threshold(
    small_metric_frame(),
    group_col="group",
    risk_col="risk",
    response_col="outcome",
    metric=metric,
    quantile=float("nan"),
    tau=0.5,
  )
  ref_row = result[result["group"] == "ref"].iloc[0]
  alt_row = result[result["group"] == "alt"].iloc[0]

  assert ref_row[original_name] == pytest.approx(ref_original)
  assert alt_row[original_name] == pytest.approx(alt_original)
  assert alt_row[metric.value] == pytest.approx(alt_adjusted)


def test_quantile_mask_refuses_misaligned_series() -> None:
  df = pd.DataFrame({"risk": [0.1, 0.2]}, index=[10, 11])
  mask = pd.Series([True, False], index=[0, 1])
  with pytest.raises(ValueError, match="exactly match"):
    resolve_quantile_mask(df, mask)


def test_typed_api_returns_all_requested_metric_frames() -> None:
  df = generate_synthetic_metrics_data(n=120, seed=1)
  config = MetricConfig(
    columns=ColumnSpec(group="group", response="outcome", risk="risk", id="patient_id"),
    ref_group="ref",
    quantiles=(0.25, 0.5, 0.75),
    metrics=(MetricName.ATPR, MetricName.AHR),
    calibration=CalibrationConfig(degree=1, cv=False),
    density_ratio=DensityRatioConfig(degree=1, cv=False),
    random_state=1,
  )
  result = adjusted_metrics(df, config)
  assert set(result.metrics) == {"aTPR", "aHR"}
  assert result.metrics["aTPR"].shape[0] == 6
  assert result.metrics["aTPR"]["TPR"].between(0, 1).all()
  assert set(result.metrics["aTPR"].columns) == {"group", "quantile", "tau", "TPR", "aTPR"}
  assert result.metrics["aHR"]["aHR"].between(0, 1).all()


def test_typed_api_includes_calibrated_columns_only_when_requested() -> None:
  df = generate_synthetic_metrics_data(n=120, seed=31)
  default_config = MetricConfig(
    columns=ColumnSpec(group="group", response="outcome", risk="risk", id="patient_id"),
    ref_group="ref",
    quantiles=(0.5,),
    metrics=(MetricName.ATPR,),
    calibration=CalibrationConfig(degree=1, cv=False),
    density_ratio=DensityRatioConfig(degree=1, cv=False),
    random_state=31,
  )
  calibrated_config = default_config.model_copy(update={"include_calibrated_metrics": True})

  default = adjusted_metrics(df, default_config).metrics["aTPR"]
  calibrated = adjusted_metrics(df, calibrated_config).metrics["aTPR"]

  assert set(default.columns) == {"group", "quantile", "tau", "TPR", "aTPR"}
  assert set(calibrated.columns) == {"group", "quantile", "tau", "TPR", "cTPR", "aTPR"}


def test_fixed_thresholds_are_appended_without_changing_quantile_rows() -> None:
  df = generate_synthetic_metrics_data(n=120, seed=11)
  config = MetricConfig(
    columns=ColumnSpec(group="group", response="outcome", risk="risk", id="patient_id"),
    ref_group="ref",
    quantiles=(0.5,),
    thresholds=(0.3,),
    metrics=(MetricName.ATPR,),
    calibration=CalibrationConfig(degree=1, cv=False),
    density_ratio=DensityRatioConfig(degree=1, cv=False),
    random_state=11,
  )

  result = adjusted_metrics(df, config).metrics["aTPR"]

  assert result.shape[0] == 4
  assert result["tau"].notna().all()
  assert result["quantile"].isna().sum() == 2
  assert result.loc[result["quantile"].isna(), "tau"].eq(0.3).all()


def test_pairwise_delta_frame_is_optional_and_arithmetic_is_stable() -> None:
  df = generate_synthetic_metrics_data(n=120, seed=12)
  config = MetricConfig(
    columns=ColumnSpec(group="group", response="outcome", risk="risk", id="patient_id"),
    ref_group="ref",
    quantiles=(),
    thresholds=(0.3,),
    metrics=(MetricName.ATPR, MetricName.AFPR),
    pairwise=True,
    calibration=CalibrationConfig(degree=1, cv=False),
    density_ratio=DensityRatioConfig(degree=1, cv=False),
    random_state=12,
  )

  result = adjusted_metrics(df, config)

  assert result.pairwise is not None
  assert set(result.pairwise["metric"]) == {"aTPR", "aFPR"}
  assert result.pairwise["quantile"].isna().all()
  assert result.pairwise["tau"].eq(0.3).all()
  first = result.pairwise.iloc[0]
  assert first["delta"] == pytest.approx(first["comparison_value"] - first["reference_value"])
  assert first["adjusted_delta"] == pytest.approx(
    first["adjusted_comparison_value"] - first["reference_value"]
  )


def test_pairwise_and_bootstrap_include_calibrated_values_when_requested() -> None:
  df = generate_synthetic_metrics_data(n=120, seed=32)
  config = MetricConfig(
    columns=ColumnSpec(group="group", response="outcome", risk="risk", id="patient_id"),
    ref_group="ref",
    quantiles=(0.5,),
    metrics=(MetricName.ATPR,),
    pairwise=True,
    include_calibrated_metrics=True,
    calibration=CalibrationConfig(degree=1, cv=False),
    density_ratio=DensityRatioConfig(degree=1, cv=False),
    bootstrap=BootstrapConfig(enabled=True, iterations=3),
    random_state=32,
  )

  result = adjusted_metrics(df, config)

  assert result.pairwise is not None
  assert {"calibrated_comparison_value", "calibrated_delta"}.issubset(result.pairwise.columns)
  assert result.bootstrap is not None
  assert "calibrated_value" in result.bootstrap.columns


def test_legacy_api_wrapper_preserves_result_shape() -> None:
  df = generate_synthetic_metrics_data(n=120, seed=2)
  result = compute_adjusted_metrics(
    df=df,
    idvar="patient_id",
    group_col="group",
    ref_group="ref",
    response_col="outcome",
    orig_risk_col="risk",
    metrics=["aTPR"],
    quantiles=[0.25, 0.5],
    cal_degree=1,
    dr_degree=1,
    random_state=2,
  )
  assert list(result["metrics"]) == ["aTPR"]
  assert result["metrics"]["aTPR"].shape[0] == 4


def test_legacy_api_wrapper_returns_pairwise_for_fixed_thresholds() -> None:
  df = generate_synthetic_metrics_data(n=120, seed=22)
  result = compute_adjusted_metrics(
    df=df,
    idvar="patient_id",
    group_col="group",
    ref_group="ref",
    response_col="outcome",
    orig_risk_col="risk",
    quantiles=[],
    thresholds=[0.3],
    metrics=["aTPR", "aFPR"],
    pairwise=True,
    cal_degree=1,
    dr_degree=1,
    random_state=22,
  )

  assert set(result["metrics"]) == {"aTPR", "aFPR"}
  assert "pairwise" in result
  assert set(result["pairwise"]["metric"]) == {"aTPR", "aFPR"}


def test_invalid_response_values_fail_fast() -> None:
  df = generate_synthetic_metrics_data(n=40, seed=3)
  df.loc[0, "outcome"] = 2
  config = MetricConfig(
    columns=ColumnSpec(group="group", response="outcome", risk="risk"),
    ref_group="ref",
    quantiles=(0.5,),
    metrics=(MetricName.ATPR,),
  )
  with pytest.raises(ValueError, match="binary"):
    adjusted_metrics(df, config)


def test_invalid_threshold_values_fail_fast() -> None:
  with pytest.raises(ValueError, match="thresholds"):
    MetricConfig(
      columns=ColumnSpec(group="group", response="outcome", risk="risk"),
      ref_group="ref",
      quantiles=(),
      thresholds=(1.0,),
      metrics=(MetricName.ATPR,),
    )


def test_requires_at_least_one_quantile_or_threshold() -> None:
  with pytest.raises(ValueError, match="at least one quantile or threshold"):
    MetricConfig(
      columns=ColumnSpec(group="group", response="outcome", risk="risk"),
      ref_group="ref",
      quantiles=(),
      metrics=(MetricName.ATPR,),
    )


def test_include_intermediates_returns_weighted_pipeline_frame(tmp_path) -> None:
  df = generate_synthetic_metrics_data(n=120, seed=4)
  config = MetricConfig(
    columns=ColumnSpec(group="group", response="outcome", risk="risk", id="patient_id"),
    ref_group="ref",
    quantiles=(0.5,),
    metrics=(MetricName.ATPR,),
    calibration=CalibrationConfig(degree=1, cv=False),
    density_ratio=DensityRatioConfig(degree=1, cv=False),
    output=OutputConfig(
      calibration_path=tmp_path / "calibration.parquet",
      density_ratio_path=tmp_path / "weights.parquet",
      include_intermediates=True,
    ),
    random_state=4,
  )
  result = adjusted_metrics(df, config)

  assert result.calibrated is not None
  assert result.weighted is not None
  assert "cal_risk" in result.calibrated.columns
  assert {"cal_risk", "dens_ratio"}.issubset(result.weighted.columns)
  assert (tmp_path / "calibration.parquet").exists()
  assert (tmp_path / "weights.parquet").exists()
  ref_weights = result.weighted.loc[result.weighted["group"] == "ref", "dens_ratio"]
  assert ref_weights.eq(1.0).all()


def test_default_result_omits_intermediate_frames() -> None:
  df = generate_synthetic_metrics_data(n=80, seed=5)
  config = MetricConfig(
    columns=ColumnSpec(group="group", response="outcome", risk="risk"),
    ref_group="ref",
    quantiles=(0.5,),
    metrics=(MetricName.ATPR,),
  )
  result = adjusted_metrics(df, config)

  assert result.calibrated is None
  assert result.weighted is None


def test_xiaoyi_simulation_scenario_produces_legacy_pairwise_metrics() -> None:
  df = generate_xiaoyi_simulation_data(n=300, seed=123456)
  config = MetricConfig(
    columns=ColumnSpec(group="group", response="outcome", risk="risk", id="patient_id"),
    ref_group="0",
    quantiles=(),
    thresholds=(0.3,),
    metrics=(
      MetricName.ATPR,
      MetricName.AFPR,
      MetricName.APPV,
      MetricName.ANPV,
      MetricName.ABSP,
      MetricName.ABSN,
      MetricName.ASP,
    ),
    pairwise=True,
    calibration=CalibrationConfig(degree=1, cv=False),
    density_ratio=DensityRatioConfig(degree=1, cv=False),
    random_state=123456,
  )

  result = adjusted_metrics(df, config)

  assert result.pairwise is not None
  expected_metrics = {"aTPR", "aFPR", "aPPV", "aNPV", "aBSP", "aBSN", "aSP"}
  assert set(result.pairwise["metric"]) == expected_metrics
  numeric_cols = ["reference_value", "comparison_value", "adjusted_comparison_value"]
  assert np.isfinite(result.pairwise[numeric_cols].to_numpy(dtype=float)).all()


def test_xiaoyi_simulation_generator_validates_bounds() -> None:
  with pytest.raises(ValueError, match="lower and upper"):
    generate_xiaoyi_simulation_data(n=10, lower=0.9, upper=0.1)
