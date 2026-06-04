from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from metrics_adjuster import (
  CalibrationConfig,
  ColumnSpec,
  DensityRatioConfig,
  MetricConfig,
  MetricName,
  adjusted_metrics,
  compute_adjusted_metrics,
)
from metrics_adjuster.core import (
  compute_logit,
  high_risk_indicator,
  metric_frame_at_threshold,
  resolve_quantile_mask,
  safe_divide,
)
from metrics_adjuster.synthetic import generate_synthetic_metrics_data


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
