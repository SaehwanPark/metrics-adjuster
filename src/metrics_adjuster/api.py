"""Public Python API for adjusted metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from metrics_adjuster.core import (
  MetricFrames,
  QuantilesFrom,
  attach_pipeline_intermediates,
  run_adjusted_metrics,
  run_metric_pipeline,
)
from metrics_adjuster.reporting import ReportBundle, build_report_bundle
from metrics_adjuster.types import (
  DEFAULT_METRICS,
  BootstrapConfig,
  CalibrationConfig,
  ColumnSpec,
  DensityRatioConfig,
  MetricConfig,
  MetricName,
  OutputConfig,
  ReportConfig,
)


def adjusted_metrics(
  df: pd.DataFrame,
  config: MetricConfig,
  quantiles_from: QuantilesFrom | None = None,
) -> MetricFrames:
  """Compute adjusted metrics from a typed configuration object."""
  return run_adjusted_metrics(df, config, quantiles_from)


def adjusted_metrics_report(
  df: pd.DataFrame,
  config: MetricConfig,
  report_config: ReportConfig | None = None,
  quantiles_from: QuantilesFrom | None = None,
) -> ReportBundle:
  """Compute adjusted metrics and render reusable report components."""
  pipeline = run_metric_pipeline(df, config, quantiles_from)
  resolved_report_config = report_config or ReportConfig()
  return build_report_bundle(
    attach_pipeline_intermediates(pipeline, config),
    pipeline.weighted,
    config,
    resolved_report_config,
  )


def compute_adjusted_metrics(
  df: pd.DataFrame,
  group_col: str,
  ref_group: Any,
  response_col: str,
  orig_risk_col: str,
  quantiles: list[float],
  thresholds: list[float] | None = None,
  quantiles_from: QuantilesFrom | None = None,
  idvar: str | None = "patienticn",
  metrics: list[str] | None = None,
  pairwise: bool = False,
  cal_degree: int = 2,
  dr_degree: int = 1,
  transform_cal: bool = True,
  transform_dr: bool = False,
  save_cal: str | Path | None = None,
  save_dr: str | Path | None = None,
  cv: bool = False,
  se_boot: bool = False,
  n_boot: int = 500,
  alpha: float = 0.05,
  random_state: int | None = None,
) -> dict[str, Any]:
  """Backward-compatible wrapper around the typed API."""
  config = MetricConfig(
    columns=ColumnSpec(group=group_col, response=response_col, risk=orig_risk_col, id=idvar),
    ref_group=ref_group,
    quantiles=tuple(quantiles),
    thresholds=tuple(thresholds or ()),
    metrics=tuple(MetricName(metric) for metric in metrics) if metrics else DEFAULT_METRICS,
    pairwise=pairwise,
    calibration=CalibrationConfig(degree=cal_degree, transform=transform_cal, cv=cv),
    density_ratio=DensityRatioConfig(degree=dr_degree, transform=transform_dr, cv=cv),
    bootstrap=BootstrapConfig(enabled=se_boot, iterations=n_boot, alpha=alpha),
    output=OutputConfig(
      calibration_path=Path(save_cal) if save_cal is not None else None,
      density_ratio_path=Path(save_dr) if save_dr is not None else None,
    ),
    random_state=random_state,
  )
  return adjusted_metrics(df, config, quantiles_from).as_dict()
