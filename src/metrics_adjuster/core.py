"""Pure adjusted-metric kernels and functional pipeline stages."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from metrics_adjuster.types import (
  BootstrapConfig,
  CalibrationConfig,
  DensityRatioConfig,
  MetricConfig,
  MetricName,
)

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int_]
QuantilesFrom: TypeAlias = pd.Series | Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class StageResult:
  """Data returned by a pipeline stage."""

  data: pd.DataFrame
  diagnostics: dict[str, Any]


@dataclass(frozen=True)
class MetricFrames:
  """Data returned by the public adjusted-metrics API."""

  metrics: dict[str, pd.DataFrame]
  bootstrap: pd.DataFrame | None = None
  calibrated: pd.DataFrame | None = None
  weighted: pd.DataFrame | None = None

  def as_dict(self) -> dict[str, Any]:
    result: dict[str, Any] = {"metrics": self.metrics}
    if self.bootstrap is not None:
      result["boot"] = self.bootstrap
    return result


@dataclass(frozen=True)
class PipelineFrames:
  """Internal data needed by metrics and optional report rendering."""

  metrics: MetricFrames
  calibrated: pd.DataFrame
  weighted: pd.DataFrame


def finite_or_nan(value: float) -> float:
  """Return a finite float or NaN when a division-like result is undefined."""
  return value if np.isfinite(value) else float("nan")


def safe_divide(numerator: float, denominator: float) -> float:
  """Divide two scalars, returning NaN for zero denominators."""
  if denominator == 0:
    return float("nan")
  return finite_or_nan(numerator / denominator)


def compute_logit(probabilities: FloatArray, eps: float = 1e-6) -> FloatArray:
  """Compute clipped log-odds for probability-like values."""
  clipped = np.clip(probabilities.astype(float), eps, 1.0 - eps)
  return np.asarray(np.log(clipped / (1.0 - clipped)), dtype=np.float64)


def maybe_logit(values: FloatArray, enabled: bool) -> FloatArray:
  """Optionally apply a logit transform."""
  return compute_logit(values) if enabled else values.astype(float)


def high_risk_indicator(values: FloatArray, threshold: float) -> IntArray:
  """Flag observations with risk strictly greater than a threshold."""
  return (values > threshold).astype(int)


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
  """Raise a stable error when required columns are missing."""
  missing = sorted(set(columns).difference(df.columns))
  if missing:
    raise ValueError(f"missing required columns: {missing}")


def require_binary_response(values: pd.Series, response_col: str) -> None:
  """Require a binary 0/1 response column."""
  observed = {int(value) for value in values.dropna().unique()}
  if not observed.issubset({0, 1}):
    raise ValueError(f"{response_col!r} must contain only binary 0/1 values")


def require_reference_group(df: pd.DataFrame, group_col: str, ref_group: Any) -> None:
  """Require the reference group to be present."""
  if not (df[group_col] == ref_group).any():
    raise ValueError(f"reference group {ref_group!r} is absent from {group_col!r}")


def validate_input_frame(df: pd.DataFrame, config: MetricConfig) -> None:
  """Validate the public data contract before running the pipeline."""
  columns = [config.columns.group, config.columns.response, config.columns.risk]
  if config.columns.id is not None:
    columns.append(config.columns.id)
  require_columns(df, columns)
  require_binary_response(df[config.columns.response], config.columns.response)
  require_reference_group(df, config.columns.group, config.ref_group)


def resolve_quantile_mask(
  df: pd.DataFrame,
  quantiles_from: QuantilesFrom | None,
) -> pd.Series | None:
  """Normalize an optional quantile-source mask while refusing silent reindexing."""
  if quantiles_from is None:
    return None
  mask = quantiles_from(df) if callable(quantiles_from) else quantiles_from
  if not isinstance(mask, pd.Series):
    raise TypeError("quantiles_from must be a boolean Series or callable")
  if mask.dtype != bool:
    raise TypeError("quantiles_from mask must have boolean dtype")
  if not mask.index.equals(df.index):
    raise ValueError("quantiles_from mask index must exactly match df.index")
  return mask


def resolve_bootstrap_quantile_mask(
  boot_df: pd.DataFrame,
  quantiles_from: QuantilesFrom | None,
  orig_index_col: str = "_orig_index",
) -> pd.Series | None:
  """Map an original-data quantile mask onto a bootstrap sample."""
  if quantiles_from is None:
    return None
  if callable(quantiles_from):
    return resolve_quantile_mask(boot_df, quantiles_from)
  if orig_index_col not in boot_df.columns:
    raise ValueError(f"bootstrap frame is missing {orig_index_col!r}")
  mapped = boot_df[orig_index_col].map(quantiles_from)
  if mapped.isna().any():
    raise ValueError("quantiles_from could not be mapped to bootstrap sample")
  return mapped.astype(bool)


def make_rng(seed: int | None) -> np.random.Generator:
  """Create a local generator so randomness is explicit and reproducible."""
  return np.random.default_rng(seed)


def assign_group_folds(
  df: pd.DataFrame,
  group_col: str,
  k_folds: int,
  rng: np.random.Generator,
) -> IntArray:
  """Assign fold ids independently within each group."""
  folds = np.zeros(len(df), dtype=int)
  for _, index in df.groupby(group_col, sort=False).groups.items():
    positions = df.index.get_indexer(index)
    folds[positions] = rng.integers(0, k_folds, size=len(positions))
  return folds


def polynomial_matrix(x: FloatArray, degree: int) -> FloatArray:
  """Create a simple polynomial design matrix without an intercept column."""
  flat = x.astype(float).reshape(-1)
  return np.column_stack([flat ** power for power in range(1, degree + 1)]).astype(float)


def sigmoid(values: FloatArray) -> FloatArray:
  """Compute a numerically stable sigmoid."""
  clipped = np.clip(values, -35.0, 35.0)
  return (1.0 / (1.0 + np.exp(-clipped))).astype(float)


def fit_logistic_coefficients(
  x_train: FloatArray,
  y_train: FloatArray,
  degree: int,
  max_iter: int = 1000,
  learning_rate: float = 0.1,
  l2: float = 1e-6,
) -> FloatArray:
  """Fit a small deterministic logistic model with gradient descent."""
  design = np.column_stack([np.ones(len(x_train)), polynomial_matrix(x_train, degree)])
  scale = np.maximum(design[:, 1:].std(axis=0), 1e-8)
  design_scaled = design.copy()
  design_scaled[:, 1:] = design[:, 1:] / scale
  beta = np.zeros(design_scaled.shape[1], dtype=float)
  for _ in range(max_iter):
    pred = sigmoid(design_scaled @ beta)
    gradient = (design_scaled.T @ (pred - y_train)) / len(y_train)
    gradient[1:] += l2 * beta[1:]
    step = learning_rate * gradient
    beta -= step
    if float(np.linalg.norm(step)) < 1e-8:
      break
  beta_unscaled = beta.copy()
  beta_unscaled[1:] = beta[1:] / scale
  return beta_unscaled


def predict_logistic(x_test: FloatArray, coefficients: FloatArray, degree: int) -> FloatArray:
  """Predict probabilities from fitted logistic coefficients."""
  design = np.column_stack([np.ones(len(x_test)), polynomial_matrix(x_test, degree)])
  return sigmoid(design @ coefficients)


def constant_probability(y: FloatArray) -> float | None:
  """Return a constant class probability when y has fewer than two classes."""
  unique = np.unique(y)
  if len(unique) == 1:
    return float(unique[0])
  return None


def fit_predict_logistic(
  x_train: FloatArray,
  y_train: FloatArray,
  x_test: FloatArray,
  degree: int,
) -> FloatArray:
  """Fit logistic regression and predict positive-class probabilities."""
  constant = constant_probability(y_train)
  if constant is not None:
    return np.repeat(constant, len(x_test)).astype(float)
  coefficients = fit_logistic_coefficients(x_train, y_train, degree)
  return predict_logistic(x_test, coefficients, degree)


def cross_validated_predictions(
  x: FloatArray,
  y: FloatArray,
  folds: IntArray,
  degree: int,
  k_folds: int,
) -> tuple[FloatArray, list[float]]:
  """Predict each row from models that exclude its fold."""
  predictions = np.zeros(len(y), dtype=float)
  losses: list[float] = []
  for fold_id in range(k_folds):
    test_mask = folds == fold_id
    if not test_mask.any():
      continue
    train_mask = ~test_mask
    predictions[test_mask] = fit_predict_logistic(
      x[train_mask],
      y[train_mask],
      x[test_mask],
      degree,
    )
    losses.append(float(np.mean((predictions[test_mask] - y[test_mask]) ** 2)))
  return predictions, losses


def full_fit_predictions(x: FloatArray, y: FloatArray, degree: int) -> FloatArray:
  """Predict all rows from a model fit on all rows."""
  return fit_predict_logistic(x, y, x, degree)


def predict_by_group(
  df: pd.DataFrame,
  group_col: str,
  x: FloatArray,
  y: FloatArray,
  config: CalibrationConfig | DensityRatioConfig,
  rng: np.random.Generator,
) -> tuple[FloatArray, list[float]]:
  """Run full-fit or CV predictions separately for each group."""
  predictions = np.zeros(len(df), dtype=float)
  losses: list[float] = []
  folds = assign_group_folds(df, group_col, config.k_folds, rng) if config.cv else None
  for _, index in df.groupby(group_col, sort=False).groups.items():
    positions = df.index.get_indexer(index)
    if folds is None:
      group_predictions = full_fit_predictions(x[positions], y[positions], config.degree)
      group_losses: list[float] = []
    else:
      group_predictions, group_losses = cross_validated_predictions(
        x[positions],
        y[positions],
        folds[positions],
        config.degree,
        config.k_folds,
      )
    predictions[positions] = group_predictions
    losses.extend(group_losses)
  return predictions, losses


def calibrate_risk(
  df: pd.DataFrame,
  group_col: str,
  response_col: str,
  risk_col: str,
  config: CalibrationConfig,
  random_state: int | None,
) -> StageResult:
  """Add a calibrated risk column using within-group logistic calibration."""
  x = maybe_logit(df[risk_col].to_numpy(dtype=float), config.transform)
  y = df[response_col].to_numpy(dtype=float)
  predictions, losses = predict_by_group(
    df.reset_index(drop=True),
    group_col,
    x,
    y,
    config,
    make_rng(random_state),
  )
  result = df.copy()
  result["cal_risk"] = predictions
  diagnostics = {"calibration_mspe": float(np.mean(losses)) if losses else None}
  return StageResult(result, diagnostics)


def density_ratio_for_pair(
  x: FloatArray,
  is_ref: FloatArray,
  n_group: int,
  n_ref: int,
  config: DensityRatioConfig,
  rng: np.random.Generator,
) -> tuple[FloatArray, float | None]:
  """Estimate f_ref(r) / f_group(r) through a binary classifier."""
  if config.cv:
    pair_df = pd.DataFrame({"pair": np.repeat("pair", len(x))})
    folds = assign_group_folds(pair_df, "pair", config.k_folds, rng)
    p_ref, losses = cross_validated_predictions(x, is_ref, folds, config.degree, config.k_folds)
    mspe = float(np.mean(losses)) if losses else None
  else:
    p_ref = full_fit_predictions(x, is_ref, config.degree)
    mspe = None
  odds = np.clip(p_ref, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - p_ref, 1e-6, 1.0)
  return odds * (n_group / n_ref), mspe


def estimate_density_ratio(
  df: pd.DataFrame,
  group_col: str,
  ref_group: Any,
  cal_risk_col: str,
  config: DensityRatioConfig,
  random_state: int | None,
) -> StageResult:
  """Add density ratios for each non-reference group."""
  x_all = maybe_logit(df[cal_risk_col].to_numpy(dtype=float), config.transform)
  groups = df[group_col].to_numpy(dtype=object)
  n_ref = int(np.sum(groups == ref_group))
  density = np.ones(len(df), dtype=float)
  mspe_by_group: dict[Any, float] = {}
  rng = make_rng(random_state)

  for group_id in pd.unique(groups):
    if group_id == ref_group:
      continue
    pair_mask = (groups == ref_group) | (groups == group_id)
    pair_density, mspe = density_ratio_for_pair(
      x_all[pair_mask],
      (groups[pair_mask] == ref_group).astype(float),
      int(np.sum(groups == group_id)),
      n_ref,
      config,
      rng,
    )
    density[np.where(pair_mask)[0]] = pair_density
    if mspe is not None:
      mspe_by_group[group_id] = mspe

  density[groups == ref_group] = 1.0
  result = df.copy()
  result["dens_ratio"] = density
  return StageResult(result, {"density_ratio_mspe_by_group": mspe_by_group or None})


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
  """Compute a ratio-of-means weighted mean."""
  return safe_divide(float(np.mean(values * weights)), float(np.mean(weights)))


def traditional_metric_name(metric: MetricName) -> str:
  """Return the conventional unadjusted companion name for an adjusted metric."""
  return metric.value[1:]


def group_traditional_metric_value(
  group_df: pd.DataFrame,
  metric: MetricName,
  tau: float,
) -> float:
  """Compute one conventional unadjusted metric for one group and threshold."""
  y = group_df["_response"].astype(float)
  high_risk = group_df["_high_risk"]
  if metric == MetricName.ATPR:
    return safe_divide(float(np.mean(y * high_risk)), float(np.mean(y)))
  if metric == MetricName.APPV:
    return safe_divide(float(np.mean(y * high_risk)), float(np.mean(high_risk)))
  if metric == MetricName.ANB:
    benefit = np.mean(high_risk * y)
    harm = np.mean(high_risk * (1.0 - y)) * tau / (1.0 - tau)
    return float(benefit - harm)
  return float(np.mean(high_risk))


def group_metric_value(group_df: pd.DataFrame, metric: MetricName, tau: float) -> float:
  """Compute one adjusted metric for one group and threshold."""
  y_hat = group_df["cal_risk"]
  high_risk = group_df["_high_risk"]
  weight = group_df["dens_ratio"]
  if metric == MetricName.ATPR:
    return safe_divide(
      float(np.mean(y_hat * high_risk * weight)),
      float(np.mean(y_hat * weight)),
    )
  if metric == MetricName.APPV:
    return safe_divide(
      float(np.mean(y_hat * high_risk * weight)),
      float(np.mean(high_risk * weight)),
    )
  if metric == MetricName.ANB:
    benefit = np.mean(high_risk * y_hat * weight)
    harm = np.mean(high_risk * (1.0 - y_hat) * weight) * tau / (1.0 - tau)
    return safe_divide(float(benefit - harm), float(np.mean(weight)))
  return weighted_mean(high_risk, weight)


def metric_frame_at_threshold(
  df: pd.DataFrame,
  group_col: str,
  risk_col: str,
  response_col: str,
  metric: MetricName,
  quantile: float,
  tau: float,
) -> pd.DataFrame:
  """Compute conventional and adjusted metric values for all groups at a threshold."""
  with_indicator = df.copy()
  with_indicator["_response"] = with_indicator[response_col].astype(float)
  with_indicator["_high_risk"] = high_risk_indicator(
    with_indicator[risk_col].to_numpy(dtype=float),
    tau,
  )
  unadjusted_name = traditional_metric_name(metric)
  records = [
    {
      group_col: group_id,
      "quantile": quantile,
      "tau": tau,
      unadjusted_name: group_traditional_metric_value(group_df, metric, tau),
      metric.value: group_metric_value(group_df, metric, tau),
    }
    for group_id, group_df in with_indicator.groupby(group_col, sort=True)
  ]
  return pd.DataFrame.from_records(records)


def threshold_for_quantile(
  df: pd.DataFrame,
  risk_col: str,
  quantile: float,
  mask: pd.Series | None,
) -> float:
  """Compute a risk threshold from all rows or a caller-selected subset."""
  source = df.loc[mask, risk_col] if mask is not None else df[risk_col]
  return float(source.quantile(quantile))


def metric_frame_across_quantiles(
  df: pd.DataFrame,
  group_col: str,
  risk_col: str,
  response_col: str,
  metric: MetricName,
  quantiles: tuple[float, ...],
  quantiles_from: QuantilesFrom | None,
) -> pd.DataFrame:
  """Compute a metric for every requested quantile."""
  mask = resolve_quantile_mask(df, quantiles_from)
  frames = [
    metric_frame_at_threshold(
      df,
      group_col,
      risk_col,
      response_col,
      metric,
      quantile,
      threshold_for_quantile(df, risk_col, quantile, mask),
    )
    for quantile in quantiles
  ]
  return pd.concat(frames, ignore_index=True)


def stratified_bootstrap_sample(
  df: pd.DataFrame,
  group_col: str,
  rng: np.random.Generator,
) -> pd.DataFrame:
  """Sample rows with replacement within each group and retain original indices."""
  indexed = df.copy()
  indexed["_orig_index"] = indexed.index
  samples = []
  for _, group_df in indexed.groupby(group_col, sort=False):
    positions = rng.integers(0, len(group_df), size=len(group_df))
    samples.append(group_df.iloc[positions].copy())
  return pd.concat(samples, ignore_index=True)


def bootstrap_records(
  df: pd.DataFrame,
  config: MetricConfig,
  quantiles_from: QuantilesFrom | None,
  bootstrap: BootstrapConfig,
) -> pd.DataFrame:
  """Generate long-form bootstrap metric records."""
  rng = make_rng(config.random_state)
  records: list[dict[str, Any]] = []
  for _ in range(bootstrap.iterations):
    seed = int(rng.integers(0, np.iinfo(np.int32).max))
    sample = stratified_bootstrap_sample(df, config.columns.group, make_rng(seed))
    cal = calibrate_risk(
      sample,
      config.columns.group,
      config.columns.response,
      config.columns.risk,
      config.calibration.model_copy(update={"cv": False}),
      seed,
    )
    dr = estimate_density_ratio(
      cal.data,
      config.columns.group,
      config.ref_group,
      "cal_risk",
      config.density_ratio.model_copy(update={"cv": False}),
      seed,
    )
    mask = resolve_bootstrap_quantile_mask(dr.data, quantiles_from)
    for quantile in config.quantiles:
      tau = threshold_for_quantile(dr.data, config.columns.risk, quantile, mask)
      for metric in config.metrics:
        metric_df = metric_frame_at_threshold(
          dr.data,
          config.columns.group,
          config.columns.risk,
          config.columns.response,
          metric,
          quantile,
          tau,
        )
        records.extend(
          {
            "metric": metric.value,
            config.columns.group: row[config.columns.group],
            "quantile": quantile,
            "tau": tau,
            "original_value": row[traditional_metric_name(metric)],
            "adjusted_value": row[metric.value],
            "value": row[metric.value],
          }
          for _, row in metric_df.iterrows()
        )
  return pd.DataFrame.from_records(records)


def summarize_bootstrap(
  boot_df: pd.DataFrame,
  group_col: str,
  alpha: float,
) -> pd.DataFrame:
  """Summarize long-form bootstrap records by metric, group, and quantile."""
  return (
    boot_df.groupby(["metric", group_col, "quantile"], dropna=False)
    .agg(
      bootmean=("value", "mean"),
      bootse=("value", "std"),
      boot_lower=("value", lambda x: float(np.quantile(x, alpha / 2.0))),
      boot_upper=("value", lambda x: float(np.quantile(x, 1.0 - alpha / 2.0))),
      n=("value", "count"),
    )
    .reset_index()
  )


def attach_bootstrap_summary(
  metric_df: pd.DataFrame,
  boot_summary: pd.DataFrame,
  metric: MetricName,
  group_col: str,
) -> pd.DataFrame:
  """Attach bootstrap summaries to one wide metric frame."""
  summary = boot_summary[boot_summary["metric"] == metric.value].drop(columns="metric")
  return metric_df.merge(summary, on=[group_col, "quantile"], how="left")


def persist_stage_outputs(config: MetricConfig, cal: StageResult, dr: StageResult) -> None:
  """Persist optional intermediate outputs at the IO boundary."""
  id_col = config.columns.id
  if config.output.calibration_path is not None:
    cols = ["cal_risk"] if id_col is None else [id_col, "cal_risk"]
    calibration_path = config.output.calibration_path
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    cal.data[cols].to_parquet(calibration_path, index=False)
  if config.output.density_ratio_path is not None:
    cols = ["dens_ratio"] if id_col is None else [id_col, "dens_ratio"]
    weights_path = config.output.density_ratio_path
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    dr.data[cols].to_parquet(weights_path, index=False)


def run_metric_pipeline(
  df: pd.DataFrame,
  config: MetricConfig,
  quantiles_from: QuantilesFrom | None = None,
) -> PipelineFrames:
  """Run pipeline stages and retain intermediate frames for optional reporting."""
  validate_input_frame(df, config)
  clean_df = df.dropna(
    subset=[config.columns.group, config.columns.response, config.columns.risk]
  ).copy()
  cal = calibrate_risk(
    clean_df,
    config.columns.group,
    config.columns.response,
    config.columns.risk,
    config.calibration,
    config.random_state,
  )
  dr = estimate_density_ratio(
    cal.data,
    config.columns.group,
    config.ref_group,
    "cal_risk",
    config.density_ratio,
    config.random_state,
  )
  persist_stage_outputs(config, cal, dr)
  metrics = {
    metric.value: metric_frame_across_quantiles(
      dr.data,
      config.columns.group,
      config.columns.risk,
      config.columns.response,
      metric,
      config.quantiles,
      quantiles_from,
    )
    for metric in config.metrics
  }
  boot_df = None
  if config.bootstrap.enabled:
    boot_df = bootstrap_records(clean_df, config, quantiles_from, config.bootstrap)
    summary = summarize_bootstrap(boot_df, config.columns.group, config.bootstrap.alpha)
    metrics = {
      metric.value: attach_bootstrap_summary(
        metrics[metric.value],
        summary,
        metric,
        config.columns.group,
      )
      for metric in config.metrics
    }
  return PipelineFrames(
    metrics=MetricFrames(metrics=metrics, bootstrap=boot_df),
    calibrated=cal.data,
    weighted=dr.data,
  )


def attach_pipeline_intermediates(
  pipeline: PipelineFrames,
  config: MetricConfig,
) -> MetricFrames:
  """Return metric frames, optionally including in-memory pipeline intermediates."""
  result = pipeline.metrics
  if not config.output.include_intermediates:
    return result
  return MetricFrames(
    metrics=result.metrics,
    bootstrap=result.bootstrap,
    calibrated=pipeline.calibrated,
    weighted=pipeline.weighted,
  )


def run_adjusted_metrics(
  df: pd.DataFrame,
  config: MetricConfig,
  quantiles_from: QuantilesFrom | None = None,
) -> MetricFrames:
  """Run calibration, density-ratio estimation, and adjusted metric computation."""
  pipeline = run_metric_pipeline(df, config, quantiles_from)
  return attach_pipeline_intermediates(pipeline, config)
