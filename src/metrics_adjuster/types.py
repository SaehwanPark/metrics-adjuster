"""Typed domain objects for adjusted metric computation."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class MetricName(StrEnum):
  """Supported adjusted metrics."""

  ATPR = "aTPR"
  APPV = "aPPV"
  ANB = "aNB"
  AHR = "aHR"


class ColumnSpec(BaseModel):
  """Column names required by the adjusted-metrics pipeline."""

  model_config = ConfigDict(frozen=True)

  group: str
  response: str
  risk: str
  id: str | None = None


class CalibrationConfig(BaseModel):
  """Configuration for within-group risk calibration."""

  model_config = ConfigDict(frozen=True)

  degree: int = Field(default=2, ge=1)
  transform: bool = True
  cv: bool = False
  k_folds: int = Field(default=5, ge=2)


class DensityRatioConfig(BaseModel):
  """Configuration for density-ratio estimation against a reference group."""

  model_config = ConfigDict(frozen=True)

  degree: int = Field(default=1, ge=1)
  transform: bool = False
  cv: bool = False
  k_folds: int = Field(default=5, ge=2)


class BootstrapConfig(BaseModel):
  """Configuration for bootstrap uncertainty estimates."""

  model_config = ConfigDict(frozen=True)

  enabled: bool = False
  iterations: int = Field(default=500, ge=1)
  alpha: float = Field(default=0.05, gt=0.0, lt=1.0)


class OutputConfig(BaseModel):
  """Optional files to persist intermediate pipeline outputs."""

  model_config = ConfigDict(frozen=True)

  calibration_path: Path | None = None
  density_ratio_path: Path | None = None


class ReportLabelConfig(BaseModel):
  """Optional display labels for human-readable reports."""

  model_config = ConfigDict(frozen=True)

  columns: dict[str, str] = Field(default_factory=dict)
  groups: dict[str, dict[str, str]] = Field(default_factory=dict)
  metrics: dict[str, str] = Field(default_factory=dict)


class ReportConfig(BaseModel):
  """Configuration for human-readable report rendering."""

  model_config = ConfigDict(frozen=True)

  title: str = "Adjusted Metrics Report"
  subtitle: str | None = None
  x_scale: Literal["probability", "log_odds"] = "probability"
  labels: ReportLabelConfig = ReportLabelConfig()
  max_cutoff_lines: int = Field(default=8, ge=0)
  density_points: int = Field(default=200, ge=20)


class MetricConfig(BaseModel):
  """Validated top-level adjusted-metrics configuration."""

  model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

  columns: ColumnSpec
  ref_group: Any
  quantiles: tuple[float, ...]
  metrics: tuple[MetricName, ...] = tuple(MetricName)
  calibration: CalibrationConfig = CalibrationConfig()
  density_ratio: DensityRatioConfig = DensityRatioConfig()
  bootstrap: BootstrapConfig = BootstrapConfig()
  output: OutputConfig = OutputConfig()
  random_state: int | None = None

  @field_validator("quantiles")
  @classmethod
  def validate_quantiles(cls, quantiles: tuple[float, ...]) -> tuple[float, ...]:
    if not quantiles:
      raise ValueError("at least one quantile is required")
    invalid = [q for q in quantiles if q <= 0.0 or q >= 1.0]
    if invalid:
      raise ValueError("quantiles must be strictly between 0 and 1")
    return quantiles

  @field_validator("metrics", mode="before")
  @classmethod
  def coerce_metrics(cls, metrics: Any) -> tuple[MetricName, ...]:
    if metrics is None:
      return tuple(MetricName)
    return tuple(MetricName(metric) for metric in metrics)

  @model_validator(mode="after")
  def validate_bootstrap_cv(self) -> MetricConfig:
    return self
