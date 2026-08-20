"""Generate the two full-cohort VA CAN multi-panel manuscript figures.

The script reads ``DATA`` from a dotenv file, runs the public
``metrics_adjuster`` API for every outcome/subgroup combination, and writes
aggregate PNG/PDF figures only. Row-level VA data and intermediates are never
persisted.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import os
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from dotenv import load_dotenv
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from metrics_adjuster import adjusted_metrics
from metrics_adjuster.types import (
  BootstrapConfig,
  CalibrationConfig,
  ColumnSpec,
  DensityRatioConfig,
  MetricConfig,
  MetricName,
  OutputConfig,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = REPOSITORY_ROOT / ".env"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "collaboration" / "jinbo-manuscript" / "figures"
INPUT_FILENAME = "atpr_input_20250620.parquet"
QUANTILES = tuple(round(value, 2) for value in np.arange(0.10, 1.00, 0.05))
RANDOM_SEED = 343
RISK_CLIP = 1e-6
LOG_RISK_EDGES = np.linspace(np.log10(RISK_CLIP), 0.0, 321)

REFERENCE_BLUE = "#0072B2"
VERMILION = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
ORANGE = "#E69F00"
GRAY = "#6B7280"


class LevelSpec(NamedTuple):
  """One configured subgroup value and its display treatment."""

  value: str
  label: str
  color: str


class GroupSpec(NamedTuple):
  """Scientific and display contract for one subgroup domain."""

  key: str
  title: str
  reference: str
  levels: tuple[LevelSpec, ...]


class OutcomeSpec(NamedTuple):
  """Outcome, risk, title, and output-name mapping."""

  response: str
  risk: str
  title: str
  output_stem: str


class PanelData(NamedTuple):
  """Plot-ready aggregate data for one subgroup row."""

  group_spec: GroupSpec
  calibration: pd.DataFrame
  densities: pd.DataFrame
  sensitivity: pd.DataFrame
  q95_tau: float


OUTCOME_SPECS = (
  OutcomeSpec(
    response="Hosp_1y",
    risk="pHosp_1y",
    title="1-Year Hospitalization",
    output_stem="va_can_hospitalization_multiplot",
  ),
  OutcomeSpec(
    response="Mort_1y",
    risk="pMort_1y",
    title="1-Year Mortality",
    output_stem="va_can_mortality_multiplot",
  ),
)

GROUP_SPECS = (
  GroupSpec(
    key="BLACK",
    title="Race",
    reference="0",
    levels=(
      LevelSpec("0", "White (reference)", REFERENCE_BLUE),
      LevelSpec("1", "Black", VERMILION),
    ),
  ),
  GroupSpec(
    key="HCC_dementia",
    title="Dementia",
    reference="0",
    levels=(
      LevelSpec("0", "No dementia (reference)", REFERENCE_BLUE),
      LevelSpec("99", "Dementia", VERMILION),
    ),
  ),
  GroupSpec(
    key="Prior1245",
    title="VA Priority Group",
    reference="99",
    levels=(
      LevelSpec("99", "Priority 5 (reference)", REFERENCE_BLUE),
      LevelSpec("0", "Default priority", VERMILION),
      LevelSpec("1", "Priority 1", GREEN),
      LevelSpec("2", "Priority 2", PURPLE),
      LevelSpec("4", "Priority 4", ORANGE),
    ),
  ),
  GroupSpec(
    key="urban",
    title="Geographic Region",
    reference="99",
    levels=(
      LevelSpec("99", "Rural (reference)", REFERENCE_BLUE),
      LevelSpec("1", "Urban", VERMILION),
      LevelSpec("2", "Urban status missing", GRAY),
    ),
  ),
  GroupSpec(
    key="sex",
    title="Sex",
    reference="99",
    levels=(
      LevelSpec("99", "Male (reference)", REFERENCE_BLUE),
      LevelSpec("1", "Female", VERMILION),
    ),
  ),
)

REQUIRED_COLUMNS = (
  "patienticn",
  "Hosp_1y",
  "pHosp_1y",
  "Mort_1y",
  "pMort_1y",
  *(spec.key for spec in GROUP_SPECS),
)


def resolve_input_path(env_file: Path) -> Path:
  """Resolve the VA CAN parquet through the configured ``DATA`` dotenv key."""
  load_dotenv(env_file, override=True)
  data_raw = os.getenv("DATA")
  if not data_raw:
    raise ValueError(f"DATA is not set in {env_file}")
  data_root = Path(data_raw).expanduser()
  candidates = (
    data_root / "va_can" / INPUT_FILENAME,
    data_root / INPUT_FILENAME,
  )
  for candidate in candidates:
    if candidate.is_file():
      return candidate.resolve()
  raise FileNotFoundError(
    f"{INPUT_FILENAME} was not found under DATA={data_root}; "
    f"checked {candidates[0]} and {candidates[1]}"
  )


def load_cohort(input_path: Path) -> pd.DataFrame:
  """Read only the columns required for the two manuscript figures."""
  available = set(pq.ParquetFile(input_path).schema.names)  # type: ignore[no-untyped-call]
  missing = sorted(set(REQUIRED_COLUMNS).difference(available))
  if missing:
    raise ValueError(f"missing required columns: {missing}")
  return pd.read_parquet(input_path, columns=list(REQUIRED_COLUMNS))


def normalize_group_id(value: object) -> str:
  """Normalize integer-like parquet group codes without changing text labels."""
  if isinstance(value, (int, np.integer)):
    return str(int(value))
  if isinstance(value, (float, np.floating)) and float(value).is_integer():
    return str(int(value))
  return str(value)


def complete_case_frame(
  cohort: pd.DataFrame,
  outcome: OutcomeSpec,
  group_spec: GroupSpec,
) -> pd.DataFrame:
  """Return an isolated outcome/subgroup analysis frame with string group IDs."""
  columns = ["patienticn", outcome.response, outcome.risk, group_spec.key]
  result = cohort[columns].dropna().copy()
  result[group_spec.key] = result[group_spec.key].map(normalize_group_id)
  observed = set(result[group_spec.key].unique())
  expected = {level.value for level in group_spec.levels}
  missing = sorted(expected.difference(observed))
  unexpected = sorted(observed.difference(expected))
  if missing or unexpected:
    raise ValueError(
      f"{group_spec.key} levels differ from the configured contract; "
      f"missing={missing}, unexpected={unexpected}"
    )
  return result


def build_metric_config(outcome: OutcomeSpec, group_spec: GroupSpec) -> MetricConfig:
  """Build the typed package configuration matching the legacy VA run."""
  return MetricConfig(
    columns=ColumnSpec(
      group=group_spec.key,
      response=outcome.response,
      risk=outcome.risk,
      id="patienticn",
    ),
    ref_group=group_spec.reference,
    quantiles=QUANTILES,
    metrics=(MetricName.ATPR,),
    calibration=CalibrationConfig(
      degree=2,
      transform=True,
      cv=True,
    ),
    density_ratio=DensityRatioConfig(
      degree=1,
      transform=False,
      cv=True,
    ),
    bootstrap=BootstrapConfig(enabled=False),
    output=OutputConfig(include_intermediates=True),
    random_state=RANDOM_SEED,
  )


def compute_calibration_summary(
  frame: pd.DataFrame,
  outcome: OutcomeSpec,
  group_spec: GroupSpec,
  n_bins: int = 10,
) -> pd.DataFrame:
  """Summarize observed versus predicted risk in common outcome-risk intervals."""
  working = frame[[group_spec.key, outcome.risk, outcome.response]].copy()
  working["_risk_bin"] = pd.qcut(
    working[outcome.risk],
    q=n_bins,
    duplicates="drop",
  )
  summary = (
    working.groupby([group_spec.key, "_risk_bin"], observed=True)
    .agg(
      mean_predicted=(outcome.risk, "mean"),
      observed_rate=(outcome.response, "mean"),
    )
    .reset_index()
    .rename(columns={group_spec.key: "group"})
  )
  return summary[["group", "mean_predicted", "observed_rate"]]


def compute_log_risk_density(
  values: Sequence[float] | np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
  """Estimate a bounded, deterministic density on the log10-risk scale.

  Non-finite values are removed. Finite probabilities are clipped to
  ``[1e-6, 1]`` only for display, histogrammed on a fixed common grid, smoothed
  with a compact Gaussian kernel, and normalized to unit area.
  """
  array = np.asarray(values, dtype=np.float64)
  finite = array[np.isfinite(array)]
  centers = (LOG_RISK_EDGES[:-1] + LOG_RISK_EDGES[1:]) / 2.0
  if finite.size == 0:
    return centers, np.zeros_like(centers)
  if np.any((finite < 0.0) | (finite > 1.0)):
    raise ValueError("risk values must be probabilities on [0, 1]")
  transformed = np.log10(np.clip(finite, RISK_CLIP, 1.0))
  counts, _ = np.histogram(transformed, bins=LOG_RISK_EDGES)
  offsets = np.arange(-8, 9, dtype=np.float64)
  kernel = np.exp(-0.5 * (offsets / 2.0) ** 2)
  kernel /= kernel.sum()
  density = np.convolve(counts.astype(np.float64), kernel, mode="same")
  area = float(np.trapezoid(density, centers))
  if area > 0.0:
    density /= area
  return centers, density


def compute_density_summary(
  calibrated: pd.DataFrame,
  outcome: OutcomeSpec,
  group_spec: GroupSpec,
) -> pd.DataFrame:
  """Reduce row-level original and recalibrated risks to density curves."""
  rows: list[pd.DataFrame] = []
  for level in group_spec.levels:
    group_ids = calibrated[group_spec.key].map(normalize_group_id)
    group_frame = calibrated[group_ids.eq(level.value)]
    for family, column in (
      ("Original CAN", outcome.risk),
      ("Recalibrated CAN", "cal_risk"),
    ):
      grid, density = compute_log_risk_density(group_frame[column])
      rows.append(
        pd.DataFrame(
          {
            "group": level.value,
            "family": family,
            "log_risk": grid,
            "density": density,
          }
        )
      )
  return pd.concat(rows, ignore_index=True)


def compute_sensitivity_differences(
  metric_frame: pd.DataFrame,
  group_spec: GroupSpec,
) -> pd.DataFrame:
  """Center conventional TPR and aTPR on their reference-group curves."""
  working = metric_frame[
    [group_spec.key, "quantile", "tau", "TPR", "aTPR"]
  ].copy()
  working[group_spec.key] = working[group_spec.key].map(normalize_group_id)
  reference = (
    working[working[group_spec.key].eq(group_spec.reference)][
      ["quantile", "TPR", "aTPR"]
    ]
    .rename(columns={"TPR": "TPR_reference", "aTPR": "aTPR_reference"})
    .copy()
  )
  if reference.empty:
    raise ValueError(
      f"reference group {group_spec.reference!r} is absent from {group_spec.key}"
    )
  result = working.merge(reference, on="quantile", how="left", validate="m:1")
  result["TPR_difference"] = result["TPR"] - result["TPR_reference"]
  result["aTPR_difference"] = result["aTPR"] - result["aTPR_reference"]
  result = result[~result[group_spec.key].eq(group_spec.reference)].copy()
  result = result.rename(columns={group_spec.key: "group"})
  return result[
    [
      "group",
      "quantile",
      "tau",
      "TPR",
      "aTPR",
      "TPR_reference",
      "aTPR_reference",
      "TPR_difference",
      "aTPR_difference",
    ]
  ].sort_values(["group", "quantile"]).reset_index(drop=True)


def analyze_panel(
  cohort: pd.DataFrame,
  outcome: OutcomeSpec,
  group_spec: GroupSpec,
) -> PanelData:
  """Run one package analysis and retain plot-level aggregates only."""
  frame = complete_case_frame(cohort, outcome, group_spec)
  calibration = compute_calibration_summary(frame, outcome, group_spec)
  result = adjusted_metrics(frame, build_metric_config(outcome, group_spec))
  calibrated = result.calibrated
  if calibrated is None:
    raise RuntimeError("metrics-adjuster did not return requested calibrated intermediates")
  densities = compute_density_summary(calibrated, outcome, group_spec)
  sensitivity = compute_sensitivity_differences(
    result.metrics[MetricName.ATPR.value],
    group_spec,
  )
  q95_values = sensitivity.loc[sensitivity["quantile"].eq(0.95), "tau"].unique()
  if len(q95_values) != 1:
    raise RuntimeError(
      f"expected one q95 threshold for {outcome.response}/{group_spec.key}, "
      f"found {q95_values.tolist()}"
    )
  return PanelData(
    group_spec=group_spec,
    calibration=calibration,
    densities=densities,
    sensitivity=sensitivity,
    q95_tau=float(q95_values[0]),
  )


def analyze_outcome(
  cohort: pd.DataFrame,
  outcome: OutcomeSpec,
) -> tuple[PanelData, ...]:
  """Analyze all subgroup domains sequentially to bound retained memory."""
  panels: list[PanelData] = []
  for group_spec in GROUP_SPECS:
    print(f"Analyzing {outcome.title}: {group_spec.title}", flush=True)
    panels.append(analyze_panel(cohort, outcome, group_spec))
    gc.collect()
  return tuple(panels)


def _level_map(group_spec: GroupSpec) -> dict[str, LevelSpec]:
  return {level.value: level for level in group_spec.levels}


def _style_axis(axis: Axes) -> None:
  axis.spines["top"].set_visible(False)
  axis.spines["right"].set_visible(False)
  axis.tick_params(labelsize=7, width=0.8, length=3)
  axis.grid(False)


def _plot_calibration(axis: Axes, panel: PanelData) -> None:
  levels = _level_map(panel.group_spec)
  for group, group_frame in panel.calibration.groupby("group", sort=False):
    level = levels[str(group)]
    ordered = group_frame.sort_values("mean_predicted")
    axis.plot(
      ordered["mean_predicted"],
      ordered["observed_rate"],
      color=level.color,
      linewidth=1.4,
      marker="o",
      markersize=2.8,
      label=level.label,
    )
  axis.legend(
    loc="upper left",
    fontsize=5.5,
    frameon=False,
    handlelength=1.8,
    borderaxespad=0.2,
  )


def _plot_densities(axis: Axes, panel: PanelData) -> None:
  levels = _level_map(panel.group_spec)
  styles = {"Original CAN": ":", "Recalibrated CAN": "-"}
  for (group, family), curve in panel.densities.groupby(
    ["group", "family"],
    sort=False,
  ):
    axis.plot(
      curve["log_risk"],
      curve["density"],
      color=levels[str(group)].color,
      linestyle=styles[str(family)],
      linewidth=1.35,
    )


def _plot_sensitivity(axis: Axes, panel: PanelData) -> None:
  levels = _level_map(panel.group_spec)
  for group, curve in panel.sensitivity.groupby("group", sort=False):
    level = levels[str(group)]
    ordered = curve.sort_values("tau")
    axis.plot(
      ordered["tau"],
      ordered["TPR_difference"],
      color=level.color,
      linewidth=1.4,
      linestyle="-",
    )
    axis.plot(
      ordered["tau"],
      ordered["aTPR_difference"],
      color=level.color,
      linewidth=1.4,
      linestyle="--",
    )
    q95 = ordered[ordered["quantile"].eq(0.95)]
    axis.scatter(
      q95["tau"],
      q95["TPR_difference"],
      color=level.color,
      s=11,
      zorder=3,
    )
    axis.scatter(
      q95["tau"],
      q95["aTPR_difference"],
      color=level.color,
      facecolors="white",
      s=13,
      linewidths=0.8,
      zorder=3,
    )
  axis.axhline(0.0, color="#333333", linewidth=0.8, linestyle=":")
  axis.axvline(panel.q95_tau, color=GRAY, linewidth=0.8, linestyle=":")


def _shared_limits(panels: Sequence[PanelData]) -> tuple[float, float, float]:
  calibration_max = max(
    max(
      float(panel.calibration["mean_predicted"].max()),
      float(panel.calibration["observed_rate"].max()),
    )
    for panel in panels
  )
  density_max = max(float(panel.densities["density"].max()) for panel in panels)
  sensitivity_max = max(
    float(
      panel.sensitivity[["TPR_difference", "aTPR_difference"]]
      .abs()
      .max()
      .max()
    )
    for panel in panels
  )
  return (
    min(1.0, calibration_max * 1.08),
    density_max * 1.05,
    max(0.05, sensitivity_max * 1.10),
  )


def build_outcome_figure(
  outcome: OutcomeSpec,
  panels: Sequence[PanelData],
  analyzed_rows: int,
) -> Figure:
  """Render one five-row by three-column manuscript figure."""
  if len(panels) != len(GROUP_SPECS):
    raise ValueError(f"expected {len(GROUP_SPECS)} panel rows, received {len(panels)}")
  figure, axes = plt.subplots(
    len(panels),
    3,
    figsize=(10.0, 13.0),
    sharex="col",
    sharey="col",
  )
  calibration_max, density_max, sensitivity_max = _shared_limits(panels)
  panel_letter = ord("A")
  for row, panel in enumerate(panels):
    calibration_axis, density_axis, sensitivity_axis = axes[row]
    _plot_calibration(calibration_axis, panel)
    _plot_densities(density_axis, panel)
    _plot_sensitivity(sensitivity_axis, panel)
    for column, axis in enumerate(
      (calibration_axis, density_axis, sensitivity_axis)
    ):
      _style_axis(axis)
      x_position = 0.98 if column == 0 else 0.01
      horizontal_alignment = "right" if column == 0 else "left"
      axis.text(
        x_position,
        0.98,
        f"({chr(panel_letter)})",
        transform=axis.transAxes,
        ha=horizontal_alignment,
        va="top",
        fontsize=8,
        fontweight="bold",
      )
      panel_letter += 1
    calibration_axis.plot(
      [0.0, calibration_max],
      [0.0, calibration_max],
      color="#222222",
      linewidth=0.9,
      linestyle="--",
      zorder=0,
    )
    calibration_axis.set_xlim(0.0, calibration_max)
    calibration_axis.set_ylim(0.0, calibration_max)
    density_axis.set_xlim(float(LOG_RISK_EDGES[0]), 0.0)
    density_axis.set_ylim(0.0, density_max)
    sensitivity_axis.set_ylim(-sensitivity_max, sensitivity_max)
    calibration_axis.annotate(
      panel.group_spec.title,
      xy=(-0.25, 0.5),
      xycoords="axes fraction",
      rotation=90,
      ha="center",
      va="center",
      fontsize=9,
      fontweight="bold",
    )

  axes[0, 0].set_title("Calibration", fontsize=12, pad=10)
  axes[0, 1].set_title("Risk Score Distribution", fontsize=12, pad=10)
  axes[0, 2].set_title("TPR vs aTPR Differences", fontsize=12, pad=10)
  axes[-1, 0].set_xlabel("Mean predicted CAN risk", fontsize=9)
  axes[-1, 1].set_xlabel(r"$\log_{10}$(risk)", fontsize=9)
  axes[-1, 2].set_xlabel(r"Risk threshold, $\tau$", fontsize=9)
  for row in range(len(panels)):
    axes[row, 0].set_ylabel("Observed rate", fontsize=8)
    axes[row, 1].set_ylabel("Density", fontsize=8)
    axes[row, 2].set_ylabel("Difference vs reference", fontsize=8)

  density_legend = (
    Line2D([0], [0], color="#222222", linestyle=":", label="Original CAN"),
    Line2D([0], [0], color="#222222", linestyle="-", label="Recalibrated CAN"),
  )
  sensitivity_legend = (
    Line2D([0], [0], color="#222222", linestyle="-", label="TPR difference"),
    Line2D([0], [0], color="#222222", linestyle="--", label="aTPR difference"),
    Line2D([0], [0], color=GRAY, linestyle=":", label="95th-percentile threshold"),
  )
  axes[0, 1].legend(
    handles=density_legend,
    loc="upper right",
    fontsize=6,
    frameon=False,
  )
  axes[0, 2].legend(
    handles=sensitivity_legend,
    loc="lower left",
    fontsize=6,
    frameon=False,
  )
  figure.suptitle(
    f"VA CAN {outcome.title}: Calibration, Risk Distributions, and TPR Adjustment",
    fontsize=14,
    y=0.995,
  )
  figure.text(
    0.5,
    0.008,
    f"N={analyzed_rows:,} complete cases; blue=reference; "
    "open q95 markers=aTPR.",
    ha="center",
    va="bottom",
    fontsize=7,
  )
  figure.subplots_adjust(
    left=0.10,
    right=0.985,
    bottom=0.055,
    top=0.955,
    hspace=0.22,
    wspace=0.25,
  )
  return figure


def save_figure(
  figure: Figure,
  output_dir: Path,
  outcome: OutcomeSpec,
) -> tuple[Path, Path]:
  """Write the manuscript figure as a 300-DPI PNG and vector PDF."""
  output_dir.mkdir(parents=True, exist_ok=True)
  png_path = output_dir / f"{outcome.output_stem}.png"
  pdf_path = output_dir / f"{outcome.output_stem}.pdf"
  figure.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
  figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
  return png_path, pdf_path


def q95_summary(outcome: OutcomeSpec, panels: Sequence[PanelData]) -> pd.DataFrame:
  """Return the aggregate q95 curve values used for manuscript reconciliation."""
  rows: list[pd.DataFrame] = []
  for panel in panels:
    q95 = panel.sensitivity[panel.sensitivity["quantile"].eq(0.95)].copy()
    q95.insert(0, "subgroup", panel.group_spec.title)
    q95.insert(0, "outcome", outcome.title)
    rows.append(q95)
  return pd.concat(rows, ignore_index=True)


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--env-file",
    type=Path,
    default=DEFAULT_ENV_FILE,
    help="Dotenv file containing the DATA root.",
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=DEFAULT_OUTPUT_DIR,
    help="Directory for manuscript PNG and PDF files.",
  )
  return parser


def main(argv: Sequence[str] | None = None) -> None:
  """Run both full-cohort analyses and write aggregate manuscript figures."""
  args = build_parser().parse_args(argv)
  input_path = resolve_input_path(args.env_file)
  cohort = load_cohort(input_path)
  print(f"metrics-adjuster {importlib.metadata.version('metrics-adjuster')}")
  print(f"Input: {input_path.name}; rows={len(cohort):,}")
  print(f"Quantiles: {QUANTILES}; seed={RANDOM_SEED}")
  summaries: list[pd.DataFrame] = []
  written: list[Path] = []
  for outcome in OUTCOME_SPECS:
    analyzed_rows = int(cohort[[outcome.response, outcome.risk]].notna().all(axis=1).sum())
    panels = analyze_outcome(cohort, outcome)
    figure = build_outcome_figure(outcome, panels, analyzed_rows)
    written.extend(save_figure(figure, args.output_dir, outcome))
    plt.close(figure)
    summaries.append(q95_summary(outcome, panels))
  summary = pd.concat(summaries, ignore_index=True)
  print("\nQ95 differences versus configured reference groups:")
  print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
  print("\nGenerated figures:")
  for path in written:
    print(f"  - {path}")


if __name__ == "__main__":
  main()
