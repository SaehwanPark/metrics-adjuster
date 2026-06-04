"""Report tables and static figures for adjusted metric outputs."""

from __future__ import annotations

import base64
import html
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from metrics_adjuster.core import MetricFrames, compute_logit, traditional_metric_name
from metrics_adjuster.types import MetricConfig, MetricName, ReportConfig


@dataclass(frozen=True)
class ReportBundle:
  """Rendered report plus reusable component objects."""

  metrics: MetricFrames
  metric_table: pd.DataFrame
  density_figure: Figure
  weight_ratio_figure: Figure
  html: str


def bootstrap_interval_summary(
  bootstrap: pd.DataFrame | None,
  group_col: str,
  alpha: float,
) -> pd.DataFrame | None:
  """Summarize bootstrap records for original and adjusted report intervals."""
  if bootstrap is None or bootstrap.empty:
    return None
  required = {"metric", group_col, "quantile", "original_value", "adjusted_value"}
  if not required.issubset(bootstrap.columns):
    return None
  return (
    bootstrap.groupby(["metric", group_col, "quantile"], dropna=False)
    .agg(
      original_ci_lower=("original_value", lambda x: float(np.quantile(x, alpha / 2.0))),
      original_ci_upper=("original_value", lambda x: float(np.quantile(x, 1.0 - alpha / 2.0))),
      adjusted_ci_lower=("adjusted_value", lambda x: float(np.quantile(x, alpha / 2.0))),
      adjusted_ci_upper=("adjusted_value", lambda x: float(np.quantile(x, 1.0 - alpha / 2.0))),
      n_boot=("adjusted_value", "count"),
    )
    .reset_index()
  )


def metric_comparison_table(
  metrics: Mapping[str, pd.DataFrame],
  group_col: str,
  bootstrap: pd.DataFrame | None = None,
  alpha: float = 0.05,
) -> pd.DataFrame:
  """Create a long table of conventional and adjusted metric values."""
  ordered = [
    "metric",
    group_col,
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
  records: list[pd.DataFrame] = []
  for metric_name, frame in metrics.items():
    metric = MetricName(metric_name)
    original_name = traditional_metric_name(metric)
    selected = frame[[group_col, "quantile", "tau", original_name, metric.value]].copy()
    selected.insert(0, "metric", metric.value)
    selected["original_metric"] = original_name
    selected["adjusted_metric"] = metric.value
    selected = selected.rename(
      columns={
        original_name: "original_value",
        metric.value: "adjusted_value",
      }
    )
    records.append(selected)

  if not records:
    return pd.DataFrame(columns=ordered)

  table = pd.concat(records, ignore_index=True)
  interval_summary = bootstrap_interval_summary(bootstrap, group_col, alpha)
  if interval_summary is not None:
    table = table.merge(interval_summary, on=["metric", group_col, "quantile"], how="left")
  else:
    table["original_ci_lower"] = np.nan
    table["original_ci_upper"] = np.nan
    table["adjusted_ci_lower"] = np.nan
    table["adjusted_ci_upper"] = np.nan
    table["n_boot"] = pd.NA

  return table[ordered].sort_values(["metric", "quantile", group_col]).reset_index(drop=True)


def cutoff_reference_lines(
  metric_table: pd.DataFrame,
  max_cutoff_lines: int,
) -> tuple[tuple[float, float], ...]:
  """Select readable cutoff reference lines as ``(quantile, tau)`` pairs."""
  if max_cutoff_lines <= 0 or metric_table.empty:
    return ()
  cutoffs = (
    metric_table[["quantile", "tau"]]
    .drop_duplicates()
    .sort_values(["quantile", "tau"])
    .to_numpy(dtype=float)
  )
  if len(cutoffs) <= max_cutoff_lines:
    return tuple((float(q), float(tau)) for q, tau in cutoffs)
  positions = np.linspace(0, len(cutoffs) - 1, max_cutoff_lines)
  indexes = sorted({int(round(position)) for position in positions})
  return tuple((float(cutoffs[index, 0]), float(cutoffs[index, 1])) for index in indexes)


def density_grid(values: pd.Series, points: int) -> tuple[np.ndarray, np.ndarray]:
  """Estimate a simple bounded Gaussian density on the probability scale."""
  clean = values.dropna().to_numpy(dtype=float)
  grid = np.linspace(0.0, 1.0, points)
  if len(clean) == 0:
    return grid, np.zeros_like(grid)
  spread = float(np.std(clean))
  bandwidth = max(0.02, 1.06 * spread * (len(clean) ** (-1.0 / 5.0)))
  scaled = (grid[:, None] - clean[None, :]) / bandwidth
  density = np.exp(-0.5 * scaled**2).mean(axis=1) / (bandwidth * np.sqrt(2.0 * np.pi))
  area = float(np.trapezoid(density, grid))
  if area > 0:
    density = density / area
  return grid, density


def transformed_values(values: pd.Series, x_scale: str) -> np.ndarray:
  """Return calibrated-risk values on the requested report x-axis scale."""
  clean = values.dropna().to_numpy(dtype=float)
  if x_scale == "log_odds":
    return np.asarray(compute_logit(clean), dtype=np.float64)
  return clean


def transformed_grid(values: pd.Series, points: int, x_scale: str) -> np.ndarray:
  """Create a common density grid for all plotted groups."""
  if x_scale == "probability":
    return np.linspace(0.0, 1.0, points)
  transformed = transformed_values(values, x_scale)
  if len(transformed) == 0:
    return np.linspace(-1.0, 1.0, points)
  lower = float(np.quantile(transformed, 0.005))
  upper = float(np.quantile(transformed, 0.995))
  if lower == upper:
    lower -= 1.0
    upper += 1.0
  padding = max(0.25, 0.04 * (upper - lower))
  return np.linspace(lower - padding, upper + padding, points)


def density_on_grid(values: pd.Series, grid: np.ndarray, x_scale: str) -> np.ndarray:
  """Estimate a Gaussian density on a caller-provided grid."""
  clean = transformed_values(values, x_scale)
  if len(clean) == 0:
    return np.zeros_like(grid)
  spread = float(np.std(clean))
  bandwidth = max(0.02, 1.06 * spread * (len(clean) ** (-1.0 / 5.0)))
  scaled = (grid[:, None] - clean[None, :]) / bandwidth
  density = np.exp(-0.5 * scaled**2).mean(axis=1) / (bandwidth * np.sqrt(2.0 * np.pi))
  area = float(np.trapezoid(density, grid))
  if area > 0:
    density = density / area
  return np.asarray(density, dtype=np.float64)


def transformed_cutoffs(
  cutoffs: Sequence[tuple[float, float]],
  x_scale: str,
) -> tuple[tuple[float, float], ...]:
  """Move cutoff x-positions to the report plot scale."""
  if x_scale == "probability":
    return tuple(cutoffs)
  if not cutoffs:
    return ()
  quantiles = [quantile for quantile, _ in cutoffs]
  thresholds = np.asarray([tau for _, tau in cutoffs], dtype=float)
  transformed = compute_logit(thresholds)
  return tuple(
    (float(quantile), float(tau))
    for quantile, tau in zip(quantiles, transformed, strict=True)
  )


def display_column(report_config: ReportConfig, column: str) -> str:
  """Return a human-readable column label when configured."""
  return report_config.labels.columns.get(column, column)


def display_group(report_config: ReportConfig, group_col: str, group_id: Any) -> str:
  """Return a human-readable group label when configured."""
  group_labels = report_config.labels.groups.get(group_col, {})
  return group_labels.get(str(group_id), str(group_id))


def display_metric(report_config: ReportConfig, metric: str) -> str:
  """Return a human-readable metric label when configured."""
  return report_config.labels.metrics.get(metric, metric)


def group_color_map(groups: Sequence[Any]) -> dict[Any, str]:
  """Assign deterministic colors shared across report figures."""
  palette = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
  )
  return {group: palette[index % len(palette)] for index, group in enumerate(groups)}


def add_cutoff_lines(
  figure: Figure,
  cutoffs: Sequence[tuple[float, float]],
  *,
  label_first: bool = True,
) -> None:
  """Draw vertical cutoff markers on all axes in a figure."""
  for ax in figure.axes:
    for index, (_, tau) in enumerate(cutoffs):
      label = "cutoff" if label_first and index == 0 else None
      ax.axvline(tau, color="#5d6d7e", alpha=0.28, linewidth=1.0, linestyle="--", label=label)


def calibrated_density_figure(
  weighted: pd.DataFrame,
  group_col: str,
  report_config: ReportConfig | None = None,
  cutoffs: Sequence[tuple[float, float]] = (),
  colors: Mapping[Any, str] | None = None,
) -> Figure:
  """Render calibrated probability densities by group."""
  resolved_config = report_config or ReportConfig()
  figure = Figure(figsize=(8.0, 4.8), constrained_layout=True)
  ax = figure.subplots()
  grid = transformed_grid(
    weighted["cal_risk"],
    resolved_config.density_points,
    resolved_config.x_scale,
  )
  for group_id, group_df in weighted.groupby(group_col, sort=True):
    density = density_on_grid(group_df["cal_risk"], grid, resolved_config.x_scale)
    label = display_group(resolved_config, group_col, group_id)
    color = colors.get(group_id) if colors is not None else None
    ax.plot(grid, density, linewidth=2.1, label=label, color=color)
  add_cutoff_lines(figure, transformed_cutoffs(cutoffs, resolved_config.x_scale))
  ax.set_title("Calibrated probability densities")
  ax.set_xlabel(
    "Calibrated probability"
    if resolved_config.x_scale == "probability"
    else "Calibrated log-odds"
  )
  ax.set_ylabel("Density")
  ax.set_xlim(float(grid[0]), float(grid[-1]))
  ax.grid(alpha=0.18)
  ax.legend(loc="best")
  return figure


def weight_ratio_figure(
  weighted: pd.DataFrame,
  group_col: str,
  ref_group: Any,
  report_config: ReportConfig | None = None,
  cutoffs: Sequence[tuple[float, float]] = (),
  colors: Mapping[Any, str] | None = None,
) -> Figure:
  """Render group densities normalized to the reference-group density."""
  resolved_config = report_config or ReportConfig()
  figure = Figure(figsize=(8.0, 4.8), constrained_layout=True)
  ax = figure.subplots()
  ref_df = weighted[weighted[group_col] == ref_group]
  grid = transformed_grid(
    weighted["cal_risk"],
    resolved_config.density_points,
    resolved_config.x_scale,
  )
  ref_density = density_on_grid(ref_df["cal_risk"], grid, resolved_config.x_scale)
  for group_id, group_df in weighted.groupby(group_col, sort=True):
    if group_id == ref_group:
      ratio = np.ones_like(grid)
    else:
      group_density = density_on_grid(group_df["cal_risk"], grid, resolved_config.x_scale)
      ratio = np.divide(
        group_density,
        ref_density,
        out=np.full_like(group_density, np.nan),
        where=ref_density > 1e-9,
      )
    label = display_group(resolved_config, group_col, group_id)
    color = colors.get(group_id) if colors is not None else None
    ax.plot(
      grid,
      ratio,
      linewidth=2.1,
      label=label,
      color=color,
    )
  add_cutoff_lines(figure, transformed_cutoffs(cutoffs, resolved_config.x_scale))
  ax.set_title("Densities normalized to reference group")
  ax.set_xlabel(
    "Calibrated probability"
    if resolved_config.x_scale == "probability"
    else "Calibrated log-odds"
  )
  ax.set_ylabel("Density relative to reference")
  ax.set_xlim(float(grid[0]), float(grid[-1]))
  ax.grid(alpha=0.18)
  ax.legend(loc="best")
  return figure


def figure_to_data_uri(figure: Figure) -> str:
  """Serialize a Matplotlib figure as an embedded SVG data URI."""
  buffer = BytesIO()
  figure.savefig(buffer, format="svg", bbox_inches="tight")
  encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
  return f"data:image/svg+xml;base64,{encoded}"


def report_summary_items(
  result: MetricFrames,
  weighted: pd.DataFrame,
  config: MetricConfig,
  report_config: ReportConfig | None = None,
) -> tuple[tuple[str, str], ...]:
  """Build compact metadata shown near the top of the HTML report."""
  resolved_config = report_config or ReportConfig()
  metrics = (
    ", ".join(display_metric(resolved_config, metric.value) for metric in config.metrics)
    if config.metrics
    else "none"
  )
  bootstrap = (
    f"enabled ({config.bootstrap.iterations} iterations, alpha={config.bootstrap.alpha:g})"
    if config.bootstrap.enabled
    else "disabled"
  )
  quantiles = ", ".join(f"{quantile:g}" for quantile in config.quantiles)
  return (
    ("Rows analyzed", f"{len(weighted):,}"),
    ("Group column", display_column(resolved_config, config.columns.group)),
    ("Reference group", display_group(resolved_config, config.columns.group, config.ref_group)),
    ("Response column", display_column(resolved_config, config.columns.response)),
    ("Risk column", display_column(resolved_config, config.columns.risk)),
    ("Quantiles", quantiles),
    ("Metrics", metrics),
    ("Bootstrap", bootstrap),
    ("Observed groups", str(weighted[config.columns.group].nunique(dropna=True))),
    ("Bootstrap records", str(len(result.bootstrap)) if result.bootstrap is not None else "0"),
  )


def format_point(value: Any) -> str:
  """Format a numeric table value for report display."""
  return "unavailable" if pd.isna(value) else f"{float(value):.4f}"


def format_adjusted_estimate(row: pd.Series) -> str:
  """Format adjusted point estimate with optional interval."""
  estimate = format_point(row["adjusted_value"])
  lower = row.get("adjusted_ci_lower")
  upper = row.get("adjusted_ci_upper")
  if pd.isna(lower) or pd.isna(upper):
    return f"{estimate} (unavailable)"
  return f"{estimate} ({float(lower):.4f}-{float(upper):.4f})"


def compact_metric_table_html(
  metric_table: pd.DataFrame,
  group_col: str,
  report_config: ReportConfig,
) -> str:
  """Render compact exhibit-style subtables for each metric."""
  if metric_table.empty:
    return '<p class="empty-note">No metrics were selected for this report.</p>'

  sections: list[str] = []
  metric_order = {metric.value: index for index, metric in enumerate(MetricName)}
  ordered = metric_table.assign(
    _metric_order=metric_table["metric"].map(lambda value: metric_order.get(str(value), 999))
  ).sort_values(["_metric_order", "quantile", group_col])
  for metric, frame in ordered.groupby("metric", sort=False):
    rows = []
    for _, row in frame.sort_values(["quantile", group_col]).iterrows():
      rows.append(
        {
          "Group": display_group(report_config, group_col, row[group_col]),
          "Threshold": f"{float(row['tau']):.4f} (q={float(row['quantile']):.3g})",
          "Original": format_point(row["original_value"]),
          "Adjusted": format_adjusted_estimate(row),
        }
      )
    table_html = pd.DataFrame(rows).to_html(
      index=False,
      escape=True,
      border=0,
      classes="metric-table compact-table",
    )
    sections.append(
      f'<div class="metric-subtable">'
      f"<h3>{html.escape(display_metric(report_config, str(metric)))}</h3>"
      f'<div class="table-wrap">{table_html}</div>'
      f"</div>"
    )
  return "".join(sections)


def render_report_html(
  metric_table: pd.DataFrame,
  density_figure: Figure,
  weight_ratio_figure_: Figure,
  report_config: ReportConfig | None = None,
  summary_items: Sequence[tuple[str, str]] = (),
) -> str:
  """Render a self-contained HTML report."""
  resolved_config = report_config or ReportConfig()
  group_col = next(
    (column for column in metric_table.columns if column not in {
      "metric",
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
    }),
    "group",
  )
  table_html = compact_metric_table_html(metric_table, group_col, resolved_config)
  density_uri = figure_to_data_uri(density_figure)
  ratio_uri = figure_to_data_uri(weight_ratio_figure_)
  title = html.escape(resolved_config.title)
  subtitle = (
    f'<p class="subtitle">{html.escape(resolved_config.subtitle)}</p>'
    if resolved_config.subtitle is not None
    else ""
  )
  summary_html = "".join(
    (
      f"<div class=\"summary-card\"><span>{html.escape(label)}</span>"
      f"<strong>{html.escape(value)}</strong></div>"
    )
    for label, value in summary_items
  )
  intro = (
    "Original and adjusted group-aware metrics are shown by cutoff quantile. "
    "Confidence interval fields use bootstrap summaries when available; otherwise "
    "they are marked unavailable."
  )
  ratio_note = (
    "Curves show each group density divided by the reference-group density on the "
    "configured calibrated-risk scale."
  )
  return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    :root {{
      --ink: #18212b;
      --muted: #607080;
      --line: #d7dee8;
      --panel: #ffffff;
      --wash: #eef4f8;
      --accent: #0f6b78;
    }}
    body {{
      margin: 0;
      color: var(--ink);
      background: linear-gradient(135deg, #f8fbfc 0%, #edf4f0 48%, #e8eff6 100%);
      font-family: "Avenir Next", "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 42px 24px 56px;
    }}
    header {{
      border-left: 8px solid var(--accent);
      padding: 8px 0 8px 22px;
      margin-bottom: 28px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(2rem, 5vw, 4.2rem);
      line-height: 0.98;
      letter-spacing: -0.045em;
    }}
    p {{
      color: var(--muted);
      max-width: 820px;
      line-height: 1.55;
    }}
    .subtitle {{
      margin: 0;
      font-size: 1rem;
      font-weight: 600;
      letter-spacing: 0.01em;
      text-transform: uppercase;
    }}
    section {{
      background: color-mix(in srgb, var(--panel) 92%, transparent);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: 0 22px 55px rgb(32 47 67 / 10%);
      margin: 22px 0;
      overflow: hidden;
      padding: 24px;
    }}
    h2 {{
      margin: 0 0 14px;
      font-size: 1.35rem;
      letter-spacing: 0;
    }}
    h3 {{
      margin: 22px 0 10px;
      font-size: 1.02rem;
      letter-spacing: 0;
    }}
    .summary-grid {{
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    }}
    .summary-card {{
      border: 1px solid var(--line);
      border-radius: 18px;
      background: white;
      padding: 14px 16px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .summary-card span {{
      color: var(--muted);
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .summary-card strong {{
      font-size: 1.02rem;
    }}
    .figure {{
      width: 100%;
      border-radius: 16px;
      background: white;
      border: 1px solid var(--line);
    }}
    .table-wrap {{
      overflow-x: auto;
    }}
    .metric-subtable:first-child h3 {{
      margin-top: 0;
    }}
    .empty-note {{
      color: var(--muted);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
    }}
    th {{
      background: var(--wash);
      color: #22303c;
      position: sticky;
      top: 0;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 12px;
      text-align: right;
      white-space: nowrap;
    }}
    th:first-child, td:first-child {{
      text-align: left;
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>{title}</h1>
      {subtitle}
      <p>{html.escape(intro)}</p>
    </header>
    <section>
      <h2>Run Summary</h2>
      <div class="summary-grid">{summary_html}</div>
    </section>
    <section>
      <h2>Table 1. Metric Estimates</h2>
      {table_html}
    </section>
    <section>
      <h2>Figure 1. Calibrated Probability Densities</h2>
      <img class="figure" alt="Calibrated probability densities by group" src="{density_uri}">
    </section>
    <section>
      <h2>Figure 2. Densities Normalized to Reference Group</h2>
      <p>{html.escape(ratio_note)}</p>
      <img class="figure" alt="Density-ratio weight curves" src="{ratio_uri}">
    </section>
  </main>
</body>
</html>
"""


def build_report_bundle(
  result: MetricFrames,
  weighted: pd.DataFrame,
  config: MetricConfig,
  report_config: ReportConfig | None = None,
) -> ReportBundle:
  """Build all report components from pipeline output."""
  resolved_config = report_config or ReportConfig()
  metric_table = metric_comparison_table(
    result.metrics,
    config.columns.group,
    result.bootstrap,
    config.bootstrap.alpha,
  )
  cutoffs = cutoff_reference_lines(metric_table, resolved_config.max_cutoff_lines)
  groups = tuple(group for group, _ in weighted.groupby(config.columns.group, sort=True))
  colors = group_color_map(groups)
  density_figure = calibrated_density_figure(
    weighted,
    config.columns.group,
    resolved_config,
    cutoffs,
    colors,
  )
  ratio_figure = weight_ratio_figure(
    weighted,
    config.columns.group,
    config.ref_group,
    resolved_config,
    cutoffs,
    colors,
  )
  summary_items = report_summary_items(result, weighted, config, resolved_config)
  return ReportBundle(
    metrics=result,
    metric_table=metric_table,
    density_figure=density_figure,
    weight_ratio_figure=ratio_figure,
    html=render_report_html(
      metric_table,
      density_figure,
      ratio_figure,
      resolved_config,
      summary_items,
    ),
  )
