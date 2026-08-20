from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


def load_figure_script() -> ModuleType:
  script_path = (
    Path(__file__).parents[2] / "scripts" / "generate_jinbo_manuscript_figures.py"
  )
  spec = importlib.util.spec_from_file_location(
    "generate_jinbo_manuscript_figures",
    script_path,
  )
  assert spec is not None
  assert spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


@pytest.fixture
def figure_script() -> ModuleType:
  return load_figure_script()


def test_resolve_input_path_uses_data_from_dotenv(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  figure_script: ModuleType,
) -> None:
  data_root = tmp_path / "portable-data"
  input_path = data_root / "va_can" / "atpr_input_20250620.parquet"
  input_path.parent.mkdir(parents=True)
  input_path.touch()
  env_path = tmp_path / ".env"
  env_path.write_text(f"DATA={data_root}\n", encoding="utf-8")
  monkeypatch.delenv("DATA", raising=False)

  assert figure_script.resolve_input_path(env_path) == input_path


def test_resolve_input_path_rejects_missing_data_key(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  figure_script: ModuleType,
) -> None:
  env_path = tmp_path / ".env"
  env_path.write_text("UNRELATED=value\n", encoding="utf-8")
  monkeypatch.delenv("DATA", raising=False)

  with pytest.raises(ValueError, match="DATA is not set"):
    figure_script.resolve_input_path(env_path)


def test_resolve_input_path_rejects_missing_parquet(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  figure_script: ModuleType,
) -> None:
  env_path = tmp_path / ".env"
  env_path.write_text(f"DATA={tmp_path}\n", encoding="utf-8")
  monkeypatch.delenv("DATA", raising=False)

  with pytest.raises(FileNotFoundError, match="atpr_input_20250620.parquet"):
    figure_script.resolve_input_path(env_path)


def test_load_cohort_reads_only_required_columns_and_rejects_missing(
  tmp_path: Path,
  figure_script: ModuleType,
) -> None:
  input_path = tmp_path / "input.parquet"
  pd.DataFrame({"patienticn": [1], "Hosp_1y": [0]}).to_parquet(input_path)

  with pytest.raises(ValueError, match="missing required columns"):
    figure_script.load_cohort(input_path)


def test_legacy_run_contract_is_preserved(figure_script: ModuleType) -> None:
  assert figure_script.QUANTILES == tuple(
    round(value, 2) for value in np.arange(0.10, 1.00, 0.05)
  )
  assert {spec.key: spec.reference for spec in figure_script.GROUP_SPECS} == {
    "BLACK": "0",
    "HCC_dementia": "0",
    "Prior1245": "99",
    "urban": "99",
    "sex": "99",
  }
  dementia = next(spec for spec in figure_script.GROUP_SPECS if spec.key == "HCC_dementia")
  assert {level.value: level.label for level in dementia.levels} == {
    "0": "No dementia (reference)",
    "99": "Dementia",
  }

  config = figure_script.build_metric_config(
    figure_script.OUTCOME_SPECS[0],
    figure_script.GROUP_SPECS[0],
  )
  assert config.quantiles == figure_script.QUANTILES
  assert config.calibration.degree == 2
  assert config.calibration.transform is True
  assert config.calibration.cv is True
  assert config.density_ratio.degree == 1
  assert config.density_ratio.transform is False
  assert config.density_ratio.cv is True
  assert config.output.include_intermediates is True
  assert config.random_state == 343


def test_complete_case_frame_is_outcome_specific(figure_script: ModuleType) -> None:
  frame = pd.DataFrame(
    {
      "patienticn": [1, 2, 3, 4],
      "Hosp_1y": [0, 1, 0, 1],
      "pHosp_1y": [0.1, np.nan, 0.3, 0.4],
      "BLACK": [0, 1, np.nan, 1],
    }
  )

  result = figure_script.complete_case_frame(
    frame,
    figure_script.OUTCOME_SPECS[0],
    figure_script.GROUP_SPECS[0],
  )

  assert len(result) == 2
  assert result["BLACK"].tolist() == ["0", "1"]


def test_log_risk_density_is_finite_nonnegative_and_normalized(
  figure_script: ModuleType,
) -> None:
  values = np.array([0.0, 0.001, 0.01, 0.1, 0.5, np.nan])

  grid, density = figure_script.compute_log_risk_density(values)

  assert grid.shape == density.shape
  assert np.all(np.isfinite(grid))
  assert np.all(np.isfinite(density))
  assert np.all(density >= 0.0)
  assert np.trapezoid(density, grid) == pytest.approx(1.0)


def test_sensitivity_differences_center_both_metrics_on_reference(
  figure_script: ModuleType,
) -> None:
  metric_frame = pd.DataFrame(
    {
      "BLACK": ["0", "1", "0", "1"],
      "quantile": [0.90, 0.90, 0.95, 0.95],
      "tau": [0.2, 0.2, 0.3, 0.3],
      "TPR": [0.4, 0.5, 0.3, 0.6],
      "aTPR": [0.45, 0.35, 0.25, 0.55],
    }
  )

  result = figure_script.compute_sensitivity_differences(
    metric_frame,
    figure_script.GROUP_SPECS[0],
  )

  assert result["group"].tolist() == ["1", "1"]
  assert result["TPR_difference"].tolist() == pytest.approx([0.1, 0.3])
  assert result["aTPR_difference"].tolist() == pytest.approx([-0.1, 0.3])
  assert result.loc[result["quantile"].eq(0.95), "tau"].item() == pytest.approx(0.3)


def _panel_data(figure_script: ModuleType, group_spec: object) -> object:
  levels = [level.value for level in group_spec.levels]
  calibration = pd.DataFrame(
    {
      "group": np.repeat(levels, 2),
      "mean_predicted": np.tile([0.05, 0.20], len(levels)),
      "observed_rate": np.tile([0.04, 0.22], len(levels)),
    }
  )
  density_rows: list[dict[str, object]] = []
  for level in levels:
    for family, offset in (("Original CAN", 0.0), ("Recalibrated CAN", 0.05)):
      for log_risk, density in zip(
        [-3.0, -2.0, -1.0],
        [0.1 + offset, 0.8, 0.1 - offset],
        strict=True,
      ):
        density_rows.append(
          {
            "group": level,
            "family": family,
            "log_risk": log_risk,
            "density": density,
          }
        )
  comparisons = [level for level in levels if level != group_spec.reference]
  sensitivity = pd.DataFrame(
    {
      "group": np.repeat(comparisons, 2),
      "quantile": np.tile([0.90, 0.95], len(comparisons)),
      "tau": np.tile([0.2, 0.3], len(comparisons)),
      "TPR_difference": np.tile([0.1, 0.05], len(comparisons)),
      "aTPR_difference": np.tile([-0.05, 0.02], len(comparisons)),
    }
  )
  return figure_script.PanelData(
    group_spec=group_spec,
    calibration=calibration,
    densities=pd.DataFrame(density_rows),
    sensitivity=sensitivity,
    q95_tau=0.3,
  )


def test_build_and_save_outcome_figure_writes_fifteen_panel_png_and_pdf(
  tmp_path: Path,
  figure_script: ModuleType,
) -> None:
  panels = tuple(_panel_data(figure_script, spec) for spec in figure_script.GROUP_SPECS)

  figure = figure_script.build_outcome_figure(
    figure_script.OUTCOME_SPECS[0],
    panels,
    analyzed_rows=128,
  )
  paths = figure_script.save_figure(
    figure,
    tmp_path,
    figure_script.OUTCOME_SPECS[0],
  )

  assert len(figure.axes) == 15
  assert {path.suffix for path in paths} == {".png", ".pdf"}
  assert all(path.exists() and path.stat().st_size > 0 for path in paths)
  sensitivity_axes = figure.axes[2::3]
  assert all(
    any(np.allclose(line.get_xdata(), [0.3, 0.3]) for line in axis.lines)
    for axis in sensitivity_axes
  )
  plt.close(figure)
