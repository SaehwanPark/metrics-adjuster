"""Unit tests for Section 5 simulation data generation and ground truth benchmarks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure collaboration directory is accessible for test imports
COLLAB_DIR = Path(__file__).resolve().parents[2] / "collaboration" / "saehwan-manuscript-adca"
if str(COLLAB_DIR) not in sys.path:
  sys.path.insert(0, str(COLLAB_DIR))

from simulation.dgp import (  # noqa: E402
  generate_scenario_data,
  sample_truncated_beta,
)
from simulation.ground_truth import compute_population_benchmarks  # noqa: E402
from simulation.scenarios import (  # noqa: E402
  get_scenario_a,
  get_scenario_b,
  get_scenario_e,
  get_scenario_f,
)


def test_sample_truncated_beta_bounds() -> None:
  rng = np.random.default_rng(123)
  samples = sample_truncated_beta(
    rng, size=1000, shape1=2.0, shape2=5.0, lower=0.05, upper=0.95
  )
  assert len(samples) == 1000
  assert np.all(samples >= 0.05)
  assert np.all(samples <= 0.95)


def test_sample_truncated_beta_invalid_args() -> None:
  rng = np.random.default_rng(123)
  with pytest.raises(ValueError, match="Beta shape parameters"):
    sample_truncated_beta(rng, size=10, shape1=-1.0, shape2=2.0)
  with pytest.raises(ValueError, match="lower and upper must satisfy"):
    sample_truncated_beta(rng, size=10, shape1=2.0, shape2=2.0, lower=0.8, upper=0.2)


def test_generate_scenario_data_schema_and_reproducibility() -> None:
  scenario = get_scenario_a()
  df1 = generate_scenario_data(scenario, n=500, seed=42)
  df2 = generate_scenario_data(scenario, n=500, seed=42)

  assert len(df1) == 500
  assert list(df1.columns) == ["patient_id", "group", "true_risk", "risk", "outcome"]
  assert set(df1["group"].unique()) == {"0", "1"}
  assert set(df1["outcome"].unique()).issubset({0, 1})
  assert np.all((df1["true_risk"] > 0.0) & (df1["true_risk"] < 1.0))
  assert np.all((df1["risk"] > 0.0) & (df1["risk"] < 1.0))
  pd_testing = pytest.importorskip("pandas.testing")
  pd_testing.assert_frame_equal(df1, df2)


def test_scenario_a_ground_truth_invariants() -> None:
  """In Scenario A, both groups have identical DGP -> zero true disparity."""
  scenario = get_scenario_a()
  benchmarks = compute_population_benchmarks(
    scenario, thresholds=[0.10, 0.20, 0.30], n_mc=100_000, seed=42
  )

  assert np.allclose(benchmarks["NB_0"], benchmarks["NB_1"], atol=0.005)
  assert np.allclose(benchmarks["aNB_0"], benchmarks["aNB_1"], atol=0.005)
  assert np.allclose(benchmarks["delta_NB"], 0.0, atol=0.005)
  assert np.allclose(benchmarks["delta_aNB"], 0.0, atol=0.005)


def test_scenario_b_ground_truth_apparent_disparity() -> None:
  """In Scenario B, model mappings are identical but Group 1 has higher risk."""
  scenario = get_scenario_b()
  benchmarks = compute_population_benchmarks(
    scenario, thresholds=[0.20, 0.30], n_mc=100_000, seed=42
  )

  # Observed delta NB is substantially positive (Group 1 higher prevalence)
  assert np.all(benchmarks["delta_NB"] > 0.05)
  # Standardized delta aNB is zero within Monte Carlo tolerance
  assert np.allclose(benchmarks["delta_aNB"], 0.0, atol=0.005)


def test_scenario_e_ground_truth_masking() -> None:
  """In Scenario E, observed DCA masks genuine disparity."""
  scenario = get_scenario_e()
  benchmarks = compute_population_benchmarks(scenario, thresholds=[0.25], n_mc=100_000, seed=42)

  row = benchmarks.iloc[0]
  # Observed delta NB is close to zero (case mix masks the bad model)
  assert abs(row["delta_NB"]) < 0.02
  # Standardized delta aNB reveals Group 1 is substantially inferior (delta < 0)
  assert row["delta_aNB"] < -0.03


def test_scenario_f_ground_truth_sign_reversal() -> None:
  """In Scenario F, observed DCA has opposite sign to standardized DCA."""
  scenario = get_scenario_f()
  benchmarks = compute_population_benchmarks(
    scenario, thresholds=[0.25, 0.30], n_mc=100_000, seed=42
  )

  # Observed DCA favors Group 1 due to high prevalence
  assert np.all(benchmarks["delta_NB"] > 0.0)
  # Standardized DCA favors Group 0 because Group 1 model is degraded
  assert np.all(benchmarks["delta_aNB"] < -0.02)
  # Explicit sign reversal
  for _, r in benchmarks.iterrows():
    assert np.sign(r["delta_NB"]) != np.sign(r["delta_aNB"])
