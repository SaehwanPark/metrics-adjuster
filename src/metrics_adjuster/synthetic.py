"""Synthetic data generators for tests, examples, and demos."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _logit(probabilities: FloatArray) -> FloatArray:
  clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
  return np.asarray(np.log(clipped / (1.0 - clipped)), dtype=np.float64)


def _expit(values: FloatArray) -> FloatArray:
  clipped = np.clip(values, -35.0, 35.0)
  return np.asarray(1.0 / (1.0 + np.exp(-clipped)), dtype=np.float64)


def _truncated_beta_sample(
  rng: np.random.Generator,
  size: int,
  shape1: float,
  shape2: float,
  lower: float,
  upper: float,
) -> FloatArray:
  values: list[float] = []
  while len(values) < size:
    batch_size = max(size - len(values), size)
    batch = rng.beta(shape1, shape2, batch_size)
    accepted = batch[(batch >= lower) & (batch <= upper)]
    values.extend(float(value) for value in accepted[: size - len(values)])
  return np.asarray(values, dtype=np.float64)


def generate_synthetic_metrics_data(
  n: int = 600,
  seed: int = 2026,
) -> pd.DataFrame:
  """Create reproducible data with calibrated-ish risk and group differences."""
  rng = np.random.default_rng(seed)
  group = rng.choice(["ref", "minority"], size=n, p=[0.58, 0.42])
  x = rng.normal(size=n)
  group_shift = np.where(group == "minority", 0.45, 0.0)
  latent = -1.1 + 1.25 * x + group_shift
  true_risk = 1.0 / (1.0 + np.exp(-latent))
  observed_risk = np.clip(0.06 + 0.86 * true_risk + rng.normal(0.0, 0.035, n), 0.01, 0.99)
  outcome = rng.binomial(1, true_risk)
  return pd.DataFrame(
    {
      "patient_id": np.arange(1, n + 1),
      "group": group,
      "outcome": outcome,
      "risk": observed_risk,
      "risk_source": np.where(x >= np.median(x), "upper", "lower"),
    }
  )


def generate_xiaoyi_simulation_data(
  n: int = 1000,
  seed: int = 123456,
  prop: float = 0.5,
  alpha1: tuple[float, float] = (2.0, 4.0),
  alpha2: tuple[float, float] = (8.0, 8.0),
  a: tuple[float, float] = (0.0, -0.2),
  b: tuple[float, float] = (0.6, 0.6),
  lower: float = 0.025,
  upper: float = 0.975,
) -> pd.DataFrame:
  """Create data matching the active logit scenario in ``legacy/xiaoyi/simulation.Rmd``."""
  if n <= 0:
    raise ValueError("n must be greater than 0")
  if prop < 0.0 or prop > 1.0:
    raise ValueError("prop must be between 0 and 1")
  if lower < 0.0 or upper > 1.0 or lower >= upper:
    raise ValueError("lower and upper must satisfy 0 <= lower < upper <= 1")
  rng = np.random.default_rng(seed)
  group_numeric = rng.binomial(n=1, p=prop, size=n)
  true_risk = np.zeros(n, dtype=np.float64)

  ref_mask = group_numeric == 0
  comp_mask = group_numeric == 1
  true_risk[ref_mask] = _truncated_beta_sample(
    rng,
    int(ref_mask.sum()),
    alpha1[0],
    alpha2[0],
    lower,
    upper,
  )
  true_risk[comp_mask] = _truncated_beta_sample(
    rng,
    int(comp_mask.sum()),
    alpha1[1],
    alpha2[1],
    lower,
    upper,
  )

  risk = np.zeros(n, dtype=np.float64)
  risk[ref_mask] = _expit(a[0] + b[0] * _logit(true_risk[ref_mask]))
  risk[comp_mask] = _expit(a[1] + b[1] * _logit(true_risk[comp_mask]))
  outcome = rng.binomial(n=1, p=true_risk)

  return pd.DataFrame(
    {
      "patient_id": np.arange(1, n + 1),
      "group": group_numeric.astype(str),
      "outcome": outcome,
      "risk": risk,
      "true_risk": true_risk,
    }
  )
