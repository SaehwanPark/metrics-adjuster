"""Synthetic data generators for tests, examples, and demos."""

from __future__ import annotations

import numpy as np
import pandas as pd


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
