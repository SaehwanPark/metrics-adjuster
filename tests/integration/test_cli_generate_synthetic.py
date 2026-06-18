from __future__ import annotations

import sys

import pandas as pd
import pytest

from metrics_adjuster.cli import main


def test_cli_generate_synthetic_writes_csv_and_parquet(tmp_path, monkeypatch) -> None:
  output_dir = tmp_path / "data" / "generated" / "synthetic-demo"
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "metrics-adjuster",
      "generate-synthetic",
      "--output-dir",
      str(output_dir),
      "--n",
      "80",
      "--seed",
      "11",
    ],
  )
  main()

  csv_path = output_dir / "synthetic_metrics_data.csv"
  parquet_path = output_dir / "sample.parquet"
  assert csv_path.exists()
  assert parquet_path.exists()
  csv_frame = pd.read_csv(csv_path)
  parquet_frame = pd.read_parquet(parquet_path)
  assert set(csv_frame.columns) == {
    "patient_id",
    "group",
    "outcome",
    "risk",
    "risk_source",
  }
  assert len(csv_frame) == 80
  assert parquet_frame.shape == csv_frame.shape


def test_cli_generate_synthetic_rejects_outside_data_generated(
  tmp_path,
  monkeypatch,
) -> None:
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "metrics-adjuster",
      "generate-synthetic",
      "--output-dir",
      str(tmp_path / "outside"),
      "--n",
      "80",
      "--seed",
      "11",
    ],
  )
  with pytest.raises(SystemExit, match="data/generated"):
    main()


def test_cli_generate_synthetic_rejects_path_traversal(tmp_path, monkeypatch) -> None:
  traversal_dir = tmp_path / "data" / "generated" / ".." / ".." / "escape"
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "metrics-adjuster",
      "generate-synthetic",
      "--output-dir",
      str(traversal_dir),
      "--n",
      "80",
      "--seed",
      "11",
    ],
  )
  with pytest.raises(SystemExit, match="data/generated"):
    main()
