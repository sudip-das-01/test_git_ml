"""Tests for multi-format inference folders."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[2]
EXTENSIONS = ["pkl", "joblib", "dill", "onnx", "pmml"]
SHARED_INPUT = ROOT / "pkl" / "dataset" / "input_with_random_column.csv"
SPLIT_FILE = ROOT / "pkl" / "dataset" / "train_test_split.json"
FEATURE_COLS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]


@pytest.fixture(scope="session", autouse=True)
def ensure_models_and_input():
    subprocess.run(
        [sys.executable, str(ROOT / "pkl" / "scripts" / "run_benchmark.py")],
        cwd=ROOT,
        check=True,
    )


@pytest.fixture
def iris_labels():
    return load_iris().target


@pytest.fixture
def train_test_indices():
    with open(SPLIT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return np.array(data["train_indices"]), np.array(data["test_indices"]), data


class TestSharedInput:
    def test_input_with_random_column_exists(self):
        assert SHARED_INPUT.exists()

    @pytest.mark.parametrize("ext", EXTENSIONS)
    def test_each_extension_uses_shared_input_as_input_csv(self, ext):
        ext_input = ROOT / ext / "dataset" / "input.csv"
        assert ext_input.exists()
        assert ext_input.read_bytes() == SHARED_INPUT.read_bytes()

    @pytest.mark.parametrize("ext", EXTENSIONS)
    def test_input_csv_contains_random_noise_column(self, ext):
        df = pd.read_csv(ROOT / ext / "dataset" / "input.csv")
        assert "random_noise" in df.columns
        assert list(df.columns[:4]) == FEATURE_COLS
        assert "target" not in df.columns
        assert len(df) == 150


class TestInferenceExecution:
    @pytest.mark.parametrize("ext", EXTENSIONS)
    def test_inference_script_succeeds(self, ext):
        proc = subprocess.run(
            [sys.executable, "inference.py"],
            cwd=ROOT / ext,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr

    @pytest.mark.parametrize("ext", EXTENSIONS)
    def test_output_row_count_matches_input(self, ext):
        input_rows = len(pd.read_csv(ROOT / ext / "dataset" / "input.csv"))
        output_rows = len(pd.read_csv(ROOT / ext / "output" / "output.csv"))
        assert output_rows == input_rows

    @pytest.mark.parametrize("ext", EXTENSIONS)
    def test_output_has_target_column(self, ext):
        out = pd.read_csv(ROOT / ext / "output" / "output.csv")
        assert "target" in out.columns
        assert set(np.unique(out["target"])).issubset({0, 1, 2})


class TestInferenceAccuracy:
    @pytest.mark.parametrize("ext", EXTENSIONS)
    def test_inference_test_accuracy_matches_sklearn_test_accuracy(
        self, ext, iris_labels, train_test_indices
    ):
        _, test_idx, split = train_test_indices
        preds = pd.read_csv(ROOT / ext / "output" / "output.csv")["target"].to_numpy()
        inference_test_acc = accuracy_score(iris_labels[test_idx], preds[test_idx])
        assert inference_test_acc == pytest.approx(split["test_accuracy"], abs=1e-6)

    @pytest.mark.parametrize("ext", EXTENSIONS)
    def test_all_extensions_share_same_predictions(self, ext):
        baseline = pd.read_csv(ROOT / "pkl" / "output" / "output.csv")["target"].to_numpy()
        preds = pd.read_csv(ROOT / ext / "output" / "output.csv")["target"].to_numpy()
        np.testing.assert_array_equal(baseline, preds)


class TestSchemaValidation:
    def test_forbidden_target_column_raises(self):
        bad_input = ROOT / "pkl" / "dataset" / "input_bad.csv"
        df = pd.read_csv(ROOT / "pkl" / "dataset" / "input.csv")
        df["target"] = 0
        df.to_csv(bad_input, index=False)

        proc = subprocess.run(
            [
                sys.executable,
                "inference.py",
                "--data",
                str(bad_input),
                "--output",
                str(ROOT / "pkl" / "output" / "output_bad.csv"),
            ],
            cwd=ROOT / "pkl",
            capture_output=True,
            text=True,
        )
        assert proc.returncode != 0
        assert "target" in proc.stderr.lower() or "forbidden" in proc.stderr.lower()


class TestAccuracyReport:
    def test_accuracy_report_exists(self):
        report = ROOT / "pkl" / "output" / "accuracy_report.csv"
        assert report.exists()
        df = pd.read_csv(report)
        assert set(df["extension"]) == set(EXTENSIONS)
        assert "train_accuracy" in df.columns
        assert "test_accuracy" in df.columns
        assert "inference_test_accuracy" in df.columns

    def test_train_accuracy_at_least_test_accuracy(self):
        df = pd.read_csv(ROOT / "pkl" / "output" / "accuracy_report.csv")
        assert (df["train_accuracy"] >= df["test_accuracy"]).all()
