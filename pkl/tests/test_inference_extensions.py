"""Tests for multi-format inference folders."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[2]
EXTENSIONS = [
    "pkl",
    "joblib",
    "dill",
    "onnx",
    "pmml",
    "pickle",
    "xgboost",
    "lightgbm",
    "catboost",
    "h5",
    "pt",
    "pb",
    "zip",
]
SKLEARN_PARITY_EXTENSIONS = ["pkl", "joblib", "dill", "pickle", "onnx", "pmml", "zip"]
SHARED_INPUT = ROOT / "pkl" / "dataset" / "input_with_empty_column.csv"
SPLIT_FILE = ROOT / "pkl" / "dataset" / "train_test_split.json"
SCHEMA_PATH = ROOT / "pkl" / "schema.json"
SERIAL_COLUMN = "id"
EMPTY_COLUMN = "class"


def _feature_columns() -> list[str]:
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["input_parameters_name"]


FEATURE_COLS = _feature_columns()


@pytest.fixture(scope="session", autouse=True)
def ensure_models_and_input():
    subprocess.run(
        [sys.executable, str(ROOT / "pkl" / "scripts" / "run_benchmark.py")],
        cwd=ROOT,
        check=True,
    )


@pytest.fixture
def credit_labels():
    sys.path.insert(0, str(ROOT / "pkl" / "scripts"))
    from training_common import get_training_data

    _, y, _, _, _ = get_training_data()
    return y


@pytest.fixture
def split_data():
    with open(SPLIT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def train_test_indices(split_data):
    return (
        np.array(split_data["train_indices"]),
        np.array(split_data["test_indices"]),
        split_data,
    )


class TestSharedInput:
    def test_canonical_empty_column_dataset_exists(self):
        assert SHARED_INPUT.exists()

    @pytest.mark.parametrize("ext", EXTENSIONS)
    def test_each_extension_uses_shared_input_as_input_csv(self, ext):
        ext_input = ROOT / ext / "dataset" / "input.csv"
        assert ext_input.exists()
        assert ext_input.read_bytes() == SHARED_INPUT.read_bytes()

    @pytest.mark.parametrize("ext", EXTENSIONS)
    def test_input_has_serial_and_empty_class_columns(self, ext):
        df = pd.read_csv(ROOT / ext / "dataset" / "input.csv")
        assert SERIAL_COLUMN in df.columns
        assert df[SERIAL_COLUMN].tolist() == list(range(1, len(df) + 1))
        assert EMPTY_COLUMN in df.columns
        assert df[EMPTY_COLUMN].isna().all() or (df[EMPTY_COLUMN].astype(str).str.strip() == "").all()
        for col in FEATURE_COLS:
            assert col in df.columns
        assert "target" not in df.columns
        assert len(df) == 50


class TestCreateModel:
    @pytest.mark.parametrize("ext", EXTENSIONS)
    def test_create_model_script_succeeds(self, ext):
        proc = subprocess.run(
            [sys.executable, "create_model.py"],
            cwd=ROOT / ext,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr


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
        assert set(np.unique(out["target"])).issubset({0, 1})


class TestInferenceAccuracy:
    @pytest.mark.parametrize("ext", EXTENSIONS)
    def test_inference_produces_valid_accuracy(self, ext, credit_labels, train_test_indices):
        from training_common import get_inference_row_indices

        _, test_idx, _split = train_test_indices
        inf_indices = get_inference_row_indices()
        inf_test_mask = np.isin(inf_indices, test_idx)
        preds = pd.read_csv(ROOT / ext / "output" / "output.csv")["target"].to_numpy()
        y_inf = credit_labels[inf_indices]
        if inf_test_mask.any():
            inference_test_acc = accuracy_score(y_inf[inf_test_mask], preds[inf_test_mask])
            assert 0.0 <= inference_test_acc <= 1.0

    @pytest.mark.parametrize("ext", SKLEARN_PARITY_EXTENSIONS)
    def test_sklearn_family_matches_pkl_predictions(self, ext):
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


class TestAccuracyReport:
    def test_accuracy_report_exists(self):
        report = ROOT / "pkl" / "output" / "accuracy_report.csv"
        assert report.exists()
        df = pd.read_csv(report)
        assert set(df["extension"]) == set(EXTENSIONS)
