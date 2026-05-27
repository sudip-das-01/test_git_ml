"""Train model and export to all extension folders."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import dill
import joblib
import numpy as np
import pandas as pd
from nyoka.skl.skl_to_pmml import skl_to_pmml
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

ROOT = Path(__file__).resolve().parents[2]
EXTENSIONS = ["pkl", "joblib", "dill", "onnx", "pmml"]
RANDOM_STATE = 42
INPUT_WITH_NOISE = ROOT / "pkl" / "dataset" / "input_with_random_column.csv"


def load_feature_columns() -> list[str]:
    with open(ROOT / "pkl" / "schema.json", "r", encoding="utf-8") as f:
        return json.load(f)["input_parameters_name"]


def train_model():
    feature_cols = load_feature_columns()
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=feature_cols)
    y = iris.target

    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    model = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
    model.fit(X.iloc[train_idx], y[train_idx])

    metrics = {
        "train_accuracy": float(accuracy_score(y[train_idx], model.predict(X.iloc[train_idx]))),
        "test_accuracy": float(accuracy_score(y[test_idx], model.predict(X.iloc[test_idx]))),
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "random_state": RANDOM_STATE,
    }

    split_path = ROOT / "pkl" / "dataset" / "train_test_split.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(
            {"train_indices": train_idx.tolist(), "test_indices": test_idx.tolist(), **metrics},
            f,
            indent=2,
        )

    return model, feature_cols, y, train_idx, test_idx, metrics


def export_models(model, feature_cols: list[str]) -> None:
    (ROOT / "pkl" / "model").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "pkl" / "model" / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    (ROOT / "joblib" / "model").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ROOT / "joblib" / "model" / "model.joblib")

    (ROOT / "dill" / "model").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "dill" / "model" / "model.dill", "wb") as f:
        dill.dump(model, f)

    (ROOT / "onnx" / "model").mkdir(parents=True, exist_ok=True)
    onnx_model = convert_sklearn(
        model, initial_types=[("float_input", FloatTensorType([None, len(feature_cols)]))]
    )
    with open(ROOT / "onnx" / "model" / "model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())

    (ROOT / "pmml" / "model").mkdir(parents=True, exist_ok=True)
    skl_to_pmml(
        Pipeline([("classifier", model)]),
        feature_cols,
        target_name="target",
        pmml_f_name=str(ROOT / "pmml" / "model" / "model.pmml"),
    )


def deploy_input_csv() -> Path:
    if not INPUT_WITH_NOISE.exists():
        raise FileNotFoundError(f"Missing shared input file: {INPUT_WITH_NOISE}")

    for ext in EXTENSIONS:
        dataset_dir = ROOT / ext / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        target = dataset_dir / "input.csv"
        target.write_bytes(INPUT_WITH_NOISE.read_bytes())

    return INPUT_WITH_NOISE
