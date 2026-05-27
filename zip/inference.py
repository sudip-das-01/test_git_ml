import argparse
import json
import os
from pathlib import Path
from typing import Callable

import pandas as pd


def discover_input_path() -> str:
    dataset_dir = Path("dataset")
    preferred = dataset_dir / "input.csv"
    if preferred.exists():
        return str(preferred)

    matches = sorted(dataset_dir.glob("input.*"))
    if not matches:
        raise FileNotFoundError("Missing input file. Expected dataset/input.<ext> (e.g. dataset/input.csv).")

    for p in matches:
        if p.suffix.lower() == ".csv":
            return str(p)

    if len(matches) == 1:
        return str(matches[0])

    raise FileNotFoundError(f"Multiple candidate inputs found: {[str(p) for p in matches]}")


def load_schema_features(schema_path: str) -> list[str]:
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Missing schema file: {schema_path}")

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    feature_cols = schema.get("input_parameters_name")
    if not isinstance(feature_cols, list) or not feature_cols:
        raise ValueError("schema.json must contain a non-empty 'input_parameters_name' list.")

    if not all(isinstance(col, str) for col in feature_cols):
        raise ValueError("'input_parameters_name' must contain only string column names.")

    return feature_cols


def run_inference(predict_fn: Callable, data_path: str, output_path: str, schema_path: str):
    if not os.path.exists(data_path):
        data_path = discover_input_path()

    df = pd.read_csv(data_path)

    forbidden_cols = [col for col in ["target", "prediction"] if col in df.columns]
    if forbidden_cols:
        raise ValueError(
            "Input dataset must not contain 'target' or 'prediction' columns. "
            f"Found forbidden columns: {forbidden_cols}"
        )

    feature_cols = load_schema_features(schema_path)
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            "Input dataset is missing required feature columns from schema.json: "
            f"{missing_cols}"
        )

    X = df[feature_cols]
    preds = predict_fn(X, df)

    output_df = pd.DataFrame({"target": preds})
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)


def build_arg_parser(description: str, default_model: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data", type=str, default="dataset/input.csv")
    parser.add_argument("--model", type=str, default=default_model)
    parser.add_argument("--output", type=str, default="output/output.csv")
    parser.add_argument("--schema", type=str, default="schema.json")
    return parser



import shutil
import tempfile
import zipfile

import numpy as np

_MODEL = None
_TEMP_DIR = None


def discover_model_path() -> str:
    model_path = Path("model") / "model.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    return str(model_path)


def load_model(model_path: str):
    global _MODEL, _TEMP_DIR
    if not os.path.exists(model_path):
        model_path = discover_model_path()

    import mlflow.pyfunc

    _TEMP_DIR = tempfile.mkdtemp(prefix="mlflow_model_")
    with zipfile.ZipFile(model_path, "r") as archive:
        archive.extractall(_TEMP_DIR)
    _MODEL = mlflow.pyfunc.load_model(_TEMP_DIR)
    return _MODEL


def _predict(X, _df):
    if _MODEL is None:
        raise RuntimeError("MLflow model not loaded.")
    preds = _MODEL.predict(X)
    return np.asarray(preds).reshape(-1).astype(int)


def main():
    parser = build_arg_parser("Run MLflow ZIP model inference", "model/model.zip")
    args = parser.parse_args()
    try:
        load_model(args.model)
        run_inference(_predict, args.data, args.output, args.schema)
    finally:
        global _TEMP_DIR
        if _TEMP_DIR and os.path.isdir(_TEMP_DIR):
            shutil.rmtree(_TEMP_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
