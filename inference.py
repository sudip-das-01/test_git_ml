import argparse
import json
import os
import pickle
from pathlib import Path

import pandas as pd


def load_model(model_path: str):
    if not os.path.exists(model_path):
        model_path = discover_model_path()

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def discover_model_path() -> str:
    model_dir = Path("model")
    preferred = model_dir / "model.pkl"
    if preferred.exists():
        return str(preferred)

    matches = sorted(model_dir.glob("model.*"))
    if not matches:
        raise FileNotFoundError("Missing model file. Expected model/model.<ext> (e.g. model/model.pkl).")

    # Prefer pickle if multiple matches exist
    for p in matches:
        if p.suffix.lower() in (".pkl", ".pickle"):
            return str(p)

    if len(matches) == 1:
        return str(matches[0])

    raise FileNotFoundError(f"Multiple candidate models found: {[str(p) for p in matches]}")


def discover_input_path() -> str:
    dataset_dir = Path("dataset")
    preferred = dataset_dir / "input.csv"
    if preferred.exists():
        return str(preferred)

    matches = sorted(dataset_dir.glob("input.*"))
    if not matches:
        raise FileNotFoundError("Missing input file. Expected dataset/input.<ext> (e.g. dataset/input.csv).")

    # Prefer CSV if multiple matches exist
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


def run_inference(model, data_path: str, output_path: str, schema_path: str):
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
    preds = model.predict(X)

    output_df = pd.DataFrame({"target": preds})
    if "id" in df.columns:
        output_df.insert(0, "id", df["id"])

    pd.DataFrame(output_df).to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Run model inference")
    parser.add_argument("--data", type=str, default="dataset/input.csv")
    parser.add_argument("--model", type=str, default="model/model.pkl")
    parser.add_argument("--output", type=str, default="output/output.csv")
    parser.add_argument("--schema", type=str, default="schema.json")
    args = parser.parse_args()

    model = load_model(args.model)
    run_inference(model, args.data, args.output, args.schema)


if __name__ == "__main__":
    main()

