import argparse
import os
import pickle

import pandas as pd


def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def run_inference(model, data_path: str, output_path: str):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    df = pd.read_csv(data_path)

    forbidden_cols = [col for col in ["target", "prediction"] if col in df.columns]
    if forbidden_cols:
        raise ValueError(
            "Input dataset must not contain 'target' or 'prediction' columns. "
            f"Found forbidden columns: {forbidden_cols}"
        )

    preds = model.predict(df)
    pd.DataFrame({"target": preds}).to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Run model inference")
    parser.add_argument("--data", type=str, default="dataset/input.csv")
    parser.add_argument("--model", type=str, default="model/model.pkl")
    parser.add_argument("--output", type=str, default="output/output.csv")
    args = parser.parse_args()

    model = load_model(args.model)
    run_inference(model, args.data, args.output)


if __name__ == "__main__":
    main()

