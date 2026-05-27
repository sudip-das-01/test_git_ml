"""Train, export, deploy input.csv, run inference, and print accuracy scores."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from train_and_export import EXTENSIONS, INPUT_WITH_NOISE, ROOT, deploy_input_csv, export_models, train_model

RESULTS_PATH = ROOT / "pkl" / "output" / "accuracy_report.csv"


def run_inference_for_extension(ext: str) -> pd.DataFrame:
    folder = ROOT / ext
    output_path = folder / "output" / "output.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    proc = subprocess.run(
        [sys.executable, "inference.py"],
        cwd=folder,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Inference failed for {ext}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    return pd.read_csv(output_path)


def main() -> int:
    model, feature_cols, y, train_idx, test_idx, metrics = train_model()
    export_models(model, feature_cols)
    deploy_input_csv()

    print(f"Deployed {INPUT_WITH_NOISE} as input.csv in: {EXTENSIONS}")
    print(f"Sklearn train accuracy: {metrics['train_accuracy']:.4f}")
    print(f"Sklearn test accuracy:  {metrics['test_accuracy']:.4f}")

    rows = []
    for ext in EXTENSIONS:
        preds_df = run_inference_for_extension(ext)
        preds = preds_df["target"].to_numpy()

        inference_full_acc = float(accuracy_score(y, preds))
        inference_train_acc = float(accuracy_score(y[train_idx], preds[train_idx]))
        inference_test_acc = float(accuracy_score(y[test_idx], preds[test_idx]))

        rows.append(
            {
                "extension": ext,
                "train_accuracy": round(metrics["train_accuracy"], 4),
                "test_accuracy": round(metrics["test_accuracy"], 4),
                "inference_full_accuracy": round(inference_full_acc, 4),
                "inference_train_accuracy": round(inference_train_acc, 4),
                "inference_test_accuracy": round(inference_test_acc, 4),
                "output_rows": len(preds),
            }
        )
        print(
            f"{ext}: train={metrics['train_accuracy']:.4f} "
            f"test={metrics['test_accuracy']:.4f} "
            f"inference_test={inference_test_acc:.4f} "
            f"inference_full={inference_full_acc:.4f}"
        )

    report = pd.DataFrame(rows)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved report: {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
