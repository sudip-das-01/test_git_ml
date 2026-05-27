"""Deploy input, create models per extension, run python inference.py, report accuracy."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from training_common import (
    CANONICAL_INPUT,
    EXTENSIONS,
    ROOT,
    deploy_all,
    get_inference_row_indices,
    get_training_data,
    metrics_for_predict_fn,
    save_extension_metrics,
    train_logistic_regression,
)

RESULTS_PATH = ROOT / "pkl" / "output" / "accuracy_report.csv"
SKLEARN_LR_EXTENSIONS = {"pkl", "joblib", "dill", "pickle", "onnx", "pmml", "zip"}


def run_create_model(ext: str) -> None:
    proc = subprocess.run(
        [sys.executable, "create_model.py"],
        cwd=ROOT / ext,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"create_model.py failed for {ext}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def run_inference(ext: str) -> pd.DataFrame:
    proc = subprocess.run(
        [sys.executable, "inference.py"],
        cwd=ROOT / ext,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"inference.py failed for {ext}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return pd.read_csv(ROOT / ext / "output" / "output.csv")


def sklearn_metrics(X, y, train_idx, test_idx) -> dict[str, float]:
    model = train_logistic_regression(X, y, train_idx)
    return metrics_for_predict_fn(lambda data, _m=model: _m.predict(data), X, y, train_idx, test_idx)


def main() -> int:
    deploy_all()
    print(f"Using dataset: {CANONICAL_INPUT}")
    print(f"Deployed as dataset/input.csv in: {EXTENSIONS}\n")

    X, y, train_idx, test_idx, _ = get_training_data()
    lr_metrics = sklearn_metrics(X, y, train_idx, test_idx)
    extension_metrics: dict[str, dict[str, float]] = {}
    inf_indices = get_inference_row_indices()
    y_inf = y[inf_indices]
    inf_train_mask = np.isin(inf_indices, train_idx)
    inf_test_mask = np.isin(inf_indices, test_idx)

    rows = []
    for ext in EXTENSIONS:
        print(f"=== {ext} ===")
        run_create_model(ext)
        if ext in SKLEARN_LR_EXTENSIONS:
            extension_metrics[ext] = lr_metrics
        else:
            extension_metrics[ext] = {"train_accuracy": 0.0, "test_accuracy": 0.0}

        preds_df = run_inference(ext)
        preds = preds_df["target"].to_numpy()

        inference_full_acc = float(accuracy_score(y_inf, preds))
        inference_train_acc = (
            float(accuracy_score(y_inf[inf_train_mask], preds[inf_train_mask]))
            if inf_train_mask.any()
            else float("nan")
        )
        inference_test_acc = (
            float(accuracy_score(y_inf[inf_test_mask], preds[inf_test_mask]))
            if inf_test_mask.any()
            else float("nan")
        )

        if ext not in SKLEARN_LR_EXTENSIONS:
            extension_metrics[ext] = {
                "train_accuracy": inference_train_acc,
                "test_accuracy": inference_test_acc,
            }

        metrics = extension_metrics[ext]
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
            f"train={metrics['train_accuracy']:.4f} test={metrics['test_accuracy']:.4f} "
            f"inference_test={inference_test_acc:.4f}\n"
        )

    save_extension_metrics(extension_metrics, train_idx, test_idx)
    report = pd.DataFrame(rows)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(RESULTS_PATH, index=False)
    print(f"Saved report: {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
