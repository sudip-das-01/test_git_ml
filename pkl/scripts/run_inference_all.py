"""Deploy canonical input.csv to all extensions and run python inference.py in each."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from training_common import EXTENSIONS, ROOT, deploy_all, get_inference_row_indices, get_training_data

RESULTS_PATH = ROOT / "pkl" / "output" / "inference_report.csv"


def main() -> int:
    deploy_all()
    print("Deployed schema.json and input with columns: id, <schema features>, class (empty)\n")

    _, y, train_idx, test_idx, _ = get_training_data()
    inf_indices = get_inference_row_indices()
    y_inf = y[inf_indices]
    inf_test_mask = np.isin(inf_indices, test_idx)

    rows = []
    for ext in EXTENSIONS:
        proc = subprocess.run(
            [sys.executable, "inference.py"],
            cwd=ROOT / ext,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(f"FAILED {ext}\n{proc.stderr}")
            return 1

        out = pd.read_csv(ROOT / ext / "output" / "output.csv")
        preds = out["target"].to_numpy()
        rows.append(
            {
                "extension": ext,
                "output_rows": len(preds),
                "inference_test_accuracy": round(
                    float(accuracy_score(y_inf[inf_test_mask], preds[inf_test_mask])), 4
                )
                if inf_test_mask.any()
                else None,
                "inference_full_accuracy": round(float(accuracy_score(y_inf, preds)), 4),
            }
        )
        print(f"{ext}: ok ({len(preds)} rows, test acc={rows[-1]['inference_test_accuracy']})")

    report = pd.DataFrame(rows)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
