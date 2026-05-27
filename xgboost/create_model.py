"""Create model.json in this folder."""
from pathlib import Path
import sys

import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pkl" / "scripts"))
from training_common import N_ESTIMATORS, RANDOM_STATE, get_training_data


def main() -> None:
    X, y, train_idx, _, _ = get_training_data()
    model = xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=3,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )
    model.fit(X.iloc[train_idx], y[train_idx])
    out = Path(__file__).resolve().parent / "model" / "model.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
