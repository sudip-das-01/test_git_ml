"""Create model.cbm in this folder."""
from pathlib import Path
import sys

from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pkl" / "scripts"))
from training_common import N_ESTIMATORS, RANDOM_STATE, get_training_data


def main() -> None:
    X, y, train_idx, _, _ = get_training_data()
    train_dir = Path(__file__).resolve().parent / "catboost_info"
    model = CatBoostClassifier(
        iterations=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        verbose=False,
        train_dir=str(train_dir),
    )
    model.fit(X.iloc[train_idx], y[train_idx])
    out = Path(__file__).resolve().parent / "model" / "model.cbm"
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
