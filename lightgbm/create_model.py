"""Create model.txt in this folder."""
from pathlib import Path
import sys

import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pkl" / "scripts"))
from training_common import N_ESTIMATORS, RANDOM_STATE, get_training_data


def main() -> None:
    X, y, train_idx, _, _ = get_training_data()
    model = lgb.LGBMClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        verbosity=-1,
    )
    model.fit(X.iloc[train_idx], y[train_idx])
    out = Path(__file__).resolve().parent / "model" / "model.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(out))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
