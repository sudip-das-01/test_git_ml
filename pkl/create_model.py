"""Create model.pkl in this folder."""
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pkl" / "scripts"))
from training_common import get_training_data, train_logistic_regression


def main() -> None:
    X, y, train_idx, _, _ = get_training_data()
    model = train_logistic_regression(X, y, train_idx)
    out = Path(__file__).resolve().parent / "model" / "model.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
