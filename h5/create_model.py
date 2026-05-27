"""Create model.h5 in this folder."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pkl" / "scripts"))
from training_common import NUM_CLASSES, build_keras_classifier, fit_keras_classifier, get_training_data


def main() -> None:
    X, y, train_idx, _, feature_cols = get_training_data()
    X_train = X.iloc[train_idx].values
    model = build_keras_classifier(len(feature_cols), NUM_CLASSES, X_train)
    fit_keras_classifier(model, X_train, y[train_idx])
    out = Path(__file__).resolve().parent / "model" / "model.h5"
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
