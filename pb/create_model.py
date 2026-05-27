"""Create TensorFlow SavedModel under model/ in this folder."""
import shutil
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
    out_dir = Path(__file__).resolve().parent / "model"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.export(str(out_dir))
    print(f"Saved SavedModel to {out_dir}")


if __name__ == "__main__":
    main()
