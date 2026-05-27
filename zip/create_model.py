"""Create model.zip (MLflow sklearn bundle) in this folder."""
import shutil
import tempfile
from pathlib import Path
import sys

import mlflow.sklearn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pkl" / "scripts"))
from training_common import get_training_data, train_logistic_regression


def main() -> None:
    X, y, train_idx, _, _ = get_training_data()
    model = train_logistic_regression(X, y, train_idx)
    out = Path(__file__).resolve().parent / "model" / "model.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    with tempfile.TemporaryDirectory() as tmp:
        mlflow.sklearn.save_model(model, tmp)
        archive_base = out.with_suffix("")
        shutil.make_archive(str(archive_base), "zip", tmp)
        generated = archive_base.with_suffix(".zip")
        if generated != out and generated.exists():
            generated.rename(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
