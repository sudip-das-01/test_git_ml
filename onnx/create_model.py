"""Create model.onnx in this folder."""
from pathlib import Path
import sys

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pkl" / "scripts"))
from training_common import get_training_data, train_logistic_regression


def main() -> None:
    X, y, train_idx, _, feature_cols = get_training_data()
    model = train_logistic_regression(X, y, train_idx)
    onnx_model = convert_sklearn(
        model, initial_types=[("float_input", FloatTensorType([None, len(feature_cols)]))]
    )
    out = Path(__file__).resolve().parent / "model" / "model.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
