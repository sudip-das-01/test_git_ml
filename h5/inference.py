import os
from pathlib import Path

import numpy as np

from inference_common import build_arg_parser, run_inference

_MODEL = None


def discover_model_path() -> str:
    model_path = Path("model") / "model.h5"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    return str(model_path)


def load_model(model_path: str):
    global _MODEL
    if not os.path.exists(model_path):
        model_path = discover_model_path()
    from tensorflow import keras

    _MODEL = keras.models.load_model(model_path)
    return _MODEL


def _predict(X, _df):
    if _MODEL is None:
        raise RuntimeError("Keras model not loaded.")
    probs = _MODEL.predict(X.values, verbose=0)
    return np.argmax(probs, axis=1).astype(int)


def main():
    parser = build_arg_parser("Run Keras H5 model inference", "model/model.h5")
    args = parser.parse_args()
    load_model(args.model)
    run_inference(_predict, args.data, args.output, args.schema)


if __name__ == "__main__":
    main()
