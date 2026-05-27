import os
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier

from inference_common import build_arg_parser, run_inference

_MODEL = None


def discover_model_path() -> str:
    model_path = Path("model") / "model.cbm"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    return str(model_path)


def load_model(model_path: str) -> CatBoostClassifier:
    global _MODEL
    if not os.path.exists(model_path):
        model_path = discover_model_path()
    model = CatBoostClassifier()
    model.load_model(model_path)
    _MODEL = model
    return model


def _predict(X, _df):
    if _MODEL is None:
        raise RuntimeError("CatBoost model not loaded.")
    preds = _MODEL.predict(X)
    return np.asarray(preds).reshape(-1).astype(int)


def main():
    parser = build_arg_parser("Run CatBoost model inference", "model/model.cbm")
    args = parser.parse_args()
    load_model(args.model)
    run_inference(_predict, args.data, args.output, args.schema)


if __name__ == "__main__":
    main()
