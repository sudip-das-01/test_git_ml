import os
from pathlib import Path

import xgboost as xgb

from inference_common import build_arg_parser, run_inference

_MODEL = None


def discover_model_path() -> str:
    model_path = Path("model") / "model.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    return str(model_path)


def load_model(model_path: str) -> xgb.XGBClassifier:
    global _MODEL
    if not os.path.exists(model_path):
        model_path = discover_model_path()
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    _MODEL = model
    return model


def _predict(X, _df):
    if _MODEL is None:
        raise RuntimeError("XGBoost model not loaded.")
    return _MODEL.predict(X)


def main():
    parser = build_arg_parser("Run XGBoost JSON model inference", "model/model.json")
    args = parser.parse_args()
    load_model(args.model)
    run_inference(_predict, args.data, args.output, args.schema)


if __name__ == "__main__":
    main()
