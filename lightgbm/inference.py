import os
from pathlib import Path

import lightgbm as lgb
import numpy as np

from inference_common import build_arg_parser, run_inference

_BOOSTER = None


def discover_model_path() -> str:
    model_path = Path("model") / "model.txt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    return str(model_path)


def load_model(model_path: str) -> lgb.Booster:
    global _BOOSTER
    if not os.path.exists(model_path):
        model_path = discover_model_path()
    _BOOSTER = lgb.Booster(model_file=model_path)
    return _BOOSTER


def _predict(X, _df):
    if _BOOSTER is None:
        raise RuntimeError("LightGBM model not loaded.")
    raw = _BOOSTER.predict(X.values)
    raw = np.asarray(raw)
    if raw.ndim > 1:
        return np.argmax(raw, axis=1)
    return np.rint(raw).astype(int)


def main():
    parser = build_arg_parser("Run LightGBM model inference", "model/model.txt")
    args = parser.parse_args()
    load_model(args.model)
    run_inference(_predict, args.data, args.output, args.schema)


if __name__ == "__main__":
    main()
