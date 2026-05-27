import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from inference_common import build_arg_parser, run_inference

_MODEL = None
_TEMP_DIR = None


def discover_model_path() -> str:
    model_path = Path("model") / "model.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    return str(model_path)


def load_model(model_path: str):
    global _MODEL, _TEMP_DIR
    if not os.path.exists(model_path):
        model_path = discover_model_path()

    import mlflow.pyfunc

    _TEMP_DIR = tempfile.mkdtemp(prefix="mlflow_model_")
    with zipfile.ZipFile(model_path, "r") as archive:
        archive.extractall(_TEMP_DIR)

    _MODEL = mlflow.pyfunc.load_model(_TEMP_DIR)
    return _MODEL


def _predict(X, _df):
    if _MODEL is None:
        raise RuntimeError("MLflow model not loaded.")
    preds = _MODEL.predict(X)
    return np.asarray(preds).reshape(-1).astype(int)


def main():
    parser = build_arg_parser("Run MLflow ZIP model inference", "model/model.zip")
    args = parser.parse_args()
    try:
        load_model(args.model)
        run_inference(_predict, args.data, args.output, args.schema)
    finally:
        global _TEMP_DIR
        if _TEMP_DIR and os.path.isdir(_TEMP_DIR):
            shutil.rmtree(_TEMP_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
