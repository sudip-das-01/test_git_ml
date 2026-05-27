"""Generate a single self-contained inference.py per extension folder."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

IO_BLOCK = '''import argparse
import json
import os
from pathlib import Path
from typing import Callable

import pandas as pd


def discover_input_path() -> str:
    dataset_dir = Path("dataset")
    preferred = dataset_dir / "input.csv"
    if preferred.exists():
        return str(preferred)

    matches = sorted(dataset_dir.glob("input.*"))
    if not matches:
        raise FileNotFoundError("Missing input file. Expected dataset/input.<ext> (e.g. dataset/input.csv).")

    for p in matches:
        if p.suffix.lower() == ".csv":
            return str(p)

    if len(matches) == 1:
        return str(matches[0])

    raise FileNotFoundError(f"Multiple candidate inputs found: {[str(p) for p in matches]}")


def load_schema_features(schema_path: str) -> list[str]:
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Missing schema file: {schema_path}")

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    feature_cols = schema.get("input_parameters_name")
    if not isinstance(feature_cols, list) or not feature_cols:
        raise ValueError("schema.json must contain a non-empty 'input_parameters_name' list.")

    if not all(isinstance(col, str) for col in feature_cols):
        raise ValueError("'input_parameters_name' must contain only string column names.")

    return feature_cols


def run_inference(predict_fn: Callable, data_path: str, output_path: str, schema_path: str):
    if not os.path.exists(data_path):
        data_path = discover_input_path()

    df = pd.read_csv(data_path)

    forbidden_cols = [col for col in ["target", "prediction"] if col in df.columns]
    if forbidden_cols:
        raise ValueError(
            "Input dataset must not contain 'target' or 'prediction' columns. "
            f"Found forbidden columns: {forbidden_cols}"
        )

    feature_cols = load_schema_features(schema_path)
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            "Input dataset is missing required feature columns from schema.json: "
            f"{missing_cols}"
        )

    X = df[feature_cols]
    preds = predict_fn(X, df)

    output_df = pd.DataFrame({"target": preds})
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)


def build_arg_parser(description: str, default_model: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data", type=str, default="dataset/input.csv")
    parser.add_argument("--model", type=str, default=default_model)
    parser.add_argument("--output", type=str, default="output/output.csv")
    parser.add_argument("--schema", type=str, default="schema.json")
    return parser


'''

TAILS: dict[str, str] = {
    "pkl": '''import pickle


def discover_model_path() -> str:
    model_path = Path("model") / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    return str(model_path)


def load_model(model_path: str):
    if not os.path.exists(model_path):
        model_path = discover_model_path()
    with open(model_path, "rb") as f:
        return pickle.load(f)


def main():
    parser = build_arg_parser("Run PKL model inference", "model/model.pkl")
    args = parser.parse_args()
    model = load_model(args.model)
    run_inference(lambda X, _df: model.predict(X), args.data, args.output, args.schema)


if __name__ == "__main__":
    main()
''',
    "pickle": '''import pickle


def discover_model_path() -> str:
    model_path = Path("model") / "model.pickle"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    return str(model_path)


def load_model(model_path: str):
    if not os.path.exists(model_path):
        model_path = discover_model_path()
    with open(model_path, "rb") as f:
        return pickle.load(f)


def main():
    parser = build_arg_parser("Run Pickle model inference", "model/model.pickle")
    args = parser.parse_args()
    model = load_model(args.model)
    run_inference(lambda X, _df: model.predict(X), args.data, args.output, args.schema)


if __name__ == "__main__":
    main()
''',
    "joblib": '''import joblib


def discover_model_path() -> str:
    model_path = Path("model") / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    return str(model_path)


def load_model(model_path: str):
    if not os.path.exists(model_path):
        model_path = discover_model_path()
    return joblib.load(model_path)


def main():
    parser = build_arg_parser("Run Joblib model inference", "model/model.joblib")
    args = parser.parse_args()
    model = load_model(args.model)
    run_inference(lambda X, _df: model.predict(X), args.data, args.output, args.schema)


if __name__ == "__main__":
    main()
''',
    "dill": '''import dill


def discover_model_path() -> str:
    model_path = Path("model") / "model.dill"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    return str(model_path)


def load_model(model_path: str):
    if not os.path.exists(model_path):
        model_path = discover_model_path()
    with open(model_path, "rb") as f:
        return dill.load(f)


def main():
    parser = build_arg_parser("Run Dill model inference", "model/model.dill")
    args = parser.parse_args()
    model = load_model(args.model)
    run_inference(lambda X, _df: model.predict(X), args.data, args.output, args.schema)


if __name__ == "__main__":
    main()
''',
    "onnx": '''import numpy as np
import onnxruntime as ort

_SESSION = None


def discover_model_path() -> str:
    model_path = Path("model") / "model.onnx"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    return str(model_path)


def load_model(model_path: str) -> ort.InferenceSession:
    global _SESSION
    if not os.path.exists(model_path):
        model_path = discover_model_path()
    _SESSION = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return _SESSION


def _predict_onnx(X, _df):
    global _SESSION
    if _SESSION is None:
        raise RuntimeError("ONNX model not loaded. Call load_model first.")
    input_name = _SESSION.get_inputs()[0].name
    arr = X.values.astype(np.float32)
    outputs = _SESSION.run(None, {input_name: arr})
    result = outputs[0]
    if isinstance(result, list):
        result = np.array(result)
    if result.ndim > 1 and result.shape[1] > 1:
        return np.argmax(result, axis=1)
    return np.asarray(result).reshape(-1)


def main():
    parser = build_arg_parser("Run ONNX model inference", "model/model.onnx")
    args = parser.parse_args()
    load_model(args.model)
    run_inference(_predict_onnx, args.data, args.output, args.schema)


if __name__ == "__main__":
    main()
''',
    "pmml": '''import numpy as np
from pypmml import Model

_PMML_MODEL = None


def discover_model_path() -> str:
    model_path = Path("model") / "model.pmml"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    return str(model_path)


def load_model(model_path: str) -> Model:
    global _PMML_MODEL
    if not os.path.exists(model_path):
        model_path = discover_model_path()
    _PMML_MODEL = Model.fromFile(model_path)
    return _PMML_MODEL


def _extract_predictions(result) -> np.ndarray:
    if isinstance(result, pd.DataFrame):
        for col in ("predicted_target", "predicted_class", "target", "prediction", "class"):
            if col in result.columns:
                return pd.to_numeric(result[col], errors="raise").astype(int).to_numpy()
        if len(result.columns) == 1:
            return pd.to_numeric(result.iloc[:, 0], errors="raise").astype(int).to_numpy()
    return np.asarray(result).reshape(-1).astype(int)


def _predict_pmml(X, _df):
    global _PMML_MODEL
    if _PMML_MODEL is None:
        raise RuntimeError("PMML model not loaded. Call load_model first.")
    return _extract_predictions(_PMML_MODEL.predict(X))


def main():
    parser = build_arg_parser("Run PMML model inference", "model/model.pmml")
    args = parser.parse_args()
    load_model(args.model)
    run_inference(_predict_pmml, args.data, args.output, args.schema)


if __name__ == "__main__":
    main()
''',
    "xgboost": '''import xgboost as xgb

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
''',
    "lightgbm": '''import lightgbm as lgb

_MODEL = None


def discover_model_path() -> str:
    model_path = Path("model") / "model.txt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    return str(model_path)


def load_model(model_path: str):
    global _MODEL
    if not os.path.exists(model_path):
        model_path = discover_model_path()
    _MODEL = lgb.Booster(model_file=model_path)
    return _MODEL


def _predict(X, _df):
    if _MODEL is None:
        raise RuntimeError("LightGBM model not loaded.")
    return _MODEL.predict(X).astype(int)


def main():
    parser = build_arg_parser("Run LightGBM model inference", "model/model.txt")
    args = parser.parse_args()
    load_model(args.model)
    run_inference(_predict, args.data, args.output, args.schema)


if __name__ == "__main__":
    main()
''',
    "h5": '''import numpy as np

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
''',
    "pt": '''import numpy as np
import torch

from model import NormalizedCreditNet, NormalizedIrisNet  # required for torch.load unpickling

_MODEL = None


def discover_model_path() -> str:
    model_path = Path("model") / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    return str(model_path)


def load_model(model_path: str):
    global _MODEL
    if not os.path.exists(model_path):
        model_path = discover_model_path()
    _MODEL = torch.load(model_path, map_location="cpu", weights_only=False)
    _MODEL.eval()
    return _MODEL


def _predict(X, _df):
    if _MODEL is None:
        raise RuntimeError("PyTorch model not loaded.")
    with torch.no_grad():
        tensor = torch.tensor(X.values, dtype=torch.float32)
        logits = _MODEL(tensor)
        return torch.argmax(logits, dim=1).cpu().numpy().astype(int)


def main():
    parser = build_arg_parser("Run PyTorch PT model inference", "model/model.pt")
    args = parser.parse_args()
    load_model(args.model)
    run_inference(_predict, args.data, args.output, args.schema)


if __name__ == "__main__":
    main()
''',
    "pb": '''import numpy as np
import tensorflow as tf

_MODEL = None
_SIGNATURE = None


def discover_model_path() -> str:
    model_dir = Path("model")
    if not model_dir.exists():
        raise FileNotFoundError("Missing SavedModel directory: model/")
    return str(model_dir)


def load_model(model_path: str):
    global _MODEL, _SIGNATURE
    if not os.path.exists(model_path):
        model_path = discover_model_path()
    _MODEL = tf.saved_model.load(model_path)
    _SIGNATURE = _MODEL.signatures["serving_default"]
    return _MODEL


def _predict(X, _df):
    if _SIGNATURE is None:
        raise RuntimeError("TensorFlow SavedModel not loaded.")
    input_key = list(_SIGNATURE.structured_input_signature[1].keys())[0]
    output = _SIGNATURE(**{input_key: tf.constant(X.values, dtype=tf.float32)})
    output_key = list(output.keys())[0]
    values = output[output_key].numpy()
    if values.ndim > 1:
        return np.argmax(values, axis=1).astype(int)
    return np.rint(values).astype(int)


def main():
    parser = build_arg_parser("Run TensorFlow SavedModel inference", "model")
    args = parser.parse_args()
    load_model(args.model)
    run_inference(_predict, args.data, args.output, args.schema)


if __name__ == "__main__":
    main()
''',
    "zip": '''import shutil
import tempfile
import zipfile

import numpy as np

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
''',
}


PT_MODEL_PY = '''"""PyTorch architecture for German credit classification (train + inference)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class NormalizedCreditNet(nn.Module):
    """MLP with StandardScaler-style normalization baked into the graph."""

    def __init__(self, n_in: int, n_out: int, X_train: np.ndarray, hidden: int = 32):
        super().__init__()
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std == 0] = 1.0
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return self.net(x)


# Backward compatibility for models saved before rename.
NormalizedIrisNet = NormalizedCreditNet
'''


def main() -> None:
    pt_model_path = ROOT / "pt" / "model.py"
    pt_model_path.write_text(PT_MODEL_PY, encoding="utf-8")
    print(f"Wrote {pt_model_path}")

    for ext, tail in TAILS.items():
        path = ROOT / ext / "inference.py"
        path.write_text(IO_BLOCK + "\n" + tail, encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
