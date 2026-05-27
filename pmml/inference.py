import os
from pathlib import Path

import numpy as np
import pandas as pd
from pypmml import Model

from inference_common import build_arg_parser, load_schema_features, run_inference

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

    result = _PMML_MODEL.predict(X)
    return _extract_predictions(result)


def main():
    parser = build_arg_parser("Run PMML model inference", "model/model.pmml")
    args = parser.parse_args()

    load_model(args.model)
    load_schema_features(args.schema)
    run_inference(_predict_pmml, args.data, args.output, args.schema)


if __name__ == "__main__":
    main()
