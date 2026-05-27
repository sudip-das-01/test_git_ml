import os
from pathlib import Path

import joblib

from inference_common import build_arg_parser, run_inference


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
