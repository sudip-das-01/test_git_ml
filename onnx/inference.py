import os
from pathlib import Path

import numpy as np
import onnxruntime as ort

from inference_common import build_arg_parser, run_inference

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
