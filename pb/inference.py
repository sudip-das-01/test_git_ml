import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from inference_common import build_arg_parser, run_inference

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
