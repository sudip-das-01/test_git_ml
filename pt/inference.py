import os
from pathlib import Path

import numpy as np
import torch

from inference_common import build_arg_parser, run_inference
from model_net import NormalizedIrisNet

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
