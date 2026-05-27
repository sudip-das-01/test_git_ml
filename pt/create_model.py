"""Create model.pt in this folder."""
from pathlib import Path
import sys

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pkl" / "scripts"))
from training_common import NUM_CLASSES, TORCH_EPOCHS, TORCH_HIDDEN_UNITS, TORCH_LR, get_training_data

from model_net import NormalizedIrisNet


def main() -> None:
    X, y, train_idx, _, feature_cols = get_training_data()
    X_train = X.iloc[train_idx].values
    model = NormalizedIrisNet(
        len(feature_cols), NUM_CLASSES, X_train, hidden=TORCH_HIDDEN_UNITS
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=TORCH_LR)
    loss_fn = nn.CrossEntropyLoss()
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y[train_idx], dtype=torch.long)
    model.train()
    for _ in range(TORCH_EPOCHS):
        optimizer.zero_grad()
        loss_fn(model(X_tensor), y_tensor).backward()
        optimizer.step()
    model.eval()
    out = Path(__file__).resolve().parent / "model" / "model.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
