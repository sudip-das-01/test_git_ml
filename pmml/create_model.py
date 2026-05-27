"""Create model.pmml in this folder."""
from pathlib import Path
import sys

from nyoka.skl.skl_to_pmml import skl_to_pmml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pkl" / "scripts"))
from training_common import RANDOM_STATE, get_training_data


def main() -> None:
    X, y, train_idx, _, feature_cols = get_training_data()
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]
    )
    model.fit(X.iloc[train_idx], y[train_idx])
    out = Path(__file__).resolve().parent / "model" / "model.pmml"
    out.parent.mkdir(parents=True, exist_ok=True)
    skl_to_pmml(
        model,
        feature_cols,
        target_name="target",
        pmml_f_name=str(out),
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
