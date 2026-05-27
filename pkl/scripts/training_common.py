"""Shared training data, canonical input CSV, and deploy helpers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
EXTENSIONS = [
    "pkl",
    "joblib",
    "dill",
    "onnx",
    "pmml",
    "pickle",
    "xgboost",
    "lightgbm",
    "h5",
    "pt",
    "pb",
    "zip",
]
RANDOM_STATE = 42
SERIAL_COLUMN_NAME = "id"
EMPTY_COLUMN_NAME = "class"
CREDIT_INFERENCE_SOURCE = ROOT / "pkl" / "dataset" / "credit_inference_source.csv"
CANONICAL_INPUT = ROOT / "pkl" / "dataset" / "input_with_empty_column.csv"
INPUT_WITH_RANDOM_COLUMN = ROOT / "pkl" / "dataset" / "input_with_random_column.csv"
SPLIT_FILE = ROOT / "pkl" / "dataset" / "train_test_split.json"
SCHEMA_PATH = ROOT / "pkl" / "schema.json"

# Training defaults (German credit, 48 features, binary target)
N_ESTIMATORS = 100
KERAS_EPOCHS = 150
KERAS_BATCH_SIZE = 32
KERAS_HIDDEN_UNITS = 64
TORCH_EPOCHS = 300
TORCH_LR = 0.01
TORCH_HIDDEN_UNITS = 64
NUM_CLASSES = 2

OPENML_NUMERIC_MAP = {
    "duration": "duration",
    "credit_amount": "credit_amount",
    "installment_rate": "installment_commitment",
    "residence_since": "residence_since",
    "age": "age",
    "existing_credits": "existing_credits",
    "num_dependents": "num_dependents",
}

OPENML_CATEGORICAL_GROUPS = [
    ("checking_status", "status_"),
    ("credit_history", "credit_history_"),
    ("purpose", "purpose_"),
    ("savings_status", "savings_"),
    ("employment", "employment_"),
    ("personal_status", "personal_status_"),
    ("other_parties", "other_debtors_"),
    ("property_magnitude", "property_"),
    ("other_payment_plans", "other_installments_"),
    ("housing", "housing_"),
    ("job", "job_"),
    ("own_telephone", "telephone_"),
    ("foreign_worker", "foreign_worker_"),
]


def load_feature_columns() -> list[str]:
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        cols = json.load(f)["input_parameters_name"]
    return [c for c in cols if c != EMPTY_COLUMN_NAME]


def _boolish_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == bool:
            out[col] = out[col].astype(int)
    return out


def _load_openml_german_credit() -> tuple[pd.DataFrame, np.ndarray]:
    data = fetch_openml(data_id=31, as_frame=True, parser="auto")
    raw = data.data.copy()
    y = (data.target.astype(str).str.lower() == "bad").astype(int).to_numpy()
    for col in OPENML_NUMERIC_MAP.values():
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    return raw, y


def _dummy_columns_for_prefix(feature_cols: list[str], prefix: str) -> list[str]:
    return [c for c in feature_cols if c.startswith(prefix)]


def _learn_openml_category_mappings(
    reference_encoded: pd.DataFrame, openml_aligned: pd.DataFrame, feature_cols: list[str]
) -> dict[str, dict[str, str]]:
    mappings: dict[str, dict[str, str]] = {}
    for openml_col, prefix in OPENML_CATEGORICAL_GROUPS:
        dummies = _dummy_columns_for_prefix(feature_cols, prefix)
        if not dummies:
            continue
        col_map: dict[str, str] = {}
        for i in range(len(reference_encoded)):
            val = str(openml_aligned.iloc[i][openml_col])
            active = [d for d in dummies if int(reference_encoded.iloc[i][d]) == 1]
            if len(active) == 1:
                col_map[val] = active[0]
        mappings[openml_col] = col_map
    return mappings


def _encode_openml_rows(
    openml_raw: pd.DataFrame, mappings: dict[str, dict[str, str]], feature_cols: list[str]
) -> pd.DataFrame:
    rows = []
    for _, raw_row in openml_raw.iterrows():
        row = {feat: 0 for feat in feature_cols}
        for feat, openml_col in OPENML_NUMERIC_MAP.items():
            row[feat] = int(raw_row[openml_col])
        for openml_col, prefix in OPENML_CATEGORICAL_GROUPS:
            dummies = _dummy_columns_for_prefix(feature_cols, prefix)
            val = str(raw_row[openml_col])
            active = mappings.get(openml_col, {}).get(val)
            if active and active in dummies:
                row[active] = 1
        rows.append(row)
    return pd.DataFrame(rows, columns=feature_cols)


def _build_training_frame_from_openml() -> tuple[pd.DataFrame, np.ndarray]:
    feature_cols = load_feature_columns()
    reference = pd.read_csv(_resolve_credit_inference_source())
    ref_features = _boolish_frame(reference[[c for c in reference.columns if c in feature_cols]])

    openml_raw, y_all = _load_openml_german_credit()
    merge_keys = list(OPENML_NUMERIC_MAP.values())
    ref_keys = ref_features.rename(columns={"installment_rate": "installment_commitment"})
    aligned = ref_keys.merge(openml_raw, on=merge_keys, how="inner")
    if len(aligned) != len(ref_features):
        raise RuntimeError(
            f"Could not align reference credit rows to OpenML ({len(aligned)} vs {len(ref_features)})"
        )

    mappings = _learn_openml_category_mappings(
        ref_features, aligned[openml_raw.columns], feature_cols
    )
    X = _encode_openml_rows(openml_raw, mappings, feature_cols)
    return X, y_all


def _resolve_credit_inference_source() -> Path:
    import os

    env_path = os.environ.get("CREDIT_INFERENCE_SOURCE")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"CREDIT_INFERENCE_SOURCE not found: {path}")

    if CREDIT_INFERENCE_SOURCE.exists():
        return CREDIT_INFERENCE_SOURCE
    raise FileNotFoundError(
        "Missing credit inference source. Place CSV at "
        f"{CREDIT_INFERENCE_SOURCE} or set CREDIT_INFERENCE_SOURCE env var."
    )


def ensure_canonical_input() -> Path:
    """Build dataset: serial id, schema features, empty class column (no values)."""
    source = _resolve_credit_inference_source()

    feature_cols = load_feature_columns()
    df = pd.read_csv(source)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Credit input missing feature columns: {missing}")

    out = pd.DataFrame()
    out[SERIAL_COLUMN_NAME] = range(1, len(df) + 1)
    out[feature_cols] = _boolish_frame(df[feature_cols])
    out[EMPTY_COLUMN_NAME] = ""

    CANONICAL_INPUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(CANONICAL_INPUT, index=False)
    return CANONICAL_INPUT


def ensure_input_with_random_column() -> Path:
    """Build credit input CSV with an extra non-schema column (for ignore-extra-column tests)."""
    if not CANONICAL_INPUT.exists():
        ensure_canonical_input()
    df = pd.read_csv(CANONICAL_INPUT)
    rng = np.random.default_rng(RANDOM_STATE)
    df["random_noise"] = rng.standard_normal(len(df))
    INPUT_WITH_RANDOM_COLUMN.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(INPUT_WITH_RANDOM_COLUMN, index=False)
    return INPUT_WITH_RANDOM_COLUMN


def get_training_data():
    feature_cols = load_feature_columns()
    X, y = _build_training_frame_from_openml()
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    return X, y, train_idx, test_idx, feature_cols


def build_keras_classifier(n_features: int, n_classes: int, X_train):
    """Keras classifier with feature normalization."""
    from tensorflow import keras
    from keras import layers

    normalizer = layers.Normalization()
    normalizer.adapt(X_train)
    model = keras.Sequential(
        [
            layers.Input(shape=(n_features,)),
            normalizer,
            layers.Dense(KERAS_HIDDEN_UNITS, activation="relu"),
            layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def fit_keras_classifier(model, X_train, y_train) -> None:
    from tensorflow import keras

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
            verbose=0,
        )
    ]
    model.fit(
        X_train,
        y_train,
        epochs=KERAS_EPOCHS,
        batch_size=KERAS_BATCH_SIZE,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=0,
    )


def train_logistic_regression(X: pd.DataFrame, y: np.ndarray, train_idx) -> Pipeline:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]
    )
    model.fit(X.iloc[train_idx], y[train_idx])
    return model


def get_inference_row_indices() -> np.ndarray:
    """Map each canonical inference row to its index in the OpenML-encoded training frame."""
    feature_cols = load_feature_columns()
    X_all, _ = _build_training_frame_from_openml()
    inf = pd.read_csv(CANONICAL_INPUT)[feature_cols]
    indices = []
    for _, row in inf.iterrows():
        mask = X_all.eq(row, axis=1).all(axis=1)
        if not mask.any():
            raise RuntimeError("Inference row not found in encoded OpenML training frame.")
        indices.append(int(X_all.index[mask][0]))
    return np.array(indices)


def metrics_for_predict_fn(predict_fn, X, y, train_idx, test_idx) -> dict[str, float]:
    return {
        "train_accuracy": float(accuracy_score(y[train_idx], predict_fn(X.iloc[train_idx]))),
        "test_accuracy": float(accuracy_score(y[test_idx], predict_fn(X.iloc[test_idx]))),
    }


def save_extension_metrics(extension_metrics: dict[str, dict[str, float]], train_idx, test_idx) -> None:
    payload = {
        "train_indices": train_idx.tolist(),
        "test_indices": test_idx.tolist(),
        "random_state": RANDOM_STATE,
        "input_dataset": str(CANONICAL_INPUT.name),
        "serial_column": SERIAL_COLUMN_NAME,
        "empty_column": EMPTY_COLUMN_NAME,
        "extension_metrics": extension_metrics,
    }
    SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SPLIT_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def deploy_schema_json() -> None:
    for ext in EXTENSIONS:
        dest = ROOT / ext / "schema.json"
        if dest.resolve() == SCHEMA_PATH.resolve():
            continue
        shutil.copy2(SCHEMA_PATH, dest)


def deploy_input_csv() -> Path:
    if not CANONICAL_INPUT.exists():
        ensure_canonical_input()
    for ext in EXTENSIONS:
        dataset_dir = ROOT / ext / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "input.csv").write_bytes(CANONICAL_INPUT.read_bytes())
    return CANONICAL_INPUT


def deploy_all() -> Path:
    deploy_schema_json()
    ensure_canonical_input()
    ensure_input_with_random_column()
    return deploy_input_csv()
