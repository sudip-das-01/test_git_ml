# Multi-format model inference test

Self-contained folders for testing UI/backend inference flows across many model file formats.

## Quick start (run inference)

Each extension folder works the same way as `pkl/`. If `model/model.<ext>` and `dataset/input.csv` are already present, you only need inference:

```bash
cd pkl
python inference.py
```

Output is written to `output/output.csv` with a single `target` column.

Use the same steps in any other extension folder (`joblib`, `onnx`, `xgboost`, etc.) — only the folder name and model file extension change:

```bash
cd joblib
python inference.py
```

```bash
cd onnx
python inference.py
```

**First-time setup** for a folder (no model yet): train the artifact, then run inference:

```bash
cd pkl
python create_model.py
python inference.py
```

## Repository layout

```text
test_git_ml/
├── README.md
├── test_model_extensions.ipynb    # trains all models + runs inference + accuracy table
├── pkl/                           # reference layout + shared scripts/tests
│   ├── create_model.py
│   ├── inference.py
│   ├── inference_common.py
│   ├── schema.json
│   ├── dataset/
│   │   ├── input_with_empty_column.csv   # canonical input (copied to all extensions)
│   │   └── train_test_split.json
│   ├── model/
│   ├── output/
│   └── scripts/
│       ├── training_common.py
│       └── run_benchmark.py
├── joblib/
├── dill/
├── onnx/
├── pmml/
├── pickle/
├── xgboost/
├── lightgbm/
├── catboost/
├── h5/
├── pt/
├── pb/
└── zip/
```

Every extension folder contains:

| File / folder | Purpose |
|---------------|---------|
| `create_model.py` | Train and save `model/model.<ext>` |
| `inference.py` | Load model, read `dataset/input.csv`, write `output/output.csv` |
| `inference_common.py` | Schema-based feature selection (ignores extra columns) |
| `schema.json` | Feature column names used for inference |
| `dataset/input.csv` | Copy of canonical input CSV |
| `model/` | Serialized model artifact |
| `output/` | Inference results |
| `Dockerfile` | Runs `python inference.py` |
| `requirements.txt` | Python dependencies |

## Input dataset

Canonical source: `pkl/dataset/input_with_empty_column.csv` (built from the credit-application inference CSV).

- **Source file:** `pkl/dataset/credit_inference_source.csv` (copy of `credit_application_model 3/dataset/input.csv`), or the external path if present
- **48 feature columns** from `schema.json` (German credit one-hot features: `duration`, `credit_amount`, `status_a12`, …)
- Column `id`: serial number (`1` … `50`) replacing the original `unique_str` column
- Column `class`: **header present, all cells empty** (ignored at inference; only schema features are used)
- Deployed to each extension as `dataset/input.csv`

Training uses OpenML German Credit (1000 rows) encoded to the same 48 features; inference accuracy is measured on the 50-row deployment file.

Input must not include `target` or `prediction` columns.

## Model extensions

| Folder | Model file | Format |
|--------|------------|--------|
| `pkl` | `model/model.pkl` | Python pickle |
| `pickle` | `model/model.pickle` | Python pickle (`.pickle` ext) |
| `joblib` | `model/model.joblib` | Joblib |
| `dill` | `model/model.dill` | Dill |
| `onnx` | `model/model.onnx` | ONNX |
| `pmml` | `model/model.pmml` | PMML (requires Java for inference) |
| `xgboost` | `model/model.json` | XGBoost JSON |
| `lightgbm` | `model/model.txt` | LightGBM text |
| `catboost` | `model/model.cbm` | CatBoost |
| `h5` | `model/model.h5` | Keras HDF5 |
| `pt` | `model/model.pt` | PyTorch |
| `pb` | `model/` | TensorFlow SavedModel directory |
| `zip` | `model/model.zip` | MLflow sklearn bundle (zip) |

Training defaults in `pkl/scripts/training_common.py`: `N_ESTIMATORS=100`, `KERAS_EPOCHS=150` (with early stopping), `TORCH_EPOCHS=300`. Keras models (`h5`, `pb`) use a **feature normalization** layer; PyTorch (`pt`) uses `pt/model_net.py` with built-in standardization. Sklearn exports use a `StandardScaler` + `LogisticRegression` pipeline.

## Setup

From the repo root, using conda env `ostrichml` (or any env with dependencies from `pkl/requirements.txt`):

```bash
cd test_git_ml
conda run -n ostrichml pip install -r pkl/requirements.txt
```

For **PMML** inference (`pmml/`), a Java runtime is required (`pypmml`). Install OpenJDK if needed, e.g.:

```bash
conda install -n ostrichml -c conda-forge openjdk
```

## Run inference on all extensions

### Shell script (cd into each folder)

From the repo root:

```bash
./run_all_inference.sh
```

With a conda environment:

```bash
PYTHON="conda run -n ostrichml python" ./run_all_inference.sh
```

The script `cd`s into each extension directory and runs `python inference.py` (same as doing it manually in `pkl/`, `joblib/`, etc.). All **13** extensions should print `OK <ext> -> .../output/output.csv`.

**Latest run** (`PYTHON="conda run -n ostrichml python" ./run_all_inference.sh`): all extensions completed successfully (50 rows each).

### Python script (deploy input + inference + accuracy report)

Regenerates canonical input (`id`, features, empty `class`), copies to every `dataset/input.csv`, runs `python inference.py` in each folder:

```bash
conda run -n ostrichml python pkl/scripts/run_inference_all.py
```

Report: `pkl/output/inference_report.csv`

## Run all extensions (train + inference)

Builds canonical input, runs `create_model.py` and `inference.py` for every extension, writes `pkl/output/accuracy_report.csv`:

```bash
conda run -n ostrichml python pkl/scripts/run_benchmark.py
```

## Commands per extension

Run from the **repository root** (`test_git_ml/`), or `cd` into each folder first.

### PKL

```bash
cd pkl
conda run -n ostrichml python create_model.py
conda run -n ostrichml python inference.py
```

### Pickle (`.pickle`)

```bash
cd pickle
conda run -n ostrichml python create_model.py
conda run -n ostrichml python inference.py
```

### Joblib

```bash
cd joblib
conda run -n ostrichml python create_model.py
conda run -n ostrichml python inference.py
```

### Dill

```bash
cd dill
conda run -n ostrichml python create_model.py
conda run -n ostrichml python inference.py
```

### ONNX

```bash
cd onnx
conda run -n ostrichml python create_model.py
conda run -n ostrichml python inference.py
```

### PMML

```bash
cd pmml
conda run -n ostrichml python create_model.py
conda run -n ostrichml python inference.py
```

### XGBoost

```bash
cd xgboost
conda run -n ostrichml python create_model.py
conda run -n ostrichml python inference.py
```

### LightGBM

```bash
cd lightgbm
conda run -n ostrichml python create_model.py
conda run -n ostrichml python inference.py
```

### CatBoost

```bash
cd catboost
conda run -n ostrichml python create_model.py
conda run -n ostrichml python inference.py
```

### Keras H5

```bash
cd h5
conda run -n ostrichml python create_model.py
conda run -n ostrichml python inference.py
```

### PyTorch

```bash
cd pt
conda run -n ostrichml python create_model.py
conda run -n ostrichml python inference.py
```

### TensorFlow SavedModel (`pb`)

```bash
cd pb
conda run -n ostrichml python create_model.py
conda run -n ostrichml python inference.py
```

### MLflow ZIP

```bash
cd zip
conda run -n ostrichml python create_model.py
conda run -n ostrichml python inference.py
```

## Optional inference flags

All `inference.py` scripts support:

```bash
python inference.py \
  --data dataset/input.csv \
  --model model/model.pkl \
  --output output/output.csv \
  --schema schema.json
```

Defaults match each extension’s model path (e.g. `model/model.onnx` in `onnx/`).

## Docker

Build and run inference inside any extension folder (example: `pkl`):

```bash
cd pkl
docker build -t inference-pkl .
docker run --rm -v "$(pwd)/output:/app/output" inference-pkl
```

Each image runs `python inference.py` (model and `dataset/input.csv` must be present in the image context).

## Notebook

Root notebook trains all models, deploys input, runs inference, and shows accuracy:

```bash
jupyter notebook test_model_extensions.ipynb
```

## Tests

```bash
conda run -n ostrichml pytest pkl/tests/test_inference_extensions.py -v
```

## Latest inference results

After `./run_all_inference.sh` (or `run_inference_all.py`), see `pkl/output/inference_report.csv`.  
Dataset: **50 rows**, columns `id`, 48 credit features, empty `class`.

| Extension | Rows | Inference test acc. | Inference full acc. |
|-----------|------|---------------------|---------------------|
| lightgbm | 50 | 0.8800 | 0.8800 |
| h5 | 50 | 0.8800 | 0.8800 |
| pkl | 50 | 0.8600 | 0.8600 |
| joblib | 50 | 0.8600 | 0.8600 |
| dill | 50 | 0.8600 | 0.8600 |
| onnx | 50 | 0.8600 | 0.8600 |
| pmml | 50 | 0.8600 | 0.8600 |
| pickle | 50 | 0.8600 | 0.8600 |
| zip | 50 | 0.8600 | 0.8600 |
| catboost | 50 | 0.8600 | 0.8600 |
| pb | 50 | 0.8400 | 0.8400 |
| xgboost | 50 | 0.8000 | 0.8000 |
| pt | 50 | 0.8000 | 0.8000 |

*Inference test acc.* = accuracy on deployment rows that fall in the OpenML hold-out test split (see `pkl/dataset/train_test_split.json`).

## Latest train + benchmark results

After `run_benchmark.py`, see `pkl/output/accuracy_report.csv`. Example snapshot:

| Extension | Train acc. | Test acc. | Inference test acc. |
|-----------|------------|-----------|---------------------|
| pkl | 0.7900 | 0.7700 | 0.8600 |
| joblib | 0.7900 | 0.7700 | 0.8600 |
| dill | 0.7900 | 0.7700 | 0.8600 |
| onnx | 0.7900 | 0.7700 | 0.8600 |
| pmml | 0.7900 | 0.7700 | 0.8600 |
| pickle | 0.7900 | 0.7700 | 0.8600 |
| zip | 0.7900 | 0.7700 | 0.8600 |
| xgboost | — | 0.8000 | 0.8000 |
| lightgbm | — | 0.8800 | 0.8800 |
| catboost | — | 0.8600 | 0.8600 |
| h5 | — | 0.8200 | 0.8800 |
| pt | — | 0.8400 | 0.8000 |
| pb | — | 0.8400 | 0.8400 |

`inference_test_accuracy` is on the 50-row deployment CSV (rows that fall in the held-out OpenML test split). Retrain from repo root: `conda run -n ostrichml python pkl/scripts/run_benchmark.py`.
