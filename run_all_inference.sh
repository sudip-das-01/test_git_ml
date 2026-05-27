#!/usr/bin/env bash
# Run python inference.py in every model extension folder.
# Usage (from repo root):
#   ./run_all_inference.sh
#   PYTHON=conda\ run\ -n\ ostrichml\ python ./run_all_inference.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

EXTENSIONS=(
  pkl
  joblib
  dill
  onnx
  pmml
  pickle
  xgboost
  lightgbm
  catboost
  h5
  pt
  pb
  zip
)

echo "Repository root: ${ROOT}"
echo "Python command: ${PYTHON}"
echo ""

for ext in "${EXTENSIONS[@]}"; do
  dir="${ROOT}/${ext}"
  if [[ ! -d "${dir}" ]]; then
    echo "SKIP ${ext}: directory not found (${dir})"
    continue
  fi
  if [[ ! -f "${dir}/inference.py" ]]; then
    echo "SKIP ${ext}: inference.py not found"
    continue
  fi

  echo "=== ${ext} ==="
  (
    cd "${dir}"
    ${PYTHON} inference.py
  )
  echo "OK ${ext} -> ${dir}/output/output.csv"
  echo ""
done

echo "All extension inference runs completed."
