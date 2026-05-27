#!/usr/bin/env bash
# Run python inference.py in every model extension folder.
# Usage (from repo root):
#   ./run_all_inference.sh
#   PYTHON=conda\ run\ -n\ ostrichml\ python ./run_all_inference.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
# Support multi-word commands, e.g. PYTHON="conda run -n ostrichml python"
read -r -a PYTHON_CMD <<< "${PYTHON}"

EXTENSIONS=()
while IFS= read -r ext; do
  [[ -n "${ext}" ]] && EXTENSIONS+=("${ext}")
done < <("${PYTHON_CMD[@]}" -c "
import sys
sys.path.insert(0, '${ROOT}/pkl/scripts')
from training_common import EXTENSIONS
for name in EXTENSIONS:
    print(name)
")

if [[ ${#EXTENSIONS[@]} -eq 0 ]]; then
  echo "ERROR: No extensions found (check PYTHON and training_common.EXTENSIONS)" >&2
  exit 1
fi

echo "Repository root: ${ROOT}"
echo "Python command: ${PYTHON}"
echo "Extensions (${#EXTENSIONS[@]}): ${EXTENSIONS[*]}"
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
    "${PYTHON_CMD[@]}" inference.py
  )
  echo "OK ${ext} -> ${dir}/output/output.csv"
  echo ""
done

echo "All extension inference runs completed."
