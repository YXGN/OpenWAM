#!/usr/bin/env bash
# Run from any directory; all Python packages and caches stay in this project.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || { echo "Install uv, then rerun this script." >&2; exit 1; }
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.cache/python"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD"
export DS_BUILD_OPS=0
if [[ -d "${OPENWAM_CUDA_HOME:-/usr/local/cuda-12.4}" ]]; then
    export CUDA_HOME="${OPENWAM_CUDA_HOME:-/usr/local/cuda-12.4}"
fi
openwam_index="${OPENWAM_PYPI_INDEX:-https://pypi.org/simple}"

uv python install 3.12.13
if [[ -e .venv/bin/python ]]; then
    .venv/bin/python - <<'PY'
import pathlib
import sys

root = pathlib.Path.cwd()
assert sys.version_info[:3] == (3, 12, 13), "Existing .venv has a different Python version."
assert pathlib.Path(sys.base_prefix).is_relative_to(root / ".cache/python"), (
    "Existing .venv uses an external Python; move it aside before setup."
)
PY
fi
uv venv --allow-existing --prompt openwam --python 3.12.13 --managed-python .venv
# DeepSpeed's metadata/build needs torch, packaging and ninja already available.
uv pip install --python .venv/bin/python --index-url "$openwam_index" \
    pip==26.1.2 setuptools==80.10.2 wheel==0.45.1 packaging==25.0 ninja==1.13.0
uv pip install --python .venv/bin/python --index-url "$openwam_index" --torch-backend cu126 \
    torch==2.7.1+cu126 torchvision==0.22.1+cu126
uv pip install --python .venv/bin/python --index-url "$openwam_index" --torch-backend cu126 \
    --no-build-isolation -r envs/requirements-cu126.lock
uv pip check --python .venv/bin/python
echo "Environment ready. Run: source scripts/activate_openwam.sh"
