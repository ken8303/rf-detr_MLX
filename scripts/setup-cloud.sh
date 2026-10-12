#!/usr/bin/env bash
set -euo pipefail
cd /workspace/rf-detr_MLX
export PIP_CACHE_DIR=/workspace/.cache/pip
python3 -m venv /workspace/.venvs/rfdetr-mlx
/workspace/.venvs/rfdetr-mlx/bin/python -m pip install -r requirements-linux-dev.txt
/workspace/.venvs/rfdetr-mlx/bin/python -m pip install -e '.[cpu,dev]'
/workspace/.venvs/rfdetr-mlx/bin/python -m pip check
/workspace/.venvs/rfdetr-mlx/bin/python -m pytest -q
/workspace/.venvs/rfdetr-mlx/bin/rfdetr-mlx smoke --steps 5 --output artifacts/smoke
