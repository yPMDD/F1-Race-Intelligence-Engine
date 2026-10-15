#!/usr/bin/env bash
# Install project dependencies, then force torch down to the CPU build.
#
# The problem: on Linux the `torch` wheel on PyPI is a CUDA build. It drags in
# ~3 GB of nvidia-* wheels, and this app never touches a GPU (all-MiniLM-L6-v2
# on CPU, XGBoost on CPU). Measured 3.59 GB -> 766 MB once handled.
#
# Why the obvious fixes do not work:
#
#   * UV_TORCH_BACKEND=cpu -- ignored for a locked install. uv.lock pins torch
#     to registry.pypi.org, and the lock wins over the backend hint.
#   * --no-install-package torch -- makes uv *remove* torch (verified: torch
#     disappears from the venv), and it does not stop the nvidia-* wheels,
#     which are separate lock entries listed as torch's dependencies.
#   * [tool.uv.sources] in pyproject.toml -- does not take effect here; a clean
#     `uv lock` still resolves torch from registry.pypi.org.
#
# So the order is: let uv sync do what it does, then remove the CUDA wheels and
# reinstall torch from the PyTorch CPU index. The redundant CUDA download costs
# a little build time; UV_NO_CACHE=1 (set in the Dockerfile) stops it leaving
# those bytes in the image layer.
set -euo pipefail

VENV_PY="${VENV_PY:-/app/.venv/bin/python}"
VENV_DIR="${VENV_DIR:-$(dirname "$(dirname "$VENV_PY")")}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"

echo "==> syncing project dependencies"
uv sync "$@"

echo "==> removing CUDA runtime packages"
nvidia_pkgs="$(uv pip list --python "$VENV_PY" 2>/dev/null \
  | awk '/^nvidia-/ {print $1}' | tr '\n' ' ' || true)"
if [ -n "${nvidia_pkgs// /}" ]; then
  echo "    removing: ${nvidia_pkgs}"
  # shellcheck disable=SC2086
  uv pip uninstall --python "$VENV_PY" ${nvidia_pkgs}
else
  echo "    none present"
fi

echo "==> forcing CPU torch ${TORCH_VERSION}"
uv pip install --python "$VENV_PY" --reinstall \
  --index-url "$TORCH_INDEX" "torch==${TORCH_VERSION}"

echo "==> verifying"
# Call the venv interpreter directly. Any bare `uv pip` invocation re-syncs
# from uv.lock and would undo all of the above.
torch_version="$("$VENV_PY" -c 'import torch; print(torch.__version__)' 2>/dev/null || true)"
if [ -z "$torch_version" ]; then
  echo "::error::torch is not importable from the venv" >&2
  exit 1
fi
echo "    torch ${torch_version}"
case "$torch_version" in
  *+cpu) ;;
  *)
    echo "::error::expected a CPU torch build, got ${torch_version}" >&2
    exit 1
    ;;
esac

remaining="$("$VENV_PY" -c '
import pathlib, sys
sp = pathlib.Path(sys.prefix) / "lib"
print(" ".join(sorted(p.name for p in sp.glob("python*/site-packages/nvidia*"))))
' 2>/dev/null || true)"
if [ -n "$remaining" ]; then
  echo "::error::CUDA packages survived: ${remaining}" >&2
  exit 1
fi

echo "==> done (CPU-only)"
echo "    NOTE: do not use 'uv run' in the container; it re-syncs from"
echo "    uv.lock and would reinstall the CUDA torch on every cold start."
