#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="python3.11"
VID2E_ENV="vid2e_env"
FILM_ENV="film_env"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu126"
TF_PACKAGE="tensorflow[and-cuda]==2.17.1"
CUDA_HOME_VALUE="/usr/local/cuda-12.8"
SET_CUDA_HOME=1
SETUP_VID2E=1
SETUP_FILM=1

usage() {
  cat <<'EOF'
Usage:
  bash scripts/setup_split_envs.sh [options]

Options:
  --python <bin>             Python executable for venv creation (default: python3.11)
  --vid2e-env <name>         Venv name for PyTorch/ESIM side (default: vid2e_env)
  --film-env <name>          Venv name for TensorFlow/FILM side (default: film_env)
  --torch-index-url <url>    PyTorch wheel index URL (default: cu126)
  --tf-package <spec>        TensorFlow package spec (default: tensorflow[and-cuda]==2.17.1)
  --cuda-home <path>         CUDA toolkit path for esim_torch build (default: /usr/local/cuda-12.8)
  --no-cuda-home             Do not export CUDA_HOME/PATH/LD_LIBRARY_PATH during build
  --skip-vid2e               Skip vid2e_env setup
  --skip-film                Skip film_env setup
  -h, --help                 Show this help

Examples:
  bash scripts/setup_split_envs.sh
  bash scripts/setup_split_envs.sh --cuda-home /usr/local/cuda-12.6
  bash scripts/setup_split_envs.sh --skip-film
EOF
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

log() {
  echo "[INFO] $*"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Command not found: $1"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      [[ $# -ge 2 ]] || die "--python requires a value"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --vid2e-env)
      [[ $# -ge 2 ]] || die "--vid2e-env requires a value"
      VID2E_ENV="$2"
      shift 2
      ;;
    --film-env)
      [[ $# -ge 2 ]] || die "--film-env requires a value"
      FILM_ENV="$2"
      shift 2
      ;;
    --torch-index-url)
      [[ $# -ge 2 ]] || die "--torch-index-url requires a value"
      TORCH_INDEX_URL="$2"
      shift 2
      ;;
    --tf-package)
      [[ $# -ge 2 ]] || die "--tf-package requires a value"
      TF_PACKAGE="$2"
      shift 2
      ;;
    --cuda-home)
      [[ $# -ge 2 ]] || die "--cuda-home requires a value"
      CUDA_HOME_VALUE="$2"
      shift 2
      ;;
    --no-cuda-home)
      SET_CUDA_HOME=0
      shift
      ;;
    --skip-vid2e)
      SETUP_VID2E=0
      shift
      ;;
    --skip-film)
      SETUP_FILM=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown option: $1"
      ;;
  esac
done

[[ "${SETUP_VID2E}" -eq 1 || "${SETUP_FILM}" -eq 1 ]] || die "Both setup targets are skipped."
require_cmd "${PYTHON_BIN}"

setup_vid2e_env() {
  log "Creating ${VID2E_ENV} with ${PYTHON_BIN}"
  "${PYTHON_BIN}" -m venv "${VID2E_ENV}"
  local py="${REPO_ROOT}/${VID2E_ENV}/bin/python"

  log "Installing base dependencies into ${VID2E_ENV}"
  "${py}" -m pip install -U pip setuptools wheel
  "${py}" -m pip install -r requirements.txt
  "${py}" -m pip install -r requirements-webapp.txt

  log "Installing PyTorch from: ${TORCH_INDEX_URL}"
  "${py}" -m pip uninstall -y torch torchvision torchaudio || true
  if [[ -n "${TORCH_INDEX_URL}" ]]; then
    "${py}" -m pip install --index-url "${TORCH_INDEX_URL}" torch torchvision torchaudio
  else
    "${py}" -m pip install torch torchvision torchaudio
  fi
  "${py}" -m pip install ninja

  log "Installing esim_torch editable package"
  if [[ "${SET_CUDA_HOME}" -eq 1 ]]; then
    if [[ -d "${CUDA_HOME_VALUE}" ]]; then
      log "Using CUDA_HOME=${CUDA_HOME_VALUE} for extension build"
      env CUDA_HOME="${CUDA_HOME_VALUE}" \
        PATH="${CUDA_HOME_VALUE}/bin:${PATH}" \
        LD_LIBRARY_PATH="${CUDA_HOME_VALUE}/lib64:${LD_LIBRARY_PATH:-}" \
        "${py}" -m pip install -e ./esim_torch --no-build-isolation
    else
      log "CUDA path ${CUDA_HOME_VALUE} not found. Building with current shell environment."
      "${py}" -m pip install -e ./esim_torch --no-build-isolation
    fi
  else
    "${py}" -m pip install -e ./esim_torch --no-build-isolation
  fi

  # log "Installing esim_py editable package"
  # "${py}" -m pip install -e ./esim_py --no-build-isolation

  log "Quick check: PyTorch in ${VID2E_ENV}"
  "${py}" - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("torch.cuda.is_available:", torch.cuda.is_available())
PY
}

setup_film_env() {
  log "Creating ${FILM_ENV} with ${PYTHON_BIN}"
  "${PYTHON_BIN}" -m venv "${FILM_ENV}"
  local py="${REPO_ROOT}/${FILM_ENV}/bin/python"

  log "Installing FILM dependencies into ${FILM_ENV}"
  "${py}" -m pip install -U pip setuptools wheel
  "${py}" -m pip install -r requirements-upsampling.txt
  "${py}" -m pip uninstall -y tensorflow tensorflow-cpu tensorflow-intel || true
  "${py}" -m pip install "${TF_PACKAGE}"

  log "Quick check: TensorFlow GPU in ${FILM_ENV}"
  "${py}" - <<'PY'
import tensorflow as tf
print("tf:", tf.__version__)
print("gpus:", tf.config.list_physical_devices("GPU"))
PY
}

if [[ "${SETUP_VID2E}" -eq 1 ]]; then
  setup_vid2e_env
fi

if [[ "${SETUP_FILM}" -eq 1 ]]; then
  setup_film_env
fi

cat <<EOF
[DONE] Environment setup complete.
- PyTorch env: ${VID2E_ENV}
- FILM env: ${FILM_ENV}

Run examples:
  source ${FILM_ENV}/bin/activate
  python upsampling/upsample.py --input_dir ./example/original --output_dir ./example/upsampled --device=0 --tf_log_level=3 --quiet
  deactivate

  source ${VID2E_ENV}/bin/activate
  python esim_torch/scripts/generate_events.py --input_dir=example/upsampled --output_dir=example/events --contrast_threshold_neg=0.2 --contrast_threshold_pos=0.2 --refractory_period_ns=0 --device=cuda
  deactivate
EOF
