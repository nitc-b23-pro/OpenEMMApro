#!/usr/bin/env bash
# setup_local.sh
# One-time environment setup for training OpenEMMApro on a local Linux
# GPU machine. Mirrors the Kaggle notebook's install cells (requirements.txt,
# then nuscenes-devkit --no-deps, then peft/diffusers/einops, then
# policy_heads, then bitsandbytes) but portable: a conda environment
# instead of Kaggle's preinstalled image, no Kaggle-specific paths, and
# safe to re-run. See docs/LOCAL_SETUP.md for the full walkthrough.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

ENV_NAME="${OPENEMMA_CONDA_ENV:-openemma}"
CUDA_TAG="${CUDA_TAG:-cu124}"   # nvidia-smi shows your driver's max supported CUDA

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found on PATH. Install Miniconda first:" >&2
  echo "  https://docs.conda.io/en/latest/miniconda.html" >&2
  exit 1
fi

# `conda activate` in a non-interactive script needs conda.sh sourced
# first. We do this (rather than `conda run -n ...` on every command)
# because `conda run`'s exit code doesn't always propagate reliably on
# every conda version -- a failed step under `conda run` could silently
# let `set -e` below do nothing, which is exactly what let a broken
# `pip install -e ./policy_heads` slip through unnoticed before.
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "==> Creating/updating conda env '$ENV_NAME' from environment.yml"
if conda env list | grep -qE "^\s*${ENV_NAME}\s"; then
  conda env update -n "$ENV_NAME" -f environment.yml
else
  conda env create -n "$ENV_NAME" -f environment.yml
fi

conda activate "$ENV_NAME"

echo "==> Installing PyTorch (CUDA build: $CUDA_TAG) into '$ENV_NAME'"
pip install torch torchvision torchaudio \
  --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

echo "==> Installing nuscenes-devkit==1.2.0 (--no-deps -- its own pins would "
echo "    downgrade numpy/matplotlib under what transformers/diffusers need; "
echo "    its real runtime deps are listed directly in environment.yml instead)"
pip install --no-deps nuscenes-devkit==1.2.0
python -c "import nuscenes" || {
  echo "ERROR: nuscenes-devkit installed but 'import nuscenes' still fails -- " >&2
  echo "see the error above (likely a missing dependency -- check environment.yml)." >&2
  exit 1
}

echo "==> Installing policy_heads (editable install of this repo's vendored "
echo "    diffusion policy head -- not a published package, can't live in environment.yml)"
pip install -e ./policy_heads
python -c "from policy_heads.models.droid_unet_diffusion import ConditionalUnet1D" || {
  echo "ERROR: 'pip install -e ./policy_heads' did not leave policy_heads importable." >&2
  echo "Re-run manually to see the real error:" >&2
  echo "  conda activate $ENV_NAME && pip install -e ./policy_heads" >&2
  exit 1
}

if [ ! -f config/local.env ]; then
  cp config/local.env.example config/local.env
  echo "==> Wrote config/local.env from the template -- you'll edit the paths in it next."
fi

echo
echo "==> Sanity check"
# An `if` condition is explicitly exempt from `set -e`, so a failing
# check here doesn't abort the script before we get to report it.
if python scripts/check_env.py; then
  CHECK_OK=1
else
  CHECK_OK=0
fi

cat <<EOM

Setup done. Next steps (see docs/LOCAL_SETUP.md for details):
  1. Get the dataset, e.g.:
       conda activate ${ENV_NAME} && python scripts/prepare_nuscenes.py --kaggle-dataset aadimator/nuscenes-test --version v1.0-test
  2. Get the Qwen2.5-VL-3B-Instruct weights:
       conda activate ${ENV_NAME} && python scripts/download_qwen_weights.py
  3. Edit config/local.env with the paths both scripts printed.
  4. ./train_local.sh
EOM

if [ "$CHECK_OK" -eq 0 ]; then
  echo
  echo "WARNING: scripts/check_env.py reported failures above -- fix those" >&2
  echo "before training (re-run it any time with: conda activate ${ENV_NAME} && python scripts/check_env.py)." >&2
  exit 1
fi
