#!/usr/bin/env bash
# train_local.sh
# One-click training entrypoint for a local GPU machine: loads
# config/local.env (the dataset + Qwen-weights paths setup_local.sh and
# scripts/*.py helped you fill in) and runs train_openemma.py -- the exact
# same training script Kaggle runs, which now reads those paths from the
# environment (OPENEMMA_QWEN_PATH / OPENEMMA_DATAROOT / OPENEMMA_VERSION)
# instead of hardcoded Kaggle paths, falling back to the Kaggle paths if
# they're unset so the Kaggle notebook keeps working unmodified.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

ENV_NAME="${OPENEMMA_CONDA_ENV:-openemma}"

if [ -f config/local.env ]; then
  set -a
  # shellcheck disable=SC1091
  source config/local.env
  set +a
else
  echo "config/local.env not found -- run ./setup_local.sh first (or copy" >&2
  echo "config/local.env.example to config/local.env and fill it in)." >&2
  exit 1
fi

echo "OPENEMMA_QWEN_PATH=$OPENEMMA_QWEN_PATH"
echo "OPENEMMA_DATAROOT=$OPENEMMA_DATAROOT"
echo "OPENEMMA_VERSION=$OPENEMMA_VERSION"
echo

conda run -n "$ENV_NAME" --no-capture-output python train_openemma.py "$@"
