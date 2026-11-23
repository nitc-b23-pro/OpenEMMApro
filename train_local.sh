#!/usr/bin/env bash
# train_local.sh
# One-click train + test entrypoint for a local GPU machine:
#   1. trains train_openemma.py on the full nuScenes trainval release
#      (v1.0-trainval, 850 scenes: official 700 train / 150 val split),
#   2. picks the saved checkpoint with the best validation ADE, and
#   3. tests it with main.py on the held-out nuScenes test release
#      (v1.0-test, 150 scenes), writing an overall summary.json.
#
# Every setting (paths, versions, test options) comes from config/local.env
# -- see config/local.env.example. This script has no built-in defaults.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -f config/local.env ]; then
  echo "config/local.env not found -- run ./setup_local.sh first (or copy" >&2
  echo "config/local.env.example to config/local.env and fill it in)." >&2
  exit 1
fi
set -a
# Strip Windows line endings so a file edited on Windows still works.
# shellcheck disable=SC1090
source <(tr -d '\r' < config/local.env)
set +a

missing=()
for var in OPENEMMA_CONDA_ENV OPENEMMA_QWEN_PATH \
           OPENEMMA_TRAIN_DATAROOT OPENEMMA_TRAIN_VERSION \
           OPENEMMA_TEST_DATAROOT OPENEMMA_TEST_VERSION \
           OPENEMMA_TEST_PLOT OPENEMMA_SKIP_TRAIN; do
  [ -n "${!var:-}" ] || missing+=("$var")
done
# OPENEMMA_TEST_CHECKPOINT must be present but may be empty (= auto-pick).
[ -n "${OPENEMMA_TEST_CHECKPOINT+x}" ] || missing+=("OPENEMMA_TEST_CHECKPOINT")
if [ "${#missing[@]}" -gt 0 ]; then
  echo "config/local.env is missing: ${missing[*]}" >&2
  echo "(see config/local.env.example for every required setting)" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$OPENEMMA_CONDA_ENV"

echo "OPENEMMA_QWEN_PATH=$OPENEMMA_QWEN_PATH"
echo "Train/val: $OPENEMMA_TRAIN_VERSION @ $OPENEMMA_TRAIN_DATAROOT"
echo "Test:      $OPENEMMA_TEST_VERSION @ $OPENEMMA_TEST_DATAROOT (plots: $OPENEMMA_TEST_PLOT)"
echo

# ---- 1. Train (train + val on trainval) ------------------------------------
if [ "$OPENEMMA_SKIP_TRAIN" != "1" ]; then
  # train_openemma.py appends to these logs if they already exist; move a
  # previous run's copies aside so step 2 only sees this run's epochs.
  backup=""
  for f in epoch_loss.json val_epoch_losses.json val_ade_epochs.json; do
    if [ -f "$f" ]; then
      if [ -z "$backup" ]; then
        backup="previous_run_$(date +%Y%m%d-%H%M%S)"
        mkdir -p "$backup"
      fi
      mv "$f" "$backup/"
    fi
  done
  [ -z "$backup" ] || echo "Moved the previous run's loss/ADE logs to $backup/"

  OPENEMMA_DATAROOT="$OPENEMMA_TRAIN_DATAROOT" OPENEMMA_VERSION="$OPENEMMA_TRAIN_VERSION" \
    python train_openemma.py "$@"
fi

# ---- 2. Pick the checkpoint to test -----------------------------------------
CKPT="$OPENEMMA_TEST_CHECKPOINT"
if [ -z "$CKPT" ]; then
  CKPT="$(python - <<'PY'
import json, os, sys
if not os.path.exists("val_ade_epochs.json"):
    sys.exit("val_ade_epochs.json not found -- train first, or set OPENEMMA_TEST_CHECKPOINT.")
with open("val_ade_epochs.json") as f:
    val_ade = json.load(f)
scored = [(v["avgade"], int(k.split("-")[1])) for k, v in val_ade.items() if v]
scored = [(ade, ep) for ade, ep in scored if os.path.isdir(f"openemma_epoch{ep}")]
if not scored:
    sys.exit("No epoch has both a validation ADE and a saved checkpoint -- set OPENEMMA_TEST_CHECKPOINT.")
best_ade, best_epoch = min(scored)
print(f"Best validation ADE: epoch {best_epoch} (avg ADE {best_ade:.3f} m)", file=sys.stderr)
print(f"openemma_epoch{best_epoch}")
PY
)"
fi
echo "Testing checkpoint: $CKPT"

# ---- 3. Test on the held-out test release -----------------------------------
python main.py --epoch "$CKPT" --dataroot "$OPENEMMA_TEST_DATAROOT" \
  --version "$OPENEMMA_TEST_VERSION" --plot "$OPENEMMA_TEST_PLOT"
