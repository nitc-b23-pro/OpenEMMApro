# Training OpenEMMApro on your own GPU machine (Linux)

This repo was developed on Kaggle, which pre-installs CUDA/PyTorch and
provides scratch disk at `/kaggle/working`. This guide replaces all of
that Kaggle-specific plumbing with a portable conda environment and
configurable paths, so you can train on your own Linux machine with one
command at the end.

## 0. Prerequisites

- A Linux machine with an NVIDIA GPU and a working driver (`nvidia-smi`
  should print your GPU). The model trains with 4-bit QLoRA + an 8-bit
  optimizer specifically so it fits a ~15 GB Kaggle GPU (T4/P100-class);
  that same setup runs fine on a bigger local GPU too, it just leaves more
  headroom.
- [Miniconda or Anaconda](https://docs.conda.io/en/latest/miniconda.html).
- `git`.
- (Optional, for the fastest dataset path below) a Kaggle account with an
  API token at `~/.kaggle/kaggle.json` -- see
  https://www.kaggle.com/docs/api#authentication.

## 1. Clone and set up the environment

```bash
git clone -b kaggle-openvlm https://github.com/nitc-b23-pro/OpenEMMApro.git
cd OpenEMMApro
./setup_local.sh
```

This creates a conda environment named `openemma` from `environment.yml`,
installs the CUDA build of PyTorch matching `CUDA_TAG`, installs
`nuscenes-devkit` and this repo's vendored `policy_heads` package (each
needs its own install step -- see "Why two install stages" below), writes
`config/local.env` from the template, and runs a sanity check
(`scripts/check_env.py`).

Check `nvidia-smi` for your driver's max supported CUDA version first if
you need something other than the default:

```bash
CUDA_TAG=cu121 ./setup_local.sh
```

## 2. Get the dataset (nuScenes)

nuScenes requires agreeing to its license at
https://www.nuscenes.org/download -- that step genuinely can't be
automated. Two paths from there, both handled by one script:

**A. Fastest, if you have a Kaggle account** (this mirrors exactly what
this project's own Kaggle development used):

```bash
conda run -n openemma python scripts/prepare_nuscenes.py \
    --kaggle-dataset aadimator/nuscenes-test --version v1.0-test
```

**B. You already downloaded nuScenes archives yourself from
nuscenes.org:** extract them so you have a metadata folder (containing
`v1.0-*/` and `maps/`) and a blobs folder (containing `samples/` and
`sweeps/`), then:

```bash
conda run -n openemma python scripts/prepare_nuscenes.py \
    --metadata-dir /path/to/metadata --blobs-dir /path/to/blobs \
    --version v1.0-trainval
```

Either way, the script prints the `OPENEMMA_DATAROOT` / `OPENEMMA_VERSION`
values to paste into `config/local.env`, and patches any map-mask PNG
that nuScenes' `map.json` references but is actually missing on disk (a
nuscenes-devkit quirk that otherwise crashes `NuScenes(...)` even though
this project never renders map masks).

## 3. Get the Qwen2.5-VL-3B-Instruct weights

Fully automated -- this is a public Hugging Face model, no license wall:

```bash
conda run -n openemma python scripts/download_qwen_weights.py
```

This downloads the weights into `models/qwen2.5-vl-3b-fixed/` and patches
`preprocessor_config.json` the same way the Kaggle notebook's cells did
(the raw checkpoint's `image_processor_type`/`processor_class` fields
aren't recognized by this project's pinned `transformers`/`qwen_vl_utils`
versions otherwise, and `AutoProcessor.from_pretrained(...)` fails or
misbehaves without the patch). If you already have the weights somewhere
(a shared lab copy, say), point at them instead of re-downloading:

```bash
conda run -n openemma python scripts/download_qwen_weights.py \
    --source /path/to/existing/Qwen2.5-VL-3B-Instruct
```

## 4. Edit config/local.env

Open `config/local.env` and fill in `OPENEMMA_QWEN_PATH`,
`OPENEMMA_DATAROOT`, `OPENEMMA_VERSION` with the absolute paths the two
scripts above printed.

## 5. Train

```bash
./train_local.sh
```

That's the "one click" step: it's exactly `train_openemma.py` -- same
QLoRA + diffusion head, same hyperparameters, same action-space
normalization -- just reading its paths from `config/local.env` instead
of hardcoded Kaggle paths. Checkpoints (`openemma_epoch<N>/`) and the
loss/ADE JSON logs land in the repo root, same as on Kaggle. Any extra
arguments are passed straight through, e.g. to point at a different env:

```bash
OPENEMMA_CONDA_ENV=myenv ./train_local.sh
```

## Why two install stages (and no exact PyTorch pin in environment.yml)

- `nuscenes-devkit==1.2.0` is installed with `--no-deps`: its own pinned
  dependencies are old enough to pull `numpy`/`matplotlib` back down below
  what `transformers==4.51.3` and the diffusion head need. Installing it
  `--no-deps` after everything else is already-installed keeps both halves
  happy -- this is the same ordering the Kaggle notebook relied on by
  running its pip cells in a specific order; `setup_local.sh` just makes
  that order explicit and reproducible instead of implicit in cell order.
- `policy_heads` (the vendored `ConditionalUnet1D` diffusion head) is this
  repo's own code, installed editable (`pip install -e ./policy_heads`),
  so it can't be listed in `environment.yml` as a normal dependency.
- PyTorch isn't version-pinned in `environment.yml` because the right
  CUDA build depends on your GPU driver, not on this project --
  `setup_local.sh` installs it in its own step via
  `--index-url .../whl/$CUDA_TAG`, so you can override `CUDA_TAG` per
  machine without editing a file that's tracked in git.

## What's no longer needed outside Kaggle

- The Kaggle notebook's first cell manually adds CUDA/cuDNN library paths
  to `LD_LIBRARY_PATH` -- a workaround for Kaggle's specific image layout.
  A conda environment with pip-installed CUDA wheels (what `setup_local.sh`
  sets up) doesn't need it. If you hit a `cudnn`/`libcublas` "undefined
  symbol" or "cannot open shared object file" error anyway, run
  `scripts/check_env.py` first -- it prints `torch.version.cuda` and your
  driver info together, which is usually enough to spot a mismatched
  `CUDA_TAG`.
- The `/kaggle/input` -> `/kaggle/working` symlink cells are replaced by
  `scripts/prepare_nuscenes.py` and `scripts/download_qwen_weights.py`,
  which work with any local directory layout, not just Kaggle's.

## Troubleshooting

- **`torch.cuda.is_available()` is False** -- almost always a `CUDA_TAG`
  mismatch. Run `nvidia-smi`, note the "CUDA Version" in the top-right,
  and re-run `CUDA_TAG=cu1xx ./setup_local.sh` with the closest tag at or
  below that version.
- **OOM during training** -- this project is tuned for ~15 GB of VRAM
  already (4-bit QLoRA, 8-bit optimizer, gradient checkpointing, a fixed
  visual-token budget). If you're on something smaller, there isn't much
  more headroom to give up; if you're on something bigger and still OOM,
  check nothing else is using the GPU (`nvidia-smi`).
- **`NuScenes()` raises an `AssertionError` about a missing map mask** --
  re-run `scripts/prepare_nuscenes.py` with the same arguments; the
  map-mask patch step runs every time and is idempotent.
- **`[FAIL] diffusers   cannot import name 'cached_download' from 'huggingface_hub'`** --
  `diffusers==0.27.2` imports a function Hugging Face removed from
  `huggingface_hub` in 0.26.0, and `diffusers`' own PyPI metadata doesn't
  declare an upper bound that would stop pip from installing a too-new
  one. `transformers==4.51.3` separately requires `huggingface-hub>=0.30.0`,
  so pinning `huggingface_hub` down to keep `diffusers==0.27.2` happy is a
  dead end -- the two constraints directly conflict (pip will say so:
  "ResolutionImpossible"). `environment.yml` instead pins
  `diffusers==0.29.0`, the first release that dropped `cached_download`
  (verified: no reference to it in that wheel, and the exact
  `DDIMScheduler` kwargs `build_model.py` uses are all still accepted).
  If you still see this, your env predates that pin: `conda activate
  openemma && pip install "diffusers==0.29.0"`.
- **`[FAIL] nuscenes   No module named 'sklearn'`** (or `scipy`, `shapely`,
  `pycocotools`) -- `nuscenes-devkit` is installed with `--no-deps` (see
  "Why two install stages" above), so its real runtime dependencies have
  to be satisfied by `environment.yml` directly; they're all listed there
  now. If you still see this, your env predates that fix: `conda activate
  openemma && pip install scikit-learn scipy "Shapely~=2.0.3" "pycocotools>=2.0.1" "numpy<2.0.0,>=1.22.0"`.
- **`[FAIL] policy_heads not importable`, but `pip install -e ./policy_heads`
  printed no error** -- this happened because `setup_local.sh` used to run
  every install step through `conda run -n openemma ...`, and `conda run`
  doesn't always propagate a failing exit code reliably across conda
  versions, so a broken install could silently fall through to the next
  step instead of stopping the script. `setup_local.sh` now activates the
  env directly (`conda activate`) and explicitly verifies this import
  right after installing it, so a real failure now stops the script with
  the actual pip error visible. If you hit this with the current
  `setup_local.sh`, the printed error is the real one to act on -- most
  often a missing C/C++ toolchain (`sudo apt install build-essential`) or
  running it outside the `openemma` env.
- **`setup_local.sh` seems to be taking far longer than a previous run,
  with no code changes in between** -- almost always pip's dependency
  resolver backtracking through candidate versions of an *unpinned*
  package (the only pinned ones are `transformers`, `peft`, `diffusers`,
  `huggingface_hub`; everything else floats to "latest"), because
  something upstream published a new release between runs. This isn't a
  sign anything is broken, just slow -- `du -sh ~/.cache/pip` ticking up
  every few seconds means it's still actively working, not stuck. Letting
  it finish once is enough; later runs reuse the pip cache and skip
  already-satisfied packages, so they're fast again.
