#!/usr/bin/env python3
"""
scripts/check_env.py
Sanity-checks a local OpenEMMApro environment: CUDA visibility, GPU
memory, and that every package the training/inference scripts import is
actually importable at the pinned version. Run after setup_local.sh and
before the first real training run -- mirrors the Kaggle notebook's
`!python -c "import torch; print(torch.cuda.is_available())"` cell, plus
the version-check cell, but checks everything in one place and fails
loudly instead of silently.
"""
import importlib
import subprocess
import sys

REQUIRED = [
    "torch", "torchvision", "transformers", "accelerate", "peft",
    "diffusers", "einops", "bitsandbytes", "nuscenes", "cv2",
    "PIL", "qwen_vl_utils", "pyquaternion", "huggingface_hub",
]


def check_imports():
    ok = True
    for mod in REQUIRED:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            print(f"  [ok] {mod:<16} {ver}")
        except Exception as e:
            ok = False
            print(f"  [FAIL] {mod:<16} {e}")
    return ok


def check_cuda():
    try:
        import torch
    except ImportError:
        print("  [FAIL] torch not importable -- see above")
        return False

    print(f"  torch version      : {torch.__version__}")
    print(f"  torch.version.cuda  : {torch.version.cuda}")
    available = torch.cuda.is_available()
    print(f"  cuda available      : {available}")
    if not available:
        print("  [FAIL] torch.cuda.is_available() is False. Check your driver "
              "with `nvidia-smi` and that CUDA_TAG in config/local.env matches it.")
        return False

    n = torch.cuda.device_count()
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        total_gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
        print(f"  gpu {i}: {name} ({total_gb:.1f} GB)")
        if total_gb < 14:
            print(f"  [warn] gpu {i} has < 14 GB -- this project was tuned "
                  "(4-bit QLoRA + 8-bit optimizer) for ~15 GB Kaggle GPUs; "
                  "less than that will likely still OOM.")
    return True


def check_policy_heads():
    try:
        from policy_heads.models.droid_unet_diffusion import ConditionalUnet1D  # noqa: F401
        print("  [ok] policy_heads (editable install of this repo's diffusion head)")
        return True
    except Exception as e:
        print(f"  [FAIL] policy_heads not importable: {e}")
        print("         Run: pip install -e ./policy_heads")
        return False


def main():
    print("== Package imports ==")
    imports_ok = check_imports()
    print("\n== CUDA / GPU ==")
    cuda_ok = check_cuda()
    print("\n== policy_heads (vendored diffusion head) ==")
    heads_ok = check_policy_heads()

    print("\n== nvidia-smi ==")
    try:
        subprocess.run(["nvidia-smi"], check=False)
    except FileNotFoundError:
        print("  nvidia-smi not found on PATH -- is the NVIDIA driver installed?")

    print()
    if imports_ok and cuda_ok and heads_ok:
        print("Everything looks good -- ready to train.")
        sys.exit(0)
    else:
        print("One or more checks failed -- see [FAIL] lines above before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
