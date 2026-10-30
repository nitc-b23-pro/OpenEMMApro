#!/usr/bin/env python3
"""
scripts/download_qwen_weights.py
Gets a local, OpenEMMApro-compatible copy of Qwen2.5-VL-3B-Instruct ready
to point OPENEMMA_QWEN_PATH at. Generalizes the Kaggle notebook's two
cells that symlinked the Kaggle-input copy into /kaggle/working and
patched its preprocessor_config.json -- here you can either download the
weights fresh from Hugging Face Hub (fully automated, it's a public
model) or reuse an existing local copy via --source.

Why the patch: the raw checkpoint's preprocessor_config.json reports
image_processor_type/processor_class values that this project's pinned
transformers/qwen_vl_utils versions don't recognize -- without the patch,
AutoProcessor.from_pretrained(...) fails or silently misbehaves.
"""
import argparse
import json
import os

HF_REPO = "Qwen/Qwen2.5-VL-3B-Instruct"


def patch_preprocessor_config(weights_dir):
    cfg_path = os.path.join(weights_dir, "preprocessor_config.json")
    if not os.path.exists(cfg_path):
        print(f"[warn] {cfg_path} not found -- nothing to patch (did the download finish?).")
        return

    # If this landed as a symlink to a read-only source (the --source
    # path), remove the link before writing so we never touch the
    # original files -- same precaution as the Kaggle notebook's cell.
    is_link = os.path.islink(cfg_path)
    with open(cfg_path) as f:
        cfg = json.load(f)

    changed = False
    if cfg.get("image_processor_type") != "Qwen2VLImageProcessor":
        cfg["image_processor_type"] = "Qwen2VLImageProcessor"
        changed = True
    if cfg.get("processor_class") != "Qwen2_5_VLProcessor":
        cfg["processor_class"] = "Qwen2_5_VLProcessor"
        changed = True

    if not changed:
        print(f"{cfg_path} already compatible, no patch needed.")
        return

    if is_link:
        os.remove(cfg_path)  # removes the symlink only, never the source file
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Patched {cfg_path} (image_processor_type -> Qwen2VLImageProcessor, "
          f"processor_class -> Qwen2_5_VLProcessor).")


def link_from_source(source, dest):
    src = os.path.abspath(source)
    os.makedirs(dest, exist_ok=True)
    for fname in os.listdir(src):
        d = os.path.join(dest, fname)
        if not os.path.exists(d) and not os.path.islink(d):
            os.symlink(os.path.join(src, fname), d)
    print(f"Linked {src} -> {dest}")


def download_from_hub(dest):
    from huggingface_hub import snapshot_download
    os.makedirs(dest, exist_ok=True)
    print(f"Downloading {HF_REPO} from Hugging Face Hub into {dest} ... "
          f"(several GB -- this can take a while)")
    snapshot_download(repo_id=HF_REPO, local_dir=dest)
    print("Download complete.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default="models/qwen2.5-vl-3b-fixed",
                     help="Local directory the (patched) weights end up in.")
    ap.add_argument("--source", default=None,
                     help="Reuse an existing local Qwen2.5-VL-3B-Instruct directory "
                          "(e.g. a shared lab copy) instead of downloading.")
    args = ap.parse_args()

    if args.source:
        link_from_source(args.source, args.dest)
    else:
        download_from_hub(args.dest)

    patch_preprocessor_config(args.dest)

    abs_dest = os.path.abspath(args.dest)
    print(f"\nSet this in config/local.env:\n  OPENEMMA_QWEN_PATH={abs_dest}")


if __name__ == "__main__":
    main()
