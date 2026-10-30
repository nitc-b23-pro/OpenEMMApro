#!/usr/bin/env python3
"""
scripts/prepare_nuscenes.py
Assembles a nuScenes dataset directory (v1.0-*/ , maps/, samples/,
sweeps/) that NuScenes(dataroot=..., version=...) can load, from pieces
you already have on disk. Generalizes the Kaggle notebook's
`ln -s /kaggle/input/... /kaggle/working/nuscenes_test_root/...` cells so
they work with any local layout, and also patches any map-mask PNG that
map.json references but is actually missing on disk (nuscenes-devkit's
NuScenes() asserts on these even though this project never renders map
masks -- the same fix the Kaggle notebook's patch_missing_map_masks cell
applied).

nuScenes itself can NOT be auto-downloaded by this script: you must
accept nuScenes' license at https://www.nuscenes.org/download first.
This script only arranges files after that step. Two supported sources:

  1. --kaggle-dataset <owner/slug>  e.g. aadimator/nuscenes-test
     Uses the Kaggle API (needs ~/.kaggle/kaggle.json -- see
     https://www.kaggle.com/docs/api#authentication) to download and
     extract a dataset laid out the same way this project's own
     development data was (Metadata/<version>, Metadata/maps,
     "Sensor file blobs"/samples, "Sensor file blobs"/sweeps).

  2. --metadata-dir <dir> --blobs-dir <dir>
     Point at folders you already downloaded/extracted yourself from
     nuscenes.org (or anywhere else) -- <metadata-dir>/<version>,
     <metadata-dir>/maps, <blobs-dir>/samples, <blobs-dir>/sweeps.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

from PIL import Image


def link_or_copy(src, dst, copy=False):
    if os.path.exists(dst) or os.path.islink(dst):
        print(f"  (skip, already exists) {dst}")
        return
    if not os.path.exists(src):
        print(f"  [warn] source missing, skipping: {src}")
        return
    src = os.path.abspath(src)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    if copy:
        (shutil.copytree if os.path.isdir(src) else shutil.copy2)(src, dst)
    else:
        os.symlink(src, dst)
    print(f"  {dst} -> {src}")


def patch_missing_map_masks(dataroot, version):
    map_json_path = os.path.join(dataroot, version, "map.json")
    if not os.path.exists(map_json_path):
        print(f"No map.json at {map_json_path}, skipping map-mask patch.")
        return

    with open(map_json_path) as f:
        map_records = json.load(f)

    patched = 0
    for rec in map_records:
        img_path = os.path.join(dataroot, rec["filename"])
        parent = os.path.dirname(img_path)

        # A broken symlink (or something that isn't a real directory)
        # at `parent` makes os.makedirs(..., exist_ok=True) still raise
        # FileExistsError -- clear it first.
        if os.path.islink(parent) or (os.path.exists(parent) and not os.path.isdir(parent)):
            os.remove(parent)
        os.makedirs(parent, exist_ok=True)

        if not os.path.exists(img_path):
            Image.new("L", (100, 100), color=0).save(img_path)
            patched += 1
            print(f"  created placeholder for missing map mask: {img_path}")

    print(f"Patched {patched} missing map mask(s) out of {len(map_records)}.")


def from_kaggle(dataset_slug, dest, version, copy):
    tmp = tempfile.mkdtemp(prefix="nuscenes_kaggle_")
    print(f"Downloading {dataset_slug} via the Kaggle API into {tmp} ...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", tmp, "--unzip"],
        check=True,
    )
    os.makedirs(dest, exist_ok=True)
    link_or_copy(os.path.join(tmp, "Metadata", version), os.path.join(dest, version), copy)
    link_or_copy(os.path.join(tmp, "Metadata", "maps"), os.path.join(dest, "maps"), copy)
    link_or_copy(os.path.join(tmp, "Sensor file blobs", "samples"), os.path.join(dest, "samples"), copy)
    link_or_copy(os.path.join(tmp, "Sensor file blobs", "sweeps"), os.path.join(dest, "sweeps"), copy)


def from_local_dirs(metadata_dir, blobs_dir, dest, version, copy):
    os.makedirs(dest, exist_ok=True)
    link_or_copy(os.path.join(metadata_dir, version), os.path.join(dest, version), copy)
    link_or_copy(os.path.join(metadata_dir, "maps"), os.path.join(dest, "maps"), copy)
    link_or_copy(os.path.join(blobs_dir, "samples"), os.path.join(dest, "samples"), copy)
    link_or_copy(os.path.join(blobs_dir, "sweeps"), os.path.join(dest, "sweeps"), copy)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default="data/nuscenes_root")
    ap.add_argument("--version", default="v1.0-test")
    ap.add_argument("--kaggle-dataset", default=None, help="e.g. aadimator/nuscenes-test")
    ap.add_argument("--metadata-dir", default=None)
    ap.add_argument("--blobs-dir", default=None)
    ap.add_argument("--copy", action="store_true",
                     help="Copy files instead of symlinking (use if --dest needs to "
                          "survive the source being removed, or symlinks aren't allowed).")
    args = ap.parse_args()

    if args.kaggle_dataset:
        from_kaggle(args.kaggle_dataset, args.dest, args.version, args.copy)
    elif args.metadata_dir and args.blobs_dir:
        from_local_dirs(args.metadata_dir, args.blobs_dir, args.dest, args.version, args.copy)
    else:
        print("Nothing to do: pass --kaggle-dataset OR both --metadata-dir and --blobs-dir.\n", file=sys.stderr)
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    patch_missing_map_masks(args.dest, args.version)

    abs_dest = os.path.abspath(args.dest)
    print(f"\nSet this in config/local.env:\n  OPENEMMA_DATAROOT={abs_dest}\n  OPENEMMA_VERSION={args.version}")


if __name__ == "__main__":
    main()
