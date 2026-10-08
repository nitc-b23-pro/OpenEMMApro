"""
eval_train_vs_test.py
Runs OpenEMMA-TinyVLA inference + ADE evaluation twice -- once on a sample
of scenes the model actually TRAINED on (nuScenes v1.0-test, the training
dataroot, restricted to the scenes train_openemma_tinyvla.py's scene-level
split put in the TRAIN set -- never the held-out validation scenes), and
once on scenes it has never seen (nuScenes v1.0-mini, or any other test
dataroot/version you point it at) -- then prints and plots a side-by-side
comparison of ADE@1s/2s/3s and failure rate between the two.

Why this exists
---------------
The per-epoch training/validation LOSS (lp_epoch_loss.json /
val_epoch_losses.json, plotted by train_openemma_tinyvla.py itself or by
plot_losses.py) tells you how the diffusion loss behaves on seen vs
held-out WINDOWS. This script answers a related but different question at
the level the project actually cares about -- trajectory error (ADE) -- and
makes the seen/unseen comparison explicit: if ADE on training scenes is
dramatically better than ADE on genuinely unseen scenes, that is a concrete
overfitting/memorization signal, the same way a large train/val loss gap
would be, but measured in meters of displacement error instead of loss
units.

This script does NOT retrain or re-run the scene split from scratch -- it
reproduces train_openemma_tinyvla.py's scene-level split (same
VAL_FRACTION, same SPLIT_SEED) purely from the scene NAMES in the training
nuScenes version, so it recovers exactly which scenes were in the TRAIN set
without needing the dataset's full OpenEMMADataset machinery (no need to
import train_openemma_tinyvla.py itself, which would re-run the whole
training script as a side effect of import -- it has no `if __name__ ==
"__main__":` guard).

--use-intent (default True) mirrors main.py's/train_openemma_tinyvla.py's
flag and MUST match how the checkpoint in --epoch was actually trained, or
this comparison conflates "seen vs unseen scenes" with "prompt the model
wasn't trained on" -- see main.py's run_inference_on_scenes docstring for
how the on-the-fly route intent is computed (no precomputed JSON needed
here either).

Usage
-----
    python eval_train_vs_test.py \\
        --epoch /kaggle/working/OpenEMMApro/openemma_tinyvla_epoch19/ \\
        --train-dataroot /kaggle/working/nuscenes_test_root/ --train-version v1.0-test \\
        --test-dataroot datasets/NuScenes --test-version v1.0-mini \\
        --n-train-scenes 5 --n-test-scenes 5
"""
import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from nuscenes import NuScenes

from main import load_model_and_processors, run_inference_on_scenes, BASE_PRETRAINED, TTL_LEN

# MUST mirror train_openemma_tinyvla.py's own constants (see that file's
# "ADDED (scene-level train/val split...)" comment) -- these two numbers are
# the only thing that needs to stay in sync for this script to recover the
# correct train_scenes set.
VAL_FRACTION = 0.1
SPLIT_SEED = 42


def recover_train_scene_names(dataroot, version):
    """
    Reproduces train_openemma_tinyvla.py's scene-level split using ONLY the
    scene names from nuScenes (no OpenEMMADataset / curvature computation
    needed for this) -- same sort -> shuffle(seed) -> first n_val held out
    logic, so the returned set is exactly the TRAIN side of that split for
    this dataroot/version.
    """
    nusc = NuScenes(version=version, dataroot=dataroot)
    scene_names = sorted(s["name"] for s in nusc.scene)
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(scene_names)
    n_val = max(1, round(len(scene_names) * VAL_FRACTION))
    val_scenes = set(scene_names[:n_val])
    train_scenes = set(scene_names[n_val:])
    return nusc, train_scenes, val_scenes


def pick_scenes(nusc, allowed_names, n, seed=SPLIT_SEED):
    """
    Deterministically samples up to n scenes from nusc.scene whose name is
    in allowed_names (or all of nusc.scene if allowed_names is None),
    skipping any scene shorter than TTL_LEN frames up front so --n-*-scenes
    isn't silently reduced later inside run_inference_on_scenes.
    """
    candidates = [s for s in nusc.scene if allowed_names is None or s["name"] in allowed_names]
    usable = []
    for s in candidates:
        tok, last = s["first_sample_token"], s["last_sample_token"]
        n_frames = 1
        while tok != last:
            sample = nusc.get("sample", tok)
            tok = sample["next"]
            n_frames += 1
        if n_frames >= TTL_LEN:
            usable.append(s)
    rng = random.Random(seed)
    rng.shuffle(usable)
    return usable[:n] if n is not None else usable


def summarize(results):
    """Aggregate mean ade1s/ade2s/ade3s/avgade/failure_rate across a list of
    per-scene result dicts (the shape run_inference_on_scenes returns)."""
    if not results:
        return {"ade1s": None, "ade2s": None, "ade3s": None, "avgade": None, "failure_rate": None, "n_scenes": 0}
    return {
        "ade1s": float(np.mean([r["ade1s"] for r in results])),
        "ade2s": float(np.mean([r["ade2s"] for r in results])),
        "ade3s": float(np.mean([r["ade3s"] for r in results])),
        "avgade": float(np.mean([r["avgade"] for r in results])),
        "failure_rate": float(np.mean([r["failure_rate"] for r in results])),
        "n_scenes": len(results),
    }


def plot_comparison(train_summary, test_summary, output_path):
    metrics = ["ade1s", "ade2s", "ade3s", "avgade"]
    labels = ["ADE@1s", "ADE@2s", "ADE@3s", "Avg ADE"]
    train_vals = [train_summary[m] for m in metrics]
    test_vals = [test_summary[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, train_vals, width, label=f"Train scenes (seen, n={train_summary['n_scenes']})")
    plt.bar(x + width / 2, test_vals, width, label=f"Test scenes (unseen, n={test_summary['n_scenes']})")
    plt.xticks(x, labels)
    plt.ylabel("ADE (m)")
    plt.title("ADE: training scenes (seen) vs. test scenes (unseen)")
    plt.legend()
    plt.grid(True, axis="y")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", type=str, default="lp")
    parser.add_argument("--epoch", type=str, required=True, help="trained checkpoint dir, e.g. openemma_tinyvla_epoch19/")
    parser.add_argument("--train-dataroot", type=str, default="/kaggle/working/nuscenes_test_root/")
    parser.add_argument("--train-version", type=str, default="v1.0-test")
    parser.add_argument("--test-dataroot", type=str, default="datasets/NuScenes")
    parser.add_argument("--test-version", type=str, default="v1.0-mini")
    parser.add_argument("--n-train-scenes", type=int, default=5, help="how many TRAIN-split scenes to evaluate on")
    parser.add_argument("--n-test-scenes", type=int, default=5, help="how many test scenes to evaluate on (None = all)")
    parser.add_argument("--plot", type=bool, default=True, help="write per-frame images/videos too (slower); the comparison bar chart is always written")
    parser.add_argument("--use-intent", dest="use_intent", action="store_true", default=True,
                         help="Must match how --epoch was trained (default: on) -- see this file's module docstring.")
    parser.add_argument("--no-intent", dest="use_intent", action="store_false")
    parser.add_argument("--lookahead-offset", type=int, default=TTL_LEN)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    timestamp_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir or f"{args.model_path}_results/train_vs_test/{timestamp_tag}"
    train_dir = os.path.join(output_dir, "train_seen")
    test_dir = os.path.join(output_dir, "test_unseen")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    print(f"Loading model from checkpoint: {args.epoch}")
    model, tokenizer, image_processor = load_model_and_processors(BASE_PRETRAINED, args.epoch)

    print(f"Recovering train/val scene split for {args.train_dataroot} ({args.train_version}) "
          f"using VAL_FRACTION={VAL_FRACTION}, SPLIT_SEED={SPLIT_SEED} (must match train_openemma_tinyvla.py)...")
    train_nusc, train_scene_names, val_scene_names = recover_train_scene_names(args.train_dataroot, args.train_version)
    print(f"{len(train_scene_names)} scenes were in TRAIN, {len(val_scene_names)} held out as VAL.")

    train_eval_scenes = pick_scenes(train_nusc, train_scene_names, args.n_train_scenes)
    print(f"Evaluating on {len(train_eval_scenes)} TRAIN-split scene(s): "
          f"{[s['name'] for s in train_eval_scenes]}")

    print(f"Loading test nuScenes: {args.test_dataroot} ({args.test_version})")
    test_nusc = NuScenes(version=args.test_version, dataroot=args.test_dataroot)
    test_eval_scenes = pick_scenes(test_nusc, None, args.n_test_scenes)
    print(f"Evaluating on {len(test_eval_scenes)} test scene(s): "
          f"{[s['name'] for s in test_eval_scenes]}")

    # run_inference_on_scenes reads args.plot / args.use_intent / args.lookahead_offset
    # directly off this same `args` namespace -- no separate copy needed.
    print("\n=== Running inference on TRAIN-split (seen) scenes ===")
    train_results = run_inference_on_scenes(train_nusc, train_eval_scenes, model, tokenizer, image_processor, train_dir, args)

    print("\n=== Running inference on TEST (unseen) scenes ===")
    test_results = run_inference_on_scenes(test_nusc, test_eval_scenes, model, tokenizer, image_processor, test_dir, args)

    train_summary = summarize(train_results)
    test_summary = summarize(test_results)

    comparison = {
        "checkpoint": args.epoch,
        "use_intent": args.use_intent,
        "train": {"dataroot": args.train_dataroot, "version": args.train_version, "summary": train_summary, "per_scene": train_results},
        "test": {"dataroot": args.test_dataroot, "version": args.test_version, "summary": test_summary, "per_scene": test_results},
    }
    comparison_path = os.path.join(output_dir, "train_vs_test_comparison.json")
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)

    print("\n=== SUMMARY ===")
    print(f"Train scenes (seen, n={train_summary['n_scenes']}): "
          f"ADE@1s={train_summary['ade1s']:.3f} ADE@2s={train_summary['ade2s']:.3f} "
          f"ADE@3s={train_summary['ade3s']:.3f} avgADE={train_summary['avgade']:.3f} "
          f"failure_rate={train_summary['failure_rate']:.1f}%")
    print(f"Test scenes (unseen, n={test_summary['n_scenes']}): "
          f"ADE@1s={test_summary['ade1s']:.3f} ADE@2s={test_summary['ade2s']:.3f} "
          f"ADE@3s={test_summary['ade3s']:.3f} avgADE={test_summary['avgade']:.3f} "
          f"failure_rate={test_summary['failure_rate']:.1f}%")
    if train_summary["avgade"] is not None and test_summary["avgade"] is not None:
        gap = test_summary["avgade"] - train_summary["avgade"]
        print(f"Gap (test avgADE - train avgADE): {gap:+.3f} m "
              f"({'larger error on unseen scenes -- expected generalization gap' if gap > 0 else 'unseen scenes did as well or better -- no obvious memorization signal'})")
    print(f"Wrote {comparison_path}")

    plot_comparison(train_summary, test_summary, os.path.join(output_dir, "train_vs_test_ade.jpg"))


if __name__ == "__main__":
    main()
