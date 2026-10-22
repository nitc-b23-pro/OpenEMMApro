"""
eval_train_vs_test_qwen.py
Same train-scenes-vs-unseen-scenes ADE comparison as eval_train_vs_test.py,
against a Qwen2.5-VL-3B-Instruct checkpoint (main_qwen.py) instead of the
LLaVA-Pythia one. Branch: kaggle-openvlm.

Only real difference from eval_train_vs_test.py: imports from main_qwen
instead of main, and load_model_and_processors()/run_inference_on_scenes()
here take (model, processor) instead of (model, tokenizer, image_processor)
-- see main_qwen.py's module docstring for why. Everything else (scene-
split recovery, scene sampling, ADE summarization, the comparison plot/
JSON) is identical and copied unchanged.

Usage
-----
    python eval_train_vs_test_qwen.py \\
        --epoch /kaggle/working/OpenEMMApro/openemma_qwen_epoch19/ \\
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

VAL_FRACTION = 0.1
SPLIT_SEED = 42


def recover_train_scene_names(dataroot, version):
    nusc = NuScenes(version=version, dataroot=dataroot)
    scene_names = sorted(s["name"] for s in nusc.scene)
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(scene_names)
    n_val = max(1, round(len(scene_names) * VAL_FRACTION))
    val_scenes = set(scene_names[:n_val])
    train_scenes = set(scene_names[n_val:])
    return nusc, train_scenes, val_scenes


def pick_scenes(nusc, allowed_names, n, seed=SPLIT_SEED):
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
    plt.title("OpenEMMA-Qwen: ADE on training scenes (seen) vs. test scenes (unseen)")
    plt.legend()
    plt.grid(True, axis="y")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", type=str, default="qwen")
    parser.add_argument("--epoch", type=str, required=True, help="trained checkpoint dir, e.g. openemma_qwen_epoch19/")
    parser.add_argument("--train-dataroot", type=str, default="/kaggle/working/nuscenes_test_root/")
    parser.add_argument("--train-version", type=str, default="v1.0-test")
    parser.add_argument("--test-dataroot", type=str, default="datasets/NuScenes")
    parser.add_argument("--test-version", type=str, default="v1.0-mini")
    parser.add_argument("--n-train-scenes", type=int, default=5)
    parser.add_argument("--n-test-scenes", type=int, default=5)
    parser.add_argument("--plot", type=bool, default=True)
    parser.add_argument("--use-intent", dest="use_intent", action="store_true", default=True)
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
    model, processor = load_model_and_processors(BASE_PRETRAINED, args.epoch)

    print(f"Recovering train/val scene split for {args.train_dataroot} ({args.train_version}) "
          f"using VAL_FRACTION={VAL_FRACTION}, SPLIT_SEED={SPLIT_SEED} (must match train_openemma_qwen.py)...")
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

    print("\n=== Running inference on TRAIN-split (seen) scenes ===")
    train_results = run_inference_on_scenes(train_nusc, train_eval_scenes, model, processor, train_dir, args)

    print("\n=== Running inference on TEST (unseen) scenes ===")
    test_results = run_inference_on_scenes(test_nusc, test_eval_scenes, model, processor, test_dir, args)

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
