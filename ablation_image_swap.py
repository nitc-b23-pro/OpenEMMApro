"""
ablation_image_swap.py
Complementary test to ablation_state_swap.py: that script found swapping a
frame's *history* (states + the text restatement of it) barely changes ADE,
which undercuts the "model leans on states" diagnosis rather than confirming
it. The remaining open question is whether the model is using the *image*
either -- or whether it has collapsed toward a near input-independent
prediction that just happens to track each scene's inherent difficulty.

For each frame in the same 5 train-seen + 5 test-unseen scenes
eval_train_vs_test.py / ablation_state_swap.py use, this runs TWO
predictions against the SAME true future and the SAME (real) observed
history:
  - real:    the frame's own (correct) image
  - swapped: a DIFFERENT frame's image (a real, in-distribution image, just
             the wrong one for this history/future -- picked via a
             derangement, so no frame ever gets its own image back)

`states` and the text prompt are identical in both conditions; only the
image passed to the VLM differs. This isolates the image pathway the same
way ablation_state_swap.py isolated the history pathway.

How to read the result alongside ablation_state_swap.py's:
  - both swaps barely change ADE            -> mode collapse: the model
    isn't really conditioning on either input, it's predicting something
    close to a fixed/average motion and the per-scene pattern we keep
    seeing everywhere just reflects how far each scene's true trajectory
    is from that fixed prediction.
  - this swap changes ADE a lot, states-swap didn't -> the model is
    genuinely using the image and the earlier result was the correct
    read: states isn't the problem, nothing further to fix there.
  - both swaps change ADE a lot -> the model uses both signals in some
    combination; revisit which one dominates by comparing magnitudes.

Standalone: does not require retraining, only a checkpoint to evaluate.

Usage
-----
    python ablation_image_swap.py \\
        --epoch /kaggle/working/OpenEMMApro/openemma_epoch9/ \\
        --train-dataroot /kaggle/working/nuscenes_test_root/ --train-version v1.0-test \\
        --test-dataroot datasets/NuScenes --test-version v1.0-mini \\
        --n-train-scenes 5 --n-test-scenes 5
"""
import argparse
import json
import os
import random
from datetime import datetime
from math import atan2

import matplotlib.pyplot as plt
import numpy as np
from nuscenes import NuScenes

from ablation_state_swap import collect_scene_frames, derangement, ade_from_prediction
from eval_train_vs_test import recover_train_scene_names, pick_scenes
from main import load_model_and_processor, predict_step, BASE_PRETRAINED

ABLATION_SEED = 42   # which "wrong" image gets swapped into each frame


def run_group(model, processor, group_name, frames, seed=ABLATION_SEED):
    if not frames:
        print(f"[{group_name}] no usable frames, skipping")
        return []

    swap_idx = derangement(len(frames), seed)
    results = []

    for i, f in enumerate(frames):
        heading = atan2(f["obs_velocities"][-1][1], f["obs_velocities"][-1][0])

        real_pred, _ = predict_step(f["image_path"], f["obs_velocities"], f["obs_curvatures"], model, processor)
        real_ade = ade_from_prediction(real_pred, f["current_position"], heading, f["fut_world"])

        other = frames[swap_idx[i]]
        # Only the image changes -- obs_velocities/obs_curvatures (and
        # therefore states + the text prompt) stay the real frame's own,
        # mirroring ablation_state_swap.py's swap but on the other input.
        swapped_pred, _ = predict_step(other["image_path"], f["obs_velocities"], f["obs_curvatures"], model, processor)
        swapped_ade = ade_from_prediction(swapped_pred, f["current_position"], heading, f["fut_world"])

        row = {
            "group": group_name, "scene": f["scene"], "frame_idx": i,
            "swapped_from_scene": other["scene"],
            "real": real_ade, "swapped": swapped_ade,
            "delta_avgade": swapped_ade["avgade"] - real_ade["avgade"],
        }
        results.append(row)
        print(f"[{group_name}] scene={f['scene']} frame={i} "
              f"real_avgade={real_ade['avgade']:.3f} swapped_avgade={swapped_ade['avgade']:.3f} "
              f"(swapped image from {other['scene']})")

    return results


def summarize(results, key):
    vals = [r[key]["avgade"] for r in results]
    return {
        "ade1s": float(np.mean([r[key]["ade1s"] for r in results])),
        "ade2s": float(np.mean([r[key]["ade2s"] for r in results])),
        "ade3s": float(np.mean([r[key]["ade3s"] for r in results])),
        "avgade": float(np.mean(vals)),
        "n_frames": len(results),
    }


def plot_comparison(train_results, test_results, output_path):
    groups = [("Train (seen)", train_results), ("Test (unseen)", test_results)]
    labels, real_vals, swapped_vals = [], [], []
    for name, results in groups:
        if not results:
            continue
        labels.append(name)
        real_vals.append(summarize(results, "real")["avgade"])
        swapped_vals.append(summarize(results, "swapped")["avgade"])

    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(8, 6))
    plt.bar(x - width / 2, real_vals, width, label="Real image (own frame)")
    plt.bar(x + width / 2, swapped_vals, width, label="Swapped image (another frame's image)")
    plt.xticks(x, labels)
    plt.ylabel("Avg ADE (m)")
    plt.title("Image-swap ablation: does ADE depend on the image, not just the history?")
    plt.legend()
    plt.grid(True, axis="y")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", type=str, default="qwen")
    parser.add_argument("--epoch", type=str, required=True)
    parser.add_argument("--train-dataroot", type=str, default="/kaggle/working/nuscenes_test_root/")
    parser.add_argument("--train-version", type=str, default="v1.0-test")
    parser.add_argument("--test-dataroot", type=str, default="datasets/NuScenes")
    parser.add_argument("--test-version", type=str, default="v1.0-mini")
    parser.add_argument("--n-train-scenes", type=int, default=5)
    parser.add_argument("--n-test-scenes", type=int, default=5)
    parser.add_argument("--max-frames-per-scene", type=int, default=10,
                         help="Evenly-sampled frame cap per scene, to keep this cheap. 0 = use every frame.")
    parser.add_argument("--seed", type=int, default=ABLATION_SEED)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    max_frames = args.max_frames_per_scene or None

    timestamp_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir or f"{args.model_path}_results/image_swap_ablation/{timestamp_tag}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model from checkpoint: {args.epoch}")
    model, processor = load_model_and_processor(BASE_PRETRAINED, args.epoch)

    print(f"Recovering train/val scene split for {args.train_dataroot} ({args.train_version})...")
    train_nusc, train_scene_names, _ = recover_train_scene_names(args.train_dataroot, args.train_version)
    train_scenes = pick_scenes(train_nusc, train_scene_names, args.n_train_scenes)
    print(f"Using {len(train_scenes)} TRAIN-split scene(s): {[s['name'] for s in train_scenes]}")

    print(f"Loading test nuScenes: {args.test_dataroot} ({args.test_version})")
    test_nusc = NuScenes(version=args.test_version, dataroot=args.test_dataroot)
    test_scenes = pick_scenes(test_nusc, None, args.n_test_scenes)
    print(f"Using {len(test_scenes)} test scene(s): {[s['name'] for s in test_scenes]}")

    train_frames = [f for s in train_scenes for f in collect_scene_frames(train_nusc, s, max_frames)]
    test_frames = [f for s in test_scenes for f in collect_scene_frames(test_nusc, s, max_frames)]
    print(f"Collected {len(train_frames)} train-seen frames, {len(test_frames)} test-unseen frames "
          f"(max_frames_per_scene={max_frames or 'all'})")

    print("\n=== TRAIN (seen) scenes ===")
    train_results = run_group(model, processor, "train_seen", train_frames, seed=args.seed)
    print("\n=== TEST (unseen) scenes ===")
    test_results = run_group(model, processor, "test_unseen", test_frames, seed=args.seed)

    all_results = train_results + test_results
    with open(os.path.join(output_dir, "image_swap_results.jsonl"), "w") as fh:
        for row in all_results:
            fh.write(json.dumps(row) + "\n")

    summary = {
        "checkpoint": args.epoch,
        "max_frames_per_scene": max_frames,
        "seed": args.seed,
        "train_seen": {
            "real": summarize(train_results, "real") if train_results else None,
            "swapped": summarize(train_results, "swapped") if train_results else None,
        },
        "test_unseen": {
            "real": summarize(test_results, "real") if test_results else None,
            "swapped": summarize(test_results, "swapped") if test_results else None,
        },
    }
    summary_path = os.path.join(output_dir, "image_swap_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n=== SUMMARY ===")
    for group_name, results in [("Train (seen)", train_results), ("Test (unseen)", test_results)]:
        if not results:
            print(f"{group_name}: no frames")
            continue
        real = summarize(results, "real")
        swapped = summarize(results, "swapped")
        gap = swapped["avgade"] - real["avgade"]
        gap_pct = (gap / real["avgade"] * 100) if real["avgade"] else float("nan")
        print(f"{group_name} (n={real['n_frames']} frames): "
              f"real avgADE={real['avgade']:.3f}  swapped avgADE={swapped['avgade']:.3f}  "
              f"delta={gap:+.3f} ({gap_pct:+.1f}%)")

    all_gap = [r["delta_avgade"] for r in all_results]
    if all_gap:
        mean_gap = float(np.mean(all_gap))
        worse_frac = float(np.mean([d > 0 for d in all_gap])) * 100
        print(f"\nOverall: mean delta (swapped - real) = {mean_gap:+.3f} m, "
              f"{worse_frac:.0f}% of frames got worse under a swapped image.")
        if mean_gap > 0.5:
            print("-> Swapping in the wrong image made predictions meaningfully worse: the model IS "
                  "using the image. Combined with the state-swap ablation showing states barely "
                  "matters, this points at the image pathway as the real driver -- not mode collapse.")
        elif mean_gap < 0.1:
            print("-> Swapping the image barely changed ADE either. Combined with the state-swap "
                  "ablation also showing little change, this points at mode collapse: the model "
                  "isn't meaningfully conditioning on EITHER input, and the per-scene pattern "
                  "reflects how far each scene's true trajectory is from a near-constant prediction, "
                  "not the model reacting to its inputs.")
        else:
            print("-> Mixed/ambiguous result -- inspect image_swap_results.jsonl per scene before concluding.")

    print(f"Wrote {summary_path}")
    plot_comparison(train_results, test_results, os.path.join(output_dir, "image_swap_ablation.jpg"))


if __name__ == "__main__":
    main()
