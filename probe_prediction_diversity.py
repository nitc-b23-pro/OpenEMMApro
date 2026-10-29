"""
probe_prediction_diversity.py
Direct test of mode collapse, complementary to ablation_state_swap.py and
ablation_image_swap.py. Both of those showed ADE barely changes when either
input (history or image) is swapped for a wrong one -- consistent with the
diffusion head having collapsed to a near input-independent prediction, but
that's an *indirect* read: ADE could in principle stay flat under swapping
for other reasons.

This script checks the direct claim instead: are the model's raw predicted
trajectories actually close to IDENTICAL vectors across very different
frames (different images, different scenes, different observed histories),
with no ground truth or swapping involved at all? If so, that's airtight
evidence of collapse -- the model isn't varying its answer regardless of
what you feed it.

For each of N frames (same train-seen/test-unseen scenes the other scripts
use) it runs ONE real prediction (no swap) and compares:
  - predictions: the flattened (10, 2) predicted speed/curvature sequences
  - states:      the flattened (10, 2) observed history fed in (what the
                 diffusion head is conditioned on)
  - last_step:   just the most recently observed [speed, curvature*100]
                 (the "keep doing what you were doing" baseline)

For each of these three, it reports the coefficient of variation: the
average distance of each frame's vector from the group mean vector,
normalized by the group mean's norm. If predictions have a MUCH smaller
coefficient of variation than states/last_step (whose diversity is a
built-in property of the driving data, not the model), the model's output
genuinely isn't tracking its input -- confirming collapse directly, not
just inferring it from flat ADE under swapping.

Standalone: does not require retraining, only a checkpoint to evaluate.

Usage
-----
    python probe_prediction_diversity.py \\
        --epoch /kaggle/working/OpenEMMApro/openemma_epoch9/ \\
        --train-dataroot /kaggle/working/nuscenes_test_root/ --train-version v1.0-test \\
        --test-dataroot datasets/NuScenes --test-version v1.0-mini \\
        --n-train-scenes 5 --n-test-scenes 5
"""
import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from nuscenes import NuScenes

from ablation_state_swap import collect_scene_frames
from eval_train_vs_test import recover_train_scene_names, pick_scenes
from main import load_model_and_processor, predict_step, BASE_PRETRAINED, FUT_LEN


def flatten_obs(obs_velocities, obs_curvatures):
    speed = np.linalg.norm(obs_velocities, axis=1)
    curv = obs_curvatures * 100
    return np.stack([speed, curv], axis=1).flatten()


def collect_vectors(model, processor, group_name, frames):
    """Runs one real (unswapped) prediction per frame and collects the
    three comparison vectors described in the module docstring."""
    predictions, states, last_steps, scenes = [], [], [], []
    for i, f in enumerate(frames):
        pred, _ = predict_step(f["image_path"], f["obs_velocities"], f["obs_curvatures"], model, processor)
        pred = np.array(pred)[:FUT_LEN]
        predictions.append(pred.flatten())

        obs_flat = flatten_obs(f["obs_velocities"], f["obs_curvatures"])
        states.append(obs_flat)
        last_steps.append(obs_flat[-2:])
        scenes.append(f["scene"])

        print(f"[{group_name}] frame={i} scene={f['scene']} "
              f"pred[0]={pred[0].tolist()} pred[-1]={pred[-1].tolist()} "
              f"last_observed={obs_flat[-2:].tolist()}")

    return {
        "scenes": scenes,
        "predictions": np.array(predictions),
        "states": np.array(states),
        "last_steps": np.array(last_steps),
    }


def coefficient_of_variation(vectors):
    """Mean per-vector distance from the group mean, normalized by the
    group mean's norm. ~0 means every vector is nearly identical
    (collapsed); larger means genuine spread across frames."""
    if len(vectors) < 2:
        return None
    mean_vec = vectors.mean(axis=0)
    mean_norm = np.linalg.norm(mean_vec)
    dists = np.linalg.norm(vectors - mean_vec, axis=1)
    return {
        "mean_dist_from_mean": float(dists.mean()),
        "mean_vector_norm": float(mean_norm),
        "coefficient_of_variation": float(dists.mean() / mean_norm) if mean_norm > 1e-8 else float("inf"),
    }


def plot_comparison(cov_by_label, output_path):
    labels = list(cov_by_label.keys())
    values = [cov_by_label[l]["coefficient_of_variation"] if cov_by_label[l] else 0.0 for l in labels]

    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color=["#4C72B0", "#DD8452", "#55A868"][:len(labels)])
    plt.ylabel("Coefficient of variation (mean distance from group mean / mean norm)")
    plt.title("Prediction diversity vs. input diversity across frames")
    plt.grid(True, axis="y")
    for i, v in enumerate(values):
        plt.text(i, v, f"{v:.3f}", ha="center", va="bottom")
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
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    max_frames = args.max_frames_per_scene or None

    timestamp_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir or f"{args.model_path}_results/prediction_diversity/{timestamp_tag}"
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
    all_frames = train_frames + test_frames
    print(f"Collected {len(train_frames)} train-seen frames, {len(test_frames)} test-unseen frames, "
          f"{len(all_frames)} total (max_frames_per_scene={max_frames or 'all'})")

    print("\n=== Running ONE real prediction per frame (no swapping) ===")
    data = collect_vectors(model, processor, "all", all_frames)

    cov_predictions = coefficient_of_variation(data["predictions"])
    cov_states = coefficient_of_variation(data["states"])
    cov_last_step = coefficient_of_variation(data["last_steps"])

    summary = {
        "checkpoint": args.epoch,
        "n_frames": len(all_frames),
        "scenes": sorted(set(data["scenes"])),
        "predictions": cov_predictions,
        "states": cov_states,
        "last_step": cov_last_step,
    }
    summary_path = os.path.join(output_dir, "prediction_diversity_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n=== SUMMARY ===")
    print(f"n_frames={len(all_frames)}, scenes={summary['scenes']}")
    print(f"Predicted trajectories  : coefficient of variation = {cov_predictions['coefficient_of_variation']:.4f} "
          f"(mean |pred - mean_pred| = {cov_predictions['mean_dist_from_mean']:.3f}, "
          f"|mean_pred| = {cov_predictions['mean_vector_norm']:.3f})")
    print(f"Observed states (input) : coefficient of variation = {cov_states['coefficient_of_variation']:.4f} "
          f"(mean |state - mean_state| = {cov_states['mean_dist_from_mean']:.3f}, "
          f"|mean_state| = {cov_states['mean_vector_norm']:.3f})")
    print(f"Last observed step      : coefficient of variation = {cov_last_step['coefficient_of_variation']:.4f} "
          f"(mean |last - mean_last| = {cov_last_step['mean_dist_from_mean']:.3f}, "
          f"|mean_last| = {cov_last_step['mean_vector_norm']:.3f})")

    ratio = cov_predictions["coefficient_of_variation"] / cov_states["coefficient_of_variation"]
    print(f"\nPredictions vary {ratio * 100:.1f}% as much (relative to their own scale) as the states they're "
          f"supposedly conditioned on.")
    if ratio < 0.3:
        print("-> Predictions are far less diverse than the inputs: the model is producing close to the SAME "
              "trajectory regardless of frame. This directly confirms mode collapse -- not inferred from flat "
              "ADE under swapping, but measured straight from the raw outputs.")
    elif ratio > 0.7:
        print("-> Predictions vary about as much as the inputs do: the model IS producing meaningfully "
              "different outputs per frame. This would be surprising given the swap-ablation results and is "
              "worth double-checking (e.g. is the variation tracking the right thing, or just noisy?).")
    else:
        print("-> Predictions vary somewhat less than the inputs -- partial collapse, or the conditioning signal "
              "is weak but not entirely absent. Inspect prediction_diversity_summary.json and the printed "
              "per-frame predictions above before concluding.")

    print(f"Wrote {summary_path}")
    plot_comparison(
        {"Predictions": cov_predictions, "States (input)": cov_states, "Last observed step": cov_last_step},
        os.path.join(output_dir, "prediction_diversity.jpg"),
    )


if __name__ == "__main__":
    main()
