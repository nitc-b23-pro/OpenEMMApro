"""
ablation_state_swap.py
Empirical test of the shortcut-learning diagnosis (see the project doc
diffusion-head-shortcut-learning-diagnosis.md): is ADE actually coming from
the image, or mostly from the raw motion-history `states` vector the
diffusion head is also conditioned on?

For each frame in the same 5 train-seen + 5 test-unseen scenes
eval_train_vs_test.py uses, this runs TWO predictions against the SAME
image and the SAME true future:
  - real:    the frame's own (correct) observed history
  - swapped: a DIFFERENT frame's observed history (a real, in-distribution
             history, just the wrong one for this image/future -- picked
             via a derangement, so no frame ever gets its own history back)

Both the raw `states` tensor AND the prompt's text restatement of history
change together when swapped (predict_step builds both from the same
obs_velocities/obs_curvatures), so this cleanly tests "what happens when
the model's belief about recent motion is wrong but the image is right."

If swapping states barely changes ADE, the model isn't leaning on states
(would undercut the diagnosis). If it makes ADE dramatically worse, that
confirms the model relies on states more than on the image -- exactly the
failure mode the diagnosis describes.

Standalone: does not require retraining, only a checkpoint to evaluate.

Usage
-----
    python ablation_state_swap.py \\
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

from eval_train_vs_test import recover_train_scene_names, pick_scenes
from main import load_model_and_processor, predict_step, BASE_PRETRAINED, OBS_LEN, FUT_LEN, TTL_LEN
from utils import EstimateCurvatureFromTrajectory, IntegrateCurvatureForPoints

DT = 0.5
ABLATION_SEED = 42   # which "wrong" history gets swapped into each frame


def collect_scene_frames(nusc, scene, max_frames_per_scene=None):
    """Rebuilds the per-frame obs/future arrays main.py's
    run_inference_on_scenes computes internally, without running
    inference -- one dict per usable frame, optionally subsampled."""
    first_sample_token, last_sample_token = scene["first_sample_token"], scene["last_sample_token"]
    front_camera_images, ego_poses = [], []
    tok = first_sample_token
    while True:
        sample = nusc.get("sample", tok)
        cam = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        front_camera_images.append(os.path.join(nusc.dataroot, cam["filename"]))
        ego_poses.append(nusc.get("ego_pose", cam["ego_pose_token"]))
        if tok == last_sample_token:
            break
        tok = sample["next"]

    scene_length = len(front_camera_images)
    if scene_length < TTL_LEN:
        return []

    ego_world = np.array([p["translation"][:3] for p in ego_poses])
    ego_velocities = np.zeros_like(ego_world)
    ego_velocities[1:] = (ego_world[1:] - ego_world[:-1]) / DT
    ego_velocities[0] = ego_velocities[1]
    ego_curvatures = EstimateCurvatureFromTrajectory(ego_world)

    all_i = list(range(scene_length - TTL_LEN))
    if max_frames_per_scene and len(all_i) > max_frames_per_scene:
        all_i = [int(x) for x in np.linspace(0, len(all_i) - 1, num=max_frames_per_scene, dtype=int)]

    frames = []
    for i in all_i:
        frames.append(dict(
            scene=scene["name"],
            image_path=front_camera_images[i + OBS_LEN - 1],
            obs_velocities=ego_velocities[i:i + OBS_LEN],
            obs_curvatures=ego_curvatures[i:i + OBS_LEN],
            current_position=ego_world[i + OBS_LEN - 1],
            fut_world=ego_world[i + OBS_LEN:i + TTL_LEN, :2],
        ))
    return frames


def derangement(n, seed):
    """A random permutation of range(n) with no fixed points (n >= 2) --
    guarantees every frame's swapped history comes from a DIFFERENT frame,
    never (by chance) its own."""
    if n < 2:
        return list(range(n))
    rng = random.Random(seed)
    idx = list(range(n))
    while True:
        rng.shuffle(idx)
        if all(idx[i] != i for i in range(n)):
            return idx


def ade_from_prediction(speed_curvature_pred, current_position, initial_heading, fut_world):
    """
    initial_heading/current_position are always the REAL frame's own values
    -- they anchor where/which-way the predicted trajectory starts from,
    which is a physical fact about this frame, not something the model
    predicts. Only the history fed INTO the model (obs_velocities/
    obs_curvatures passed to predict_step) differs between conditions.
    """
    pred_len = min(FUT_LEN, len(speed_curvature_pred))
    pred_speeds = np.array(speed_curvature_pred)[:, 0]
    pred_curvatures = np.array(speed_curvature_pred)[:, 1] / 100
    pred_traj = IntegrateCurvatureForPoints(pred_curvatures, pred_speeds, current_position, initial_heading, DT)
    err = np.linalg.norm(fut_world[:pred_len] - pred_traj[:pred_len], axis=1)
    return {
        "ade1s": float(np.mean(err[:min(pred_len, 2)])),
        "ade2s": float(np.mean(err[:min(pred_len, 4)])),
        "ade3s": float(np.mean(err[:min(pred_len, 6)])),
        "avgade": float(np.mean(err)),
    }


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
        swapped_pred, _ = predict_step(f["image_path"], other["obs_velocities"], other["obs_curvatures"], model, processor)
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
              f"(swapped history from {other['scene']})")

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
    plt.bar(x - width / 2, real_vals, width, label="Real history (own states)")
    plt.bar(x + width / 2, swapped_vals, width, label="Swapped history (another frame's states)")
    plt.xticks(x, labels)
    plt.ylabel("Avg ADE (m)")
    plt.title("State-swap ablation: does ADE depend on the history, not the image?")
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
    output_dir = args.output_dir or f"{args.model_path}_results/state_swap_ablation/{timestamp_tag}"
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
    with open(os.path.join(output_dir, "state_swap_results.jsonl"), "w") as fh:
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
    summary_path = os.path.join(output_dir, "state_swap_summary.json")
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
              f"{worse_frac:.0f}% of frames got worse under a swapped history.")
        if mean_gap > 0.5:
            print("-> Swapping in the wrong history made predictions meaningfully worse: the model is "
                  "leaning on `states` far more than on the image. Confirms the shortcut-learning diagnosis.")
        elif mean_gap < 0.1:
            print("-> Swapping history barely changed ADE: the model isn't relying much on `states` after "
                  "all. This would undercut the shortcut-learning diagnosis -- worth looking elsewhere.")
        else:
            print("-> Mixed/ambiguous result -- inspect state_swap_results.jsonl per scene before concluding.")

    print(f"Wrote {summary_path}")
    plot_comparison(train_results, test_results, os.path.join(output_dir, "state_swap_ablation.jpg"))


if __name__ == "__main__":
    main()
