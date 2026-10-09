"""
scene_diagnostics.py
Cheap, zero-GPU diagnostic: for every scene in a nuScenes dataroot/version,
pulls nuScenes' own human-written scene `description` plus simple per-scene
motion statistics (mean/max |curvature|, mean speed, speed std) computed
the same way the rest of this project does
(openemma_dataset.iter_scene_motion), and optionally joins that against an
ADE results file to test whether harder (higher-curvature /
higher-speed-variance) scenes are in fact the ones with worse ADE -- the
hypothesis raised when diagnosing why some scenes were near-perfect
(sub-meter) and others 10-20+ m off by 3 seconds.

Accepts either main.py's ade_results.jsonl or eval_train_vs_test.py's
train_vs_test_comparison.json (reads both its train.per_scene and
test.per_scene lists, tagging each row with which split it came from).

Usage
-----
    # just dump descriptions + motion stats for every scene
    python scene_diagnostics.py --dataroot datasets/NuScenes --version v1.0-mini

    # join against a main.py run's ade_results.jsonl
    python scene_diagnostics.py --dataroot datasets/NuScenes --version v1.0-mini \\
        --ade-results lp_results/openemma/<timestamp>/ade_results.jsonl

    # join against eval_train_vs_test.py's comparison file -- run this once
    # per dataroot if train/test used different ones (the --dataroot/--version
    # here only control which scenes get descriptions/stats pulled; rows for
    # scenes not found in that dataroot/version are simply skipped)
    python scene_diagnostics.py --dataroot /kaggle/working/nuscenes_test_root/ --version v1.0-test \\
        --ade-results lp_results/train_vs_test/<timestamp>/train_vs_test_comparison.json --output train_scene_diagnostics.json
    python scene_diagnostics.py --dataroot <mini dataroot> --version v1.0-mini \\
        --ade-results lp_results/train_vs_test/<timestamp>/train_vs_test_comparison.json --output test_scene_diagnostics.json
"""
import argparse
import json

import numpy as np
from nuscenes import NuScenes

from openemma_dataset import iter_scene_motion


def load_ade_results(path):
    """
    Returns {scene_name: {..per-scene ADE fields.., "split": "train"|"test"|None}}.
    Accepts a main.py-style .jsonl (one json object per line, split=None) or
    an eval_train_vs_test.py train_vs_test_comparison.json (reads both its
    train.per_scene and test.per_scene lists).
    """
    results = {}
    if path.endswith(".jsonl"):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                results[r["name"]] = {**r, "split": None}
    else:
        with open(path) as f:
            data = json.load(f)
        for split in ("train", "test"):
            if split in data:
                for r in data[split].get("per_scene", []):
                    results[r["name"]] = {**r, "split": split}
    return results


def compute_scene_stats(nusc):
    """{scene_name: {"mean_abs_curv", "max_abs_curv", "mean_speed", "speed_std", "n_frames"}}"""
    stats = {}
    for name, images, speed, curv, world, vel in iter_scene_motion(nusc):
        stats[name] = {
            "mean_abs_curv": float(np.mean(np.abs(curv))),
            "max_abs_curv": float(np.max(np.abs(curv))),
            "mean_speed": float(np.mean(speed)),
            "speed_std": float(np.std(speed)),
            "n_frames": len(images),
        }
    return stats


def pearson(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataroot", type=str, default="datasets/NuScenes")
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--ade-results", type=str, default=None,
                         help="main.py's ade_results.jsonl, or eval_train_vs_test.py's train_vs_test_comparison.json")
    parser.add_argument("--output", type=str, default="scene_diagnostics.json")
    args = parser.parse_args()

    print(f"Loading nuScenes: dataroot={args.dataroot} version={args.version}")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot)
    descriptions = {s["name"]: s["description"] for s in nusc.scene}

    print("Computing per-scene motion statistics (mean/max |curvature|, speed mean/std)...")
    stats = compute_scene_stats(nusc)

    ade_by_scene = load_ade_results(args.ade_results) if args.ade_results else {}

    rows = []
    for name, stat in stats.items():
        row = {"name": name, "description": descriptions.get(name, ""), **stat}
        if name in ade_by_scene:
            row.update({k: v for k, v in ade_by_scene[name].items() if k not in ("name", "token")})
        rows.append(row)

    # Only scenes that actually have an ADE number are interesting to rank/correlate.
    scored = [r for r in rows if "avgade" in r]
    scored.sort(key=lambda r: r["avgade"], reverse=True)

    if scored:
        print(f"\n{'scene':<14}{'split':<7}{'avgADE':>9}  {'mean|curv|':>11}{'max|curv|':>11}{'mean_spd':>10}{'spd_std':>9}  description")
        for r in scored:
            print(f"{r['name']:<14}{str(r.get('split') or '-'):<7}{r['avgade']:>9.2f}  "
                  f"{r['mean_abs_curv']:>11.4f}{r['max_abs_curv']:>11.4f}"
                  f"{r['mean_speed']:>10.2f}{r['speed_std']:>9.2f}  {r['description']}")
    else:
        print("\nNo scenes in this dataroot/version matched the --ade-results file (or none was given) "
              "-- writing descriptions/motion stats only.")

    if len(scored) >= 3:
        print("\nCorrelation of avgADE with per-scene motion statistics (Pearson r, small-N -- read as a trend, not proof):")
        for key in ("mean_abs_curv", "max_abs_curv", "mean_speed", "speed_std"):
            r = pearson([s["avgade"] for s in scored], [s[key] for s in scored])
            print(f"  avgADE vs {key}: r={r:.3f}" if r is not None else f"  avgADE vs {key}: not enough variation to compute")
    elif scored:
        print("\nFewer than 3 ADE-scored scenes -- skipping correlation (not meaningful at this sample size).")

    with open(args.output, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote {args.output} ({len(rows)} scenes total, {len(scored)} with ADE scores)")


if __name__ == "__main__":
    main()
