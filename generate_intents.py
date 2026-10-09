"""
generate_intents.py
Offline labeling pass: for every (image, past-10, future-10) sample that
OpenEMMADataset would produce for a given nuScenes dataroot/version,
compute a short "turn-by-turn navigator" intent phrase -- PURELY from the
ego vehicle's own recorded path geometry (curvature + speed over time).
No image, no VLM, no GPU, no scene/object reasoning of any kind. Results
are written to a single JSON file: {sample_id: intent_string}.

REPLACES the earlier Qwen2.5-VL-based version of this file. That version
showed the model the camera image plus the past/future motion numbers and
asked it to describe the maneuver -- which meant the resulting intent was
conditioned on whatever was happening in the SCENE (nearby vehicles,
pedestrians, etc.), not on the planned ROUTE. Per explicit direction: the
intent this script produces must be like a Google-Maps-style navigator --
"next left turn", "continue straight", "bear right" -- derived only from
the road geometry the vehicle's own path traces out, never from critical
objects or scene understanding. See route_intent.py for the full design
rationale (vocabulary, thresholds, sign convention, and -- importantly --
the discussion of WHERE to start the look-ahead relative to the
prediction window, which is the one real design decision left open here).

sample_id convention
---------------------
sample_id is exactly the string OpenEMMADataset puts on every sample
(f"{scene_name}__{window_index}"). This script does NOT re-walk scenes
itself -- it reuses openemma_dataset.iter_scene_motion(), the exact same
generator OpenEMMADataset.__init__ uses, so the per-scene speed/curvature
arrays and the window index `i` line up with what OpenEMMADataset will
look up later for the SAME dataroot/version. There is only one place
(iter_scene_motion) that knows how to walk a scene and compute
speed/curvature -- this script and the dataset class both just consume it.

Look-ahead offset -- read before using the output in training
-----------------------------------------------------------------
Each sample's intent is computed from route_intent_for_window(curv, speed,
start_idx=i + LOOKAHEAD_OFFSET, ...), where i is the sample's own window
start index (the same `i` used for obs=[i:i+OBS_LEN], fut=[i+OBS_LEN:i+TTL_LEN]).

    --lookahead-offset OBS_LEN   (= i + OBS_LEN)
        Look-ahead starts exactly where the prediction target (fut) does.
        The intent then coarsely summarizes the model's own answer for
        this window -- a weak but real leak.

    --lookahead-offset FUT_LEN+OBS_LEN, i.e. TTL_LEN   (DEFAULT)
        Look-ahead starts AFTER the prediction window ends. The intent
        describes the route beyond the immediate 5s being predicted --
        the same way a real turn-by-turn navigator knows the route further
        out than your next five seconds. This is the safer default and
        matches route_intent.py's own documented recommendation.

Change this deliberately with --lookahead-offset if you want the other
behavior; it is a single CLI flag, not a hidden default.

Usage
-----
    python generate_intents.py \\
        --dataroot data/nuscenes_mini --version v1.0-mini \\
        --output intents_v1.0-mini.json

For the real training-data pass on Kaggle, point --dataroot/--version at
the same v1.0-test root train_openemma_tinyvla.py uses, and name the
output to match INTENTS_PATH there (intents_v1.0-test.json by default):

    python generate_intents.py \\
        --dataroot /kaggle/working/nuscenes_test_root/ --version v1.0-test \\
        --output intents_v1.0-test.json

No GPU, no model weights, no resumability machinery needed -- this runs in
well under a second per scene (pure numpy arithmetic on arrays already in
memory), so unlike the old Qwen version there's no multi-hour run to
protect against interruption. The whole dataset's intents are computed in
one pass and written once.
"""
import argparse
import json

from openemma_dataset import iter_scene_motion, OBS_LEN, FUT_LEN, TTL_LEN
from route_intent import route_intent_for_window
from nuscenes import NuScenes


def build_intents(nusc, lookahead_offset, lookahead_steps, segment_steps, max_segments):
    """
    Walks every scene via iter_scene_motion (the same generator
    OpenEMMADataset uses) and returns {sample_id: intent_string} for every
    window OpenEMMADataset would itself produce for this nusc.
    """
    intents = {}
    n_scenes = 0
    for name, images, speed, curv, world, vel in iter_scene_motion(nusc):   # CHANGED: iter_scene_motion now also yields vel (unused here)
        n_scenes += 1
        for i in range(len(images) - TTL_LEN):
            sample_id = f"{name}__{i}"
            intents[sample_id] = route_intent_for_window(
                curv, speed,
                start_idx=i + lookahead_offset,
                lookahead_steps=lookahead_steps,
                segment_steps=segment_steps,
                max_segments=max_segments,
            )
    return intents, n_scenes


def atomic_write_json(data, path):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    import os
    os.replace(tmp_path, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataroot", type=str, default="data/nuscenes_mini")
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--output", type=str, default="intents_v1.0-mini.json")
    parser.add_argument("--lookahead-offset", type=int, default=TTL_LEN,
                         help=f"window-relative start index for the route look-ahead "
                              f"(default {TTL_LEN} = OBS_LEN+FUT_LEN, i.e. AFTER the "
                              f"prediction window ends -- see this file's module docstring; "
                              f"pass {OBS_LEN} to instead start where the prediction target begins)")
    parser.add_argument("--lookahead-steps", type=int, default=20,
                         help="how many future steps (0.5s each) the route intent looks across (default 20 = 10s)")
    parser.add_argument("--segment-steps", type=int, default=10,
                         help="steps per segment within the look-ahead window (default 10 = 5s)")
    parser.add_argument("--max-segments", type=int, default=2,
                         help="max number of combined atomic intents per phrase (default 2)")
    args = parser.parse_args()

    print(f"Loading nuScenes: dataroot={args.dataroot} version={args.version}")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot)

    print(f"Computing route intents (lookahead_offset={args.lookahead_offset}, "
          f"lookahead_steps={args.lookahead_steps}, segment_steps={args.segment_steps}, "
          f"max_segments={args.max_segments}) -- pure geometry, no image/VLM/GPU ...")
    intents, n_scenes = build_intents(
        nusc,
        lookahead_offset=args.lookahead_offset,
        lookahead_steps=args.lookahead_steps,
        segment_steps=args.segment_steps,
        max_segments=args.max_segments,
    )

    atomic_write_json(intents, args.output)
    print(f"Done. {n_scenes} scenes walked, {len(intents)} sample_ids labeled. "
          f"Wrote {args.output}")

    # Small sanity printout: vocabulary distribution, so a bad threshold
    # choice (e.g. everything landing on "continue straight") is obvious
    # immediately rather than silently baked into the dataset.
    from collections import Counter
    counts = Counter(intents.values())
    print("Intent phrase distribution (top 15):")
    for phrase, n in counts.most_common(15):
        print(f"  {n:6d}  {phrase}")


if __name__ == "__main__":
    main()
