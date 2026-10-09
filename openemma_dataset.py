"""
openemma_dataset.py
Turns your nuScenes scenes into a PyTorch Dataset of
(image_path, prompt_text, history[10,2], future[10,2]) samples.

CHANGED (prompt redesign): no more `intents_cache.json` / precomputed
intent text. There is no separate intent field any more -- the model
gets the image plus the raw motion-history numbers and must infer scene
understanding, object behavior, and intent itself (see prompts.py). The
`if key not in self.intents: continue` filter is gone too, since there
is no cache to be missing from -- every window with enough frames is now
used.

CHANGED (dataset split): `version` is no longer defaulted silently for
training -- train_openemma_tinyvla.py now passes version="v1.0-test"
explicitly. v1.0-test has no sample_annotation (object-box labels are
withheld for the leaderboard), but it DOES have ego_pose / sample_data /
calibrated_sensor, and this dataset's ground truth (`fut`, the future
speed/curvature) comes only from ego_pose -- never from
sample_annotation -- so v1.0-test is usable here even though it would
NOT be usable for anything that needs labeled object boxes.

ADDED (scene id + sample id, for validation split + intent lookup):
every sample dict now also carries `scene` (the nuScenes scene name) and
`sample_id` (f"{scene}__{i}", stable and unique across the whole
dataset). Neither changes what __getitem__ returns to the training loop
by default -- they exist so train_openemma_tinyvla.py can split by
WHOLE SCENE (not by individual frame) when carving out a validation set,
and so a precomputed intents-by-sample-id JSON (see generate_intents.py)
can be looked up per sample. Splitting by scene matters here because
consecutive frames in the same scene come from heavily overlapping
10-past/10-future windows -- a frame-level random split would leak
nearly-identical windows across train/val.

ADDED (intent, re-added -- see prompts.py's docstring for why this is
safe to do again): `intents_path`, optional. When given and the file
exists, it's loaded as a {sample_id: intent_string} JSON map (produced
by generate_intents.py) and each sample's intent is looked up and
threaded into build_prompt(). A sample with no entry in the map (or no
map at all) gets intent=None, which build_prompt() treats exactly like
before this change -- so this is backward compatible with every existing
caller that doesn't pass intents_path. (generate_intents.py's intents
are now a deterministic, map-free "turn-by-turn navigator" phrase --
see route_intent.py -- not a VLM-generated scene description, so there
is no image-model dependency anywhere in this file or its intent path.)

ADDED (iter_scene_motion, extracted): the scene-walking + curvature/speed
computation that used to live only inside __init__'s loop is now its own
generator function, reused by generate_intents.py so that script doesn't
re-implement (and risk drifting from) this exact logic. __init__'s own
behavior/output is unchanged by this refactor -- it's the same
computation, just factored out.

ADDED (fut_class, cur_pos/cur_vel/fut_world -- for training-time class
balancing and real-ADE tracking, NEITHER is a model input):
- fut_class: a coarse maneuver label (route_intent.label_for_delta on the
  heading change of THIS SAMPLE's own fut window, i.e. exactly the 10
  steps the diffusion head is trained to predict) -- "continue straight",
  "bear left"/"bear right", "left turn"/"right turn", or "u-turn". Used
  ONLY as a sampling-weight key by train_openemma_tinyvla.py's
  --balance-classes (WeightedRandomSampler) and by scene_diagnostics.py.
  It is never fed into the model, so computing it from the target window
  itself raises no leakage concern -- that's different from the
  route-navigation `intent` TEXT fed into the prompt, which is
  deliberately derived from BEYOND the target window for exactly that
  reason (see route_intent.py's docstring). Using the target to decide
  how often to SAMPLE an example is standard difficulty-based resampling,
  not a label leak.
- cur_pos / cur_vel: the vehicle's own world-frame (x, y) position and
  velocity at the last OBSERVED frame (i + OBS_LEN - 1) -- the same
  reference point main.py's predict_step reconstructs a predicted
  trajectory from (current position, plus heading derived from the last
  observed velocity vector).
- fut_world: the GT world-frame (x, y) positions for the FUT_LEN steps
  being predicted.
These three let train_openemma_tinyvla.py compute REAL ADE (in meters,
via the exact same IntegrateCurvatureForPoints reconstruction main.py
uses) directly from a validation sample, with no image/cv2/YOLO3D
dependency and no second pass over nuScenes -- see
train_openemma_tinyvla.py's run_validation_ade().
"""
import json
import os
from math import atan2
import numpy as np
import torch
from nuscenes import NuScenes
from utils import EstimateCurvatureFromTrajectory   # your existing file, unchanged
from prompts import build_prompt, OBS_LEN as _OBS_LEN, FUT_LEN as _FUT_LEN
from route_intent import segment_heading_change_deg, label_for_delta

OBS_LEN, FUT_LEN = _OBS_LEN, _FUT_LEN
TTL_LEN = OBS_LEN + FUT_LEN

DT = 0.5   # nuScenes keyframes are 2 Hz


def iter_scene_motion(nusc):
    """
    Walks every scene in `nusc` once, yielding
    (scene_name, image_paths, speed, curv, world, vel) for every scene with
    at least TTL_LEN frames. speed/curv/world/vel are FULL per-scene arrays
    (one entry per keyframe), not cropped to OBS_LEN/FUT_LEN -- a caller
    that needs more look-ahead than a single obs/fut window (e.g.
    generate_intents.py, which looks past the prediction window for a
    navigator-style intent) can slice these directly instead of
    recomputing curvature/speed itself.

    ADDED: `vel` (the raw per-frame world-frame velocity VECTOR, not just
    its norm) is now also yielded -- OpenEMMADataset needs it to recover
    the heading at a window's last observed frame (atan2(vel_y, vel_x)),
    the same way main.py's predict_step does, so training-time ADE
    tracking can reconstruct a trajectory without re-deriving this itself.
    Existing callers that only unpacked 5 values need a one-token update
    (generate_intents.py has already been updated).
    """
    for scene in nusc.scene:
        name = scene["name"]
        tok = scene["first_sample_token"]
        images, poses = [], []
        while True:
            sample = nusc.get("sample", tok)
            cam = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
            images.append(os.path.join(nusc.dataroot, cam["filename"]))
            poses.append(nusc.get("ego_pose", cam["ego_pose_token"]))
            if tok == scene["last_sample_token"]:
                break
            tok = sample["next"]

        if len(images) < TTL_LEN:
            continue

        world = np.array([p["translation"][:3] for p in poses])
        vel = np.zeros_like(world)
        vel[1:] = (world[1:] - world[:-1]) / DT
        vel[0] = vel[1]
        curv = EstimateCurvatureFromTrajectory(world)
        speed = np.linalg.norm(vel, axis=1)

        yield name, images, speed, curv, world, vel


class OpenEMMADataset(torch.utils.data.Dataset):
    def __init__(self, dataroot="datasets/NuScenes", version="v1.0-test", intents_path=None):
        self.nusc = NuScenes(version=version, dataroot=dataroot)

        self.samples = []   # each entry: dict(image_path, obs, fut, scene, sample_id, fut_class, cur_pos, cur_vel, fut_world)
        for name, images, speed, curv, world, vel in iter_scene_motion(self.nusc):
            for i in range(len(images) - TTL_LEN):
                obs = np.stack([speed[i:i+OBS_LEN], curv[i:i+OBS_LEN] * 100], axis=1)   # (10,2)
                fut = np.stack([speed[i+OBS_LEN:i+TTL_LEN],
                                 curv[i+OBS_LEN:i+TTL_LEN] * 100], axis=1)               # (10,2)

                # ADDED (fut_class): coarse maneuver label of THIS sample's
                # own prediction target, for sampling-weight use only (see
                # this file's module docstring) -- never fed to the model.
                fut_delta_deg = segment_heading_change_deg(curv[i+OBS_LEN:i+TTL_LEN], speed[i+OBS_LEN:i+TTL_LEN])
                fut_class = label_for_delta(fut_delta_deg)

                self.samples.append(dict(
                    image_path=images[i + OBS_LEN - 1],
                    obs=obs.astype(np.float32),
                    fut=fut.astype(np.float32),
                    scene=name,
                    sample_id=f"{name}__{i}",
                    fut_class=fut_class,
                    # ADDED: world-frame position/velocity/future-positions
                    # for training-time ADE tracking (meters, 2D x,y only --
                    # see module docstring).
                    cur_pos=world[i + OBS_LEN - 1, :2].astype(np.float32),
                    cur_vel=vel[i + OBS_LEN - 1, :2].astype(np.float32),
                    fut_world=world[i + OBS_LEN:i + TTL_LEN, :2].astype(np.float32),
                ))

        # ADDED: optional {sample_id: intent_string} lookup, produced offline
        # by generate_intents.py. Silently absent (self.intents = {}) if no
        # path is given or the file doesn't exist yet -- every __getitem__
        # lookup then just falls back to intent=None, identical to before
        # this feature existed.
        self.intents = {}
        if intents_path and os.path.exists(intents_path):
            with open(intents_path, "r") as f:
                self.intents = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        intent = self.intents.get(s["sample_id"])   # None if no map, or no entry for this sample
        return dict(
            image_path=s["image_path"],
            raw_lang=build_prompt(s["obs"], intent=intent),   # SAME prompt fn used at inference time
            state=torch.from_numpy(s["obs"].flatten()),       # (20,) -- the "history" numbers
            action=torch.from_numpy(s["fut"]),                 # (10,2) -- the correct-answer numbers (diffusion target)
            is_pad=torch.zeros(10, dtype=torch.bool),
            scene=s["scene"],
            sample_id=s["sample_id"],
            intent=intent if intent is not None else "",
            # ADDED (not model inputs -- see module docstring):
            fut_class=s["fut_class"],
            cur_pos=torch.from_numpy(s["cur_pos"]),
            cur_vel=torch.from_numpy(s["cur_vel"]),
            fut_world=torch.from_numpy(s["fut_world"]),
        )
