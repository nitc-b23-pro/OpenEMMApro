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
"""
import os
from math import atan2
import numpy as np
import torch
from nuscenes import NuScenes
from utils import EstimateCurvatureFromTrajectory   # your existing file, unchanged
from prompts import build_prompt, OBS_LEN as _OBS_LEN, FUT_LEN as _FUT_LEN

OBS_LEN, FUT_LEN = _OBS_LEN, _FUT_LEN
TTL_LEN = OBS_LEN + FUT_LEN


class OpenEMMADataset(torch.utils.data.Dataset):
    def __init__(self, dataroot="datasets/NuScenes", version="v1.0-test"):
        self.nusc = NuScenes(version=version, dataroot=dataroot)

        self.samples = []   # each entry: dict(image_path=..., obs=..., fut=...)
        for scene in self.nusc.scene:
            name = scene["name"]
            tok = scene["first_sample_token"]
            images, poses = [], []
            while True:
                sample = self.nusc.get("sample", tok)
                cam = self.nusc.get("sample_data", sample["data"]["CAM_FRONT"])
                images.append(os.path.join(self.nusc.dataroot, cam["filename"]))
                poses.append(self.nusc.get("ego_pose", cam["ego_pose_token"]))
                if tok == scene["last_sample_token"]:
                    break
                tok = sample["next"]

            if len(images) < TTL_LEN:
                continue

            DT = 0.5
            world = np.array([p["translation"][:3] for p in poses])
            vel = np.zeros_like(world)
            vel[1:] = (world[1:] - world[:-1]) / DT
            vel[0] = vel[1]
            curv = EstimateCurvatureFromTrajectory(world)
            speed = np.linalg.norm(vel, axis=1)

            for i in range(len(images) - TTL_LEN):
                obs = np.stack([speed[i:i+OBS_LEN], curv[i:i+OBS_LEN] * 100], axis=1)   # (10,2)
                fut = np.stack([speed[i+OBS_LEN:i+TTL_LEN],
                                 curv[i+OBS_LEN:i+TTL_LEN] * 100], axis=1)               # (10,2)
                self.samples.append(dict(
                    image_path=images[i + OBS_LEN - 1],
                    obs=obs.astype(np.float32),
                    fut=fut.astype(np.float32),
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return dict(
            image_path=s["image_path"],
            raw_lang=build_prompt(s["obs"]),                 # SAME prompt fn used at inference time
            state=torch.from_numpy(s["obs"].flatten()),       # (20,) -- the "history" numbers
            action=torch.from_numpy(s["fut"]),                 # (10,2) -- the correct-answer numbers (diffusion target)
            is_pad=torch.zeros(10, dtype=torch.bool),
        )
