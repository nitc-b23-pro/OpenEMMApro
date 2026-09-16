"""
openemma_dataset.py
Turns your nuScenes scenes into a PyTorch Dataset of
(image, intent_text, history[10,2], future[10,2]) samples.
"""
import os, json
from math import atan2
import numpy as np
import torch
from nuscenes import NuScenes
from utils import EstimateCurvatureFromTrajectory   # your existing file, unchanged

OBS_LEN, FUT_LEN = 10, 10
TTL_LEN = OBS_LEN + FUT_LEN

class OpenEMMADataset(torch.utils.data.Dataset):
    def __init__(self, dataroot="datasets/NuScenes", version="v1.0-mini",
                 intents_cache_path="intents_cache.json"):
        self.nusc = NuScenes(version=version, dataroot=dataroot)
        with open(intents_cache_path) as f:
            self.intents = json.load(f)   # built by precompute_intents.py

        self.samples = []   # each entry: dict(image=..., key=..., obs=..., fut=...)
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
                key = f"{name}_{i}"
                if key not in self.intents:
                    continue   # skip frames we didn't precompute an intent for
                obs = np.stack([speed[i:i+OBS_LEN], curv[i:i+OBS_LEN] * 100], axis=1)   # (10,2)
                fut = np.stack([speed[i+OBS_LEN:i+TTL_LEN],
                                 curv[i+OBS_LEN:i+TTL_LEN] * 100], axis=1)               # (10,2)
                self.samples.append(dict(
                    image_path=images[i + OBS_LEN - 1],
                    intent=self.intents[key],
                    obs=obs.astype(np.float32),
                    fut=fut.astype(np.float32),
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return dict(
            image_path=s["image_path"],
            raw_lang=f"Given the driving intent: {s['intent']}. Predict the future speeds and curvatures.",
            state=torch.from_numpy(s["obs"].flatten()),      # (20,) -- the "history" numbers
            action=torch.from_numpy(s["fut"]),                # (10,2) -- the correct-answer numbers
            is_pad=torch.zeros(10, dtype=torch.bool),
        )