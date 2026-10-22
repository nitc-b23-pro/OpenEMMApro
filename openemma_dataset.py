"""
openemma_dataset.py
Turns nuScenes scenes into a PyTorch Dataset of
(image_path, prompt, history[10,2], future[10,2]) samples for OpenEMMA
local trajectory prediction.
"""
import os

import numpy as np
import torch
from nuscenes import NuScenes

from prompts import build_prompt, OBS_LEN as _OBS_LEN, FUT_LEN as _FUT_LEN
from utils import EstimateCurvatureFromTrajectory

OBS_LEN, FUT_LEN = _OBS_LEN, _FUT_LEN
TTL_LEN = OBS_LEN + FUT_LEN

DT = 0.5   # nuScenes keyframes are 2 Hz


def iter_scene_motion(nusc):
    """
    Yields (scene_name, image_paths, speed, curvature, world_positions,
    velocity) for every scene with at least TTL_LEN frames.
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
    def __init__(self, dataroot="datasets/NuScenes", version="v1.0-test"):
        self.nusc = NuScenes(version=version, dataroot=dataroot)

        self.samples = []
        for name, images, speed, curv, world, vel in iter_scene_motion(self.nusc):
            for i in range(len(images) - TTL_LEN):
                obs = np.stack([speed[i:i + OBS_LEN], curv[i:i + OBS_LEN] * 100], axis=1)
                fut = np.stack([speed[i + OBS_LEN:i + TTL_LEN], curv[i + OBS_LEN:i + TTL_LEN] * 100], axis=1)

                self.samples.append(dict(
                    image_path=images[i + OBS_LEN - 1],
                    obs=obs.astype(np.float32),
                    fut=fut.astype(np.float32),
                    scene=name,
                    sample_id=f"{name}__{i}",
                    # World-frame position/velocity/future-positions for
                    # ADE tracking in meters (train_openemma.py), not model inputs.
                    cur_pos=world[i + OBS_LEN - 1, :2].astype(np.float32),
                    cur_vel=vel[i + OBS_LEN - 1, :2].astype(np.float32),
                    fut_world=world[i + OBS_LEN:i + TTL_LEN, :2].astype(np.float32),
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return dict(
            image_path=s["image_path"],
            raw_lang=build_prompt(s["obs"]),
            state=torch.from_numpy(s["obs"].flatten()),
            action=torch.from_numpy(s["fut"]),
            scene=s["scene"],
            sample_id=s["sample_id"],
            cur_pos=torch.from_numpy(s["cur_pos"]),
            cur_vel=torch.from_numpy(s["cur_vel"]),
            fut_world=torch.from_numpy(s["fut_world"]),
        )
