"""
train_openemma.py
Trains the OpenEMMA diffusion policy head + LoRA adapters on a
Qwen2.5-VL-3B-Instruct backbone (build_model.py) for local trajectory
prediction on nuScenes: scene-level train/val split, per-epoch loss
tracking, periodic real (meters) validation ADE, and checkpointing.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import random
import time
from math import atan2

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor

from build_model import build_openemma, compute_action_stats
from openemma_dataset import OpenEMMADataset
from utils import IntegrateCurvatureForPoints

# EDIT ME (or set the OPENEMMA_QWEN_PATH env var): local directory holding
# Qwen2.5-VL-3B-Instruct's weights/config/processor files -- must be a real
# path on disk, not a HF Hub repo id. If your Kaggle input's
# preprocessor_config.json isn't recognized by the installed transformers
# version, fix it in a working copy (e.g. symlink the input dir into
# /kaggle/working/ and patch image_processor_type/processor_class there)
# and point this at that copy -- not the raw input. Keep in sync with
# main.py's BASE_PRETRAINED.
PRETRAINED = os.environ.get("OPENEMMA_QWEN_PATH", "/kaggle/working/qwen2.5-vl-3b-fixed")
DATAROOT = "/kaggle/working/nuscenes_test_root/"
VERSION = "v1.0-test"

VAL_FRACTION = 0.1
SPLIT_SEED = 42
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4   # effective batch size stays 4, but peak per-step memory is halved
EPOCHS = 40
DT_ADE = 0.5
ADE_EVAL_EVERY = 5
ADE_EVAL_SAMPLES = 64

# Fixed (not just capped) visual-token budget: MIN==MAX means every frame
# becomes exactly the same sequence length, so memory use no longer varies
# frame to frame. With a range, GPU memory was staying pinned within a few
# dozen MB of the limit for most frames, and the occasional frame needing
# more tokens than usual was enough to OOM. Keep this in sync with
# main.py's MIN_PIXELS/MAX_PIXELS so inference sees the same token budget
# the model was trained on.
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 256 * 28 * 28

model = build_openemma(PRETRAINED).cuda()
processor = AutoProcessor.from_pretrained(PRETRAINED, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

dataset = OpenEMMADataset(dataroot=DATAROOT, version=VERSION)

# Diffusion sampling (build_model.py's DDIMScheduler) clamps every denoised
# sample to [-1, 1] -- a standard assumption that the data being diffused is
# already normalized to roughly that range. `state`/`action` here are raw
# physical units ([speed in m/s, curvature*100]), nowhere near that scale, so
# without this the model's own forward() refuses to run (see
# OpenEMMAPolicy._stats_set). Must be computed before the scene split below
# only in the sense that it needs `dataset`, not the split itself -- using
# the full dataset here (not just the train split) is fine since these are
# just normalization bounds, not something the model can overfit to.
action_low, action_high = compute_action_stats(dataset)
model.set_action_stats(action_low, action_high)
print(f"Action/state normalization bounds (1st/99th percentile): "
      f"speed=[{action_low[0]:.3f}, {action_high[0]:.3f}] m/s, "
      f"curvature*100=[{action_low[1]:.3f}, {action_high[1]:.3f}]")

scene_names = sorted({s["scene"] for s in dataset.samples})
rng = random.Random(SPLIT_SEED)
rng.shuffle(scene_names)
n_val_scenes = max(1, round(len(scene_names) * VAL_FRACTION))
val_scenes = set(scene_names[:n_val_scenes])
train_scenes = set(scene_names[n_val_scenes:])

train_indices = [i for i, s in enumerate(dataset.samples) if s["scene"] in train_scenes]
val_indices = [i for i, s in enumerate(dataset.samples) if s["scene"] in val_scenes]
print(f"Scene split: {len(train_scenes)} train scenes / {len(val_scenes)} val scenes "
      f"-> {len(train_indices)} train samples / {len(val_indices)} val samples")

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Two learning-rate groups: a small LR for the LoRA patches on the
# pretrained backbone, a larger one for the from-scratch diffusion head.
lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "embed_out" not in n]
head_params = [p for n, p in model.named_parameters() if p.requires_grad and "embed_out" in n]
param_groups = [
    {"params": lora_params, "lr": 2e-4},
    {"params": head_params, "lr": 2e-5},
]
try:
    # 8-bit optimizer state (bitsandbytes) instead of AdamW's usual two
    # fp32 moment buffers per parameter -- roughly a 4x cut in optimizer
    # memory, which matters once the GPU is already tight from the 4-bit
    # backbone + activations (see build_model.py's USE_4BIT).
    import bitsandbytes as bnb
    optimizer = bnb.optim.PagedAdamW8bit(param_groups)
except ImportError:
    optimizer = torch.optim.AdamW(param_groups)


def preprocess_batch(batch):
    messages = []
    for image_path, raw_lang in zip(batch["image_path"], batch["raw_lang"]):
        img = Image.open(image_path).convert("RGB")
        messages.append([{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": raw_lang},
        ]}])

    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)
    return inputs


def run_validation():
    model.eval()
    v_loss, v_steps = 0.0, 0
    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            inputs = preprocess_batch(batch)
            state = batch["state"].cuda()
            action = batch["action"].cuda()
            try:
                out = model(**inputs, states=state, actions=action)
                loss = out["loss"]
            except torch.OutOfMemoryError as e:
                print(f"[val] step {step} | OOM, skipping: {e}")
                torch.cuda.empty_cache()
                continue
            if not torch.isfinite(loss):
                continue
            v_loss += loss.item()
            v_steps += 1
    model.train()
    return (v_loss / v_steps) if v_steps > 0 else None


def run_validation_ade(max_samples=ADE_EVAL_SAMPLES):
    model.eval()
    n = min(max_samples, len(val_dataset))
    if n == 0:
        model.train()
        return None
    idxs = np.linspace(0, len(val_dataset) - 1, num=n, dtype=int)

    ade1s_list, ade2s_list, ade3s_list = [], [], []
    with torch.no_grad():
        for idx in idxs:
            s = val_dataset[int(idx)]
            try:
                inputs = preprocess_batch({"image_path": [s["image_path"]], "raw_lang": [s["raw_lang"]]})
                state = s["state"].unsqueeze(0).cuda()
                pred = model(**inputs, states=state, eval=True)
            except torch.OutOfMemoryError as e:
                print(f"[val-ade] OOM, skipping one sample: {e}")
                torch.cuda.empty_cache()
                continue

            pred = pred[0].cpu().numpy()
            pred_speeds, pred_curvatures = pred[:, 0], pred[:, 1] / 100
            cur_pos, cur_vel, fut_world = s["cur_pos"].numpy(), s["cur_vel"].numpy(), s["fut_world"].numpy()
            initial_heading = atan2(cur_vel[1], cur_vel[0])
            pred_traj = IntegrateCurvatureForPoints(pred_curvatures, pred_speeds, cur_pos, initial_heading, DT_ADE)

            err = np.linalg.norm(fut_world - pred_traj, axis=1)
            ade1s_list.append(float(np.mean(err[:2])))
            ade2s_list.append(float(np.mean(err[:4])))
            ade3s_list.append(float(np.mean(err[:6])))

    model.train()
    if not ade1s_list:
        return None
    mean_ade1s, mean_ade2s, mean_ade3s = float(np.mean(ade1s_list)), float(np.mean(ade2s_list)), float(np.mean(ade3s_list))
    return {
        "ade1s": mean_ade1s, "ade2s": mean_ade2s, "ade3s": mean_ade3s,
        "avgade": float(np.mean([mean_ade1s, mean_ade2s, mean_ade3s])),
        "n_samples": len(ade1s_list),
    }


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


LOSS_FILE = "epoch_loss.json"
VAL_LOSS_FILE = "val_epoch_losses.json"
VAL_ADE_FILE = "val_ade_epochs.json"

losses = load_json(LOSS_FILE)
val_losses = load_json(VAL_LOSS_FILE)
val_ade = load_json(VAL_ADE_FILE)

global_frame_count = 0
total_process_time = 0.0
optimizer.zero_grad()

for epoch in range(EPOCHS):
    print(f"Epoch {epoch} running...")
    epoch_loss, epoch_steps = 0.0, 0

    for step, batch in enumerate(loader):
        frame_start = time.time()
        inputs = preprocess_batch(batch)
        state = batch["state"].cuda()
        action = batch["action"].cuda()

        try:
            out = model(**inputs, states=state, actions=action)
            loss = out["loss"]
            if not torch.isfinite(loss):
                print(f"epoch {epoch} step {step} | SKIPPED (loss={loss.item()})")
                # Don't touch the optimizer's grads here: backward() was
                # never called for this micro-batch, so there's nothing of
                # this step's to clear, and a blanket zero_grad() would
                # wipe out gradients legitimately accumulated from earlier
                # micro-steps in this same accumulation window.
                continue
            (loss / GRAD_ACCUM_STEPS).backward()
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                # Clip before stepping -- unclipped gradients are the
                # likely source of the occasional NaN loss, and a runaway
                # update also risks leaving weights (and the next
                # forward's activations) at a scale that's more likely to
                # blow the memory budget too.
                torch.nn.utils.clip_grad_norm_(lora_params + head_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
        except torch.OutOfMemoryError as e:
            print(f"epoch {epoch} step {step} | OOM, skipping: {e}")
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        batch_size = inputs["input_ids"].shape[0]
        global_frame_count += batch_size
        total_process_time += time.time() - frame_start
        epoch_loss += loss.item()
        epoch_steps += 1

        if step % 20 == 0:
            print(f"epoch {epoch} step {step} | loss={loss.item():.4f} | "
                  f"running_avg_frame_time={total_process_time / global_frame_count:.3f}s")

    losses[f"epoch-{epoch}"] = epoch_loss / epoch_steps
    save_json(LOSS_FILE, losses)

    val_avg = run_validation()
    print(f"epoch {epoch} | validation loss = {val_avg}")
    val_losses[f"epoch-{epoch}"] = val_avg
    save_json(VAL_LOSS_FILE, val_losses)

    if epoch % ADE_EVAL_EVERY == 0 or epoch == EPOCHS - 1:
        ade_summary = run_validation_ade()
        print(f"epoch {epoch} | validation ADE (meters) = {ade_summary}")
        val_ade[f"epoch-{epoch}"] = ade_summary
        save_json(VAL_ADE_FILE, val_ade)

    model.save_pretrained(f"openemma_epoch{epoch}")

print(f"Training done. Total frames processed: {global_frame_count}, "
      f"final running average frame time: {total_process_time / max(global_frame_count, 1):.3f}s")

plt.figure(figsize=(12, 6))
plt.plot(list(losses.keys()), list(losses.values()), marker="o", linestyle="-", label="train")
plt.plot(list(val_losses.keys()), list(val_losses.values()), marker="s", linestyle="--", label="val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("OpenEMMA: Training vs. Validation Loss per Epoch")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.savefig("ep_loss_graph.jpg", dpi=300, bbox_inches="tight")
plt.show()

ade_epochs = [k for k, v in val_ade.items() if v is not None]
if ade_epochs:
    plt.figure(figsize=(12, 6))
    plt.plot(ade_epochs, [val_ade[k]["ade1s"] for k in ade_epochs], marker="o", label="ADE@1s")
    plt.plot(ade_epochs, [val_ade[k]["ade2s"] for k in ade_epochs], marker="o", label="ADE@2s")
    plt.plot(ade_epochs, [val_ade[k]["ade3s"] for k in ade_epochs], marker="o", label="ADE@3s")
    plt.plot(ade_epochs, [val_ade[k]["avgade"] for k in ade_epochs], marker="s", linestyle="--", label="Avg ADE")
    plt.xlabel("Epoch")
    plt.ylabel("ADE (m)")
    plt.title(f"OpenEMMA: Validation ADE per Epoch (n={ADE_EVAL_SAMPLES} sampled, every {ADE_EVAL_EVERY} epochs)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.savefig("val_ade_graph.jpg", dpi=300, bbox_inches="tight")
    plt.show()
else:
    print("No validation ADE was recorded -- skipping the ADE plot.")
