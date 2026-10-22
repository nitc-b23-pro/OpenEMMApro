"""
train_openemma_qwen.py
Same training recipe as train_openemma_tinyvla.py -- diffusion policy
head + LoRA, scene-level train/val split, class-balanced sampling, real
(meters) validation ADE tracking, NaN/OOM guards -- on a Qwen2.5-VL-3B-
Instruct backbone instead of LLaVA-Pythia (build_model_qwen.py). Branch:
kaggle-openvlm.

What's identical to train_openemma_tinyvla.py (unchanged, reused as-is)
------------------------------------------------------------------------
- openemma_dataset.OpenEMMADataset: backbone-agnostic. It only ever
  returns an image path + prompt TEXT (prompts.build_prompt, also
  backbone-agnostic) + numeric state/action/is_pad tensors -- nothing
  LLaVA-Pythia-specific. No changes needed for Qwen.
- The scene-level train/val split (VAL_FRACTION/SPLIT_SEED), the
  --balance-classes WeightedRandomSampler, the --use-intent /
  intents_v1.0-test.json wiring, the two-learning-rate-group optimizer
  split (LoRA params vs. the diffusion head), the OOM/NaN-loss guards,
  the per-epoch loss/val-loss/val-ADE JSON logging and plotting -- all
  identical logic, copied over unchanged.

What's different (Qwen-specific)
---------------------------------
- preprocess_batch(): LLaVA-Pythia's version hand-builds input_ids via
  tokenizer_image_token() (a single <image> placeholder token, DIY
  padding) and runs CLIPImageProcessor on raw images. Qwen2.5-VL has its
  own native multimodal input format (<|image_pad|> tokens expanded to
  match the ViT's patch grid, packed `pixel_values` + `image_grid_thw`,
  not a per-image fixed-size tensor) built by its own AutoProcessor via
  a chat-template message + qwen_vl_utils.process_vision_info -- see
  this file's preprocess_batch() below. This is the exact same
  message/processor pattern this repo's own pre-TinyVLA Qwen path used
  (see git history: commit b33a7b7's vlm_inference()), just reused here
  to build model inputs instead of calling .generate().
- build_model_qwen.build_openemma_qwen() instead of build_model.
  build_openemma_tinyvla(); model(...) is called with
  input_ids/attention_mask/pixel_values/image_grid_thw instead of
  input_ids/images.
- Output file names are prefixed `qwen_` (qwen_epoch_loss.json etc.) and
  checkpoints are written to openemma_qwen_epoch{N}/, so running this
  alongside the LLaVA-Pythia pipeline in the same working directory
  never overwrites its lp_*/openemma_tinyvla_epoch* outputs.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import random
import time
from collections import Counter
from math import atan2

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from transformers import AutoProcessor

from build_model import build_openemma_qwen
from openemma_dataset import OpenEMMADataset
from utils import IntegrateCurvatureForPoints


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--use-intent", dest="use_intent", action="store_true", default=True,
                         help="Thread a route-navigation intent into each sample's prompt (default: on).")
    parser.add_argument("--no-intent", dest="use_intent", action="store_false")
    parser.add_argument("--balance-classes", dest="balance_classes", action="store_true", default=True,
                         help="Oversample rare maneuver classes via a WeightedRandomSampler (default: on).")
    parser.add_argument("--no-balance-classes", dest="balance_classes", action="store_false")
    parser.add_argument("--ade-eval-every", type=int, default=5,
                         help="Run a real validation ADE check every N epochs, plus always on the final epoch (default: 5).")
    parser.add_argument("--ade-eval-samples", type=int, default=64,
                         help="How many validation samples the ADE check uses each time it runs (default: 64).")
    return parser.parse_args()


args = parse_args()

# EDIT ME: point this at wherever your Kaggle Notebook mounts the
# Qwen2.5-VL-3B-Instruct input (Kaggle Models path convention, same idea
# as train_openemma_tinyvla.py's LLaVA-Pythia PRETRAINED constant --
# adjust the slug/variation/version segments to match your own input).
PRETRAINED = "/kaggle/input/qwen2.5-vl-3b-instruct/transformers/default/1/"
DATAROOT = "/kaggle/working/nuscenes_test_root/"
VERSION = "v1.0-test"   # training split (per project requirement)

INTENTS_PATH = "intents_v1.0-test.json" if args.use_intent else None
if args.use_intent:
    _found = os.path.exists(INTENTS_PATH)
    print(f"Intent ENABLED: looking up {INTENTS_PATH} "
          f"({'found' if _found else 'NOT FOUND -- falling back to no-intent prompts until generate_intents.py has been run'})")
else:
    print("Intent DISABLED (--no-intent).")

model = build_openemma_qwen(PRETRAINED).cuda()
processor = AutoProcessor.from_pretrained(PRETRAINED)

model.gradient_checkpointing_enable()
model.enable_input_require_grads()

dataset = OpenEMMADataset(dataroot=DATAROOT, version=VERSION, intents_path=INTENTS_PATH)

VAL_FRACTION = 0.1
SPLIT_SEED = 42
# NOTE: eval_train_vs_test_qwen.py reproduces this exact VAL_FRACTION/
# SPLIT_SEED pair -- keep both in sync if you change either here.

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

BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 2

train_classes = [dataset.samples[i]["fut_class"] for i in train_indices]
class_counts = Counter(train_classes)
print(f"Train-set maneuver class distribution (before any balancing): {dict(class_counts)}")

if args.balance_classes:
    class_weight = {c: 1.0 / n for c, n in class_counts.items()}
    mean_w = float(np.mean(list(class_weight.values())))
    class_weight = {c: w / mean_w for c, w in class_weight.items()}
    print(f"Class weights (--balance-classes, inverse-frequency): {class_weight}")

    sample_weights = [class_weight[c] for c in train_classes]
    train_sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(train_indices), replacement=True,
        generator=torch.Generator().manual_seed(SPLIT_SEED),
    )
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
else:
    print("Class balancing DISABLED.")
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Same two-learning-rate-group recipe as train_openemma_tinyvla.py: a
# small LR for the LoRA patches on the pretrained VLM, a bigger one for
# the from-scratch diffusion head. The substring filter works unchanged
# here even though the parameter names now have a `qwen.` prefix (from
# OpenEMMAQwenPolicy) instead of none -- "embed_out"/"proj_to_action"
# still only ever appear in the head's own parameter names.
lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "embed_out" not in n and "proj_to_action" not in n]
head_params = [p for n, p in model.named_parameters() if p.requires_grad and ("embed_out" in n or "proj_to_action" in n)]
optimizer = torch.optim.AdamW([
    {"params": lora_params, "lr": 2e-4},
    {"params": head_params, "lr": 2e-5},
])


def preprocess_batch(batch):
    """
    Qwen2.5-VL analogue of train_openemma_tinyvla.py's preprocess_batch:
    builds one chat-template message per sample ({"type": "image", ...} +
    {"type": "text", "text": raw_lang}, raw_lang itself already built by
    the SAME prompts.build_prompt() the LLaVA-Pythia backend uses), lets
    qwen_vl_utils.process_vision_info + the processor's own batching
    handle padding/packing (Qwen2.5-VL's `pixel_values` are flattened ViT
    patches across the WHOLE batch, split back out per-image via
    `image_grid_thw` -- there is no separate per-sample image tensor to
    manually stack the way CLIPImageProcessor needed).
    """
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
    # FIXED (dtype mismatch, same root cause as build_model_qwen.py's base
    # model being loaded fp16): the processor returns pixel_values as
    # fp32 by default; the frozen ViT patch-embed/merger weights are fp16.
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)
    return inputs


EPOCHS = 40
DT_ADE = 0.5
ADE_EVAL_EVERY = args.ade_eval_every
ADE_EVAL_SAMPLES = args.ade_eval_samples

global_frame_count = 0
total_process_time = 0.0

optimizer.zero_grad()

filename = "qwen_epoch_loss.json"
val_filename = "qwen_val_epoch_losses.json"
val_ade_filename = "qwen_val_ade_epochs.json"

losses = {}
if not os.path.exists(filename):
    with open(filename, "w") as file:
        json.dump(losses, file)

val_losses = {}
if not os.path.exists(val_filename):
    with open(val_filename, "w") as file:
        json.dump(val_losses, file)

val_ade = {}
if not os.path.exists(val_ade_filename):
    with open(val_ade_filename, "w") as file:
        json.dump(val_ade, file)


def run_validation(epoch):
    model.eval()
    v_loss, v_t = 0.0, 0
    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            inputs = preprocess_batch(batch)
            state = batch["state"].cuda()
            action = batch["action"].cuda()
            is_pad = batch["is_pad"].cuda()

            try:
                out = model(**inputs, states=state, actions=action, is_pad=is_pad)
                loss = out["loss"] if isinstance(out, dict) else out.loss
            except torch.OutOfMemoryError as e:
                print(f"[val] epoch {epoch} step {step} | OOM, SKIPPING this micro-batch: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            if not torch.isfinite(loss):
                print(f"[val] epoch {epoch} step {step} | SKIPPED (loss={loss.item()})")
                continue

            v_loss += loss.item()
            v_t += 1

    model.train()
    return (v_loss / v_t) if v_t > 0 else None


def run_validation_ade(epoch, max_samples=None):
    model.eval()
    max_samples = max_samples if max_samples is not None else ADE_EVAL_SAMPLES
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
                print(f"[val-ade] epoch {epoch} | OOM, SKIPPING one sample: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            pred = pred[0].cpu().numpy()
            pred_speeds = pred[:, 0]
            pred_curvatures = pred[:, 1] / 100
            cur_pos = s["cur_pos"].numpy()
            cur_vel = s["cur_vel"].numpy()
            fut_world = s["fut_world"].numpy()
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


for epoch in range(EPOCHS):
    print(f"Epoch {epoch} running...!")
    s_loss, s_t = 0, 0
    for step, batch in enumerate(loader):
        frame_start = time.time()

        inputs = preprocess_batch(batch)
        state = batch["state"].cuda()
        action = batch["action"].cuda()
        is_pad = batch["is_pad"].cuda()

        try:
            out = model(**inputs, states=state, actions=action, is_pad=is_pad)
            loss = out["loss"] if isinstance(out, dict) else out.loss

            loss_is_finite = torch.isfinite(loss)
            if not loss_is_finite:
                print(f"epoch {epoch} step {step} | SKIPPED (loss={loss.item()}) | "
                      f"state min/max={state.min().item():.3f}/{state.max().item():.3f} | "
                      f"action min/max={action.min().item():.3f}/{action.max().item():.3f}")
                optimizer.zero_grad()
                continue
            else:
                (loss / GRAD_ACCUM_STEPS).backward()
                if (step + 1) % GRAD_ACCUM_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()
        except torch.OutOfMemoryError as e:
            print(f"epoch {epoch} step {step} | OOM, SKIPPING this micro-batch "
                  f"(batch_size={inputs['input_ids'].shape[0]}): {e}")
            optimizer.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_time = time.time() - frame_start
        batch_size = inputs["input_ids"].shape[0]

        global_frame_count += batch_size
        total_process_time += step_time
        per_frame_time = step_time / batch_size
        running_avg_frame_time = total_process_time / global_frame_count

        s_loss += loss.item()
        s_t += 1

        if step % 20 == 0:
            print(f"epoch {epoch} step {step} | global_frame_count={global_frame_count} | "
                  f"loss={loss.item():.4f} | per_frame_time={per_frame_time:.3f}s | "
                  f"running_avg_frame_time={running_avg_frame_time:.3f}s")

    with open(filename, "r") as file:
        losses = json.load(file)
    losses[f"epoch-{epoch}"] = s_loss / s_t
    with open(filename, "w") as file:
        json.dump(losses, file, indent=4)

    val_avg = run_validation(epoch)
    print(f"epoch {epoch} | validation loss = {val_avg}")
    with open(val_filename, "r") as file:
        val_losses = json.load(file)
    val_losses[f"epoch-{epoch}"] = val_avg
    with open(val_filename, "w") as file:
        json.dump(val_losses, file, indent=4)

    if (epoch % ADE_EVAL_EVERY == 0) or (epoch == EPOCHS - 1):
        ade_summary = run_validation_ade(epoch)
        print(f"epoch {epoch} | validation ADE (meters) = {ade_summary}")
        with open(val_ade_filename, "r") as file:
            val_ade = json.load(file)
        val_ade[f"epoch-{epoch}"] = ade_summary
        with open(val_ade_filename, "w") as file:
            json.dump(val_ade, file, indent=4)

    model.save_pretrained(f"openemma_qwen_epoch{epoch}")

print(f"Training done. Total frames processed: {global_frame_count}, "
      f"final running average frame-processing time: {total_process_time / max(global_frame_count, 1):.3f}s")

import matplotlib.pyplot as plt

with open(filename, "r") as file:
    losses = json.load(file)
with open(val_filename, "r") as file:
    val_losses = json.load(file)
with open(val_ade_filename, "r") as file:
    val_ade = json.load(file)

plt.figure(figsize=(12, 6))
plt.plot(list(losses.keys()), list(losses.values()), marker="o", linestyle="-", label="train")
plt.plot(list(val_losses.keys()), list(val_losses.values()), marker="s", linestyle="--", label="val")
plt.xlabel("Epoch")
plt.ylabel("Loss Value")
plt.title("OpenEMMA-Qwen: Training vs. Validation Loss per Epoch")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.savefig("qwen_ep_loss_graph.jpg", dpi=300, bbox_inches="tight")
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
    plt.title(f"OpenEMMA-Qwen: Validation ADE per Epoch (n={ADE_EVAL_SAMPLES} sampled, every {ADE_EVAL_EVERY} epochs)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.savefig("qwen_val_ade_graph.jpg", dpi=300, bbox_inches="tight")
    plt.show()
else:
    print("No validation ADE was recorded -- skipping the ADE plot.")
