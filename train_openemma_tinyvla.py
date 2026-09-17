"""
train_openemma_tinyvla.py
Simple single-GPU training loop for OpenEMMA-TinyVLA.

CHANGED (dataset split): trains on nuScenes v1.0-test (see
openemma_dataset.py's docstring for why v1.0-test is valid here -- ground
truth comes only from ego_pose, not from the withheld sample_annotation
labels).

CHANGED (prompt): the dataset now builds each sample's prompt text via
prompts.build_prompt() -- the same function main.py uses at inference --
so there is nothing prompt-related to change here; this file already
just forwards batch["raw_lang"] into the tokenizer, unchanged.

ADDED: a global frame counter, per-frame processing time, and a running
global-average frame-processing time, printed every logged step. "Frame"
here means one training IMAGE (a batch of batch_size=4 contains 4
frames), so per-frame time = this step's wall time / batch_size, and the
running average is total accumulated time / total frames seen so far --
directly comparable to main.py's per-frame inference timing.

ADDED (OOM mitigation): PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,
set BEFORE torch is imported -- this is literally what the "CUDA out of
memory" error you hit earlier suggested trying, and it reduces allocator
fragmentation that can trigger an OOM even when total usage looks like it
should fit.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time
import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

from build_model import build_openemma_tinyvla
from openemma_dataset import OpenEMMADataset
from llava_pythia.mm_utils import tokenizer_image_token
from llava_pythia.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from transformers import AutoTokenizer, CLIPImageProcessor

PRETRAINED = "/kaggle/input/models/latheeshpoondla/llava-pythia/transformers/h/1/"   # swap in B or H later
DATAROOT = "/kaggle/working/nuscenes_test_root/"
VERSION = "v1.0-test"   # <-- training split (per project requirement)

model = build_openemma_tinyvla(PRETRAINED).cuda()
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED)
image_processor = CLIPImageProcessor.from_pretrained(PRETRAINED)

dataset = OpenEMMADataset(dataroot=DATAROOT, version=VERSION)   # reads datasets/NuScenes, no intents_cache.json any more
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Two learning rates, exactly like TinyVLA's original recipe:
#   - a small one for the LoRA "patches" on the pretrained VLM (don't want to move it far)
#   - a bigger one for the brand-new diffusion head (it's starting from random, needs to move more)
lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "embed_out" not in n and "proj_to_action" not in n]
head_params = [p for n, p in model.named_parameters() if p.requires_grad and ("embed_out" in n or "proj_to_action" in n)]
optimizer = torch.optim.AdamW([
    {"params": lora_params, "lr": 2e-4},
    {"params": head_params, "lr": 2e-5},
])

def preprocess_batch(batch):
    input_ids_list, images_list = [], []
    for image_path, raw_lang in zip(batch["image_path"], batch["raw_lang"]):
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + raw_lang
        input_ids_list.append(tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"))
        img = Image.open(image_path).convert("RGB")
        images_list.append(image_processor.preprocess(img, return_tensors="pt")["pixel_values"][0])

    max_len = max(x.shape[0] for x in input_ids_list)
    padded = torch.stack([torch.nn.functional.pad(x, (0, max_len - x.shape[0]), value=tokenizer.pad_token_id)
                           for x in input_ids_list])
    # FIXED (dtype mismatch): build_model.py now loads the base VLM (including
    # mm_projector, which is plain nn.Linear -- not LoRA-adapted, so there is
    # no PEFT-level dtype auto-casting protecting it) in fp16. CLIPVisionTower's
    # own forward() casts its OUTPUT back to match whatever dtype the INPUT
    # image tensor was, so if we hand it fp32 here, mm_projector (fp16 weights)
    # gets an fp32 input and F.linear crashes on the dtype mismatch. Casting to
    # fp16 here makes that round-trip land on fp16, matching mm_projector.
    return padded.cuda(), torch.stack(images_list).cuda().half()

EPOCHS = 5

# --- global frame counter + running average frame-processing time ---
global_frame_count = 0
total_process_time = 0.0

for epoch in range(EPOCHS):
    print(f"Epoch {epoch} running...!")
    for step, batch in enumerate(loader):
        frame_start = time.time()

        input_ids, images = preprocess_batch(batch)
        state = batch["state"].cuda()      # (B, 20)
        action = batch["action"].cuda()    # (B, 10, 2) -- GT future trajectories (diffusion regression target)
        is_pad = batch["is_pad"].cuda()    # (B, 10)

        out = model(input_ids=input_ids, images=images, states=state,
                    actions=action, is_pad=is_pad)
        loss = out["loss"] if isinstance(out, dict) else out.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_time = time.time() - frame_start
        batch_size = input_ids.shape[0]

        global_frame_count += batch_size
        total_process_time += step_time
        per_frame_time = step_time / batch_size
        running_avg_frame_time = total_process_time / global_frame_count

        if step % 20 == 0:
            print(f"epoch {epoch} step {step} | global_frame_count={global_frame_count} | "
                  f"loss={loss.item():.4f} | per_frame_time={per_frame_time:.3f}s | "
                  f"running_avg_frame_time={running_avg_frame_time:.3f}s")

    model.save_pretrained(f"openemma_tinyvla_epoch{epoch}")

print(f"Training done. Total frames processed: {global_frame_count}, "
      f"final running average frame-processing time: {total_process_time / max(global_frame_count, 1):.3f}s")
