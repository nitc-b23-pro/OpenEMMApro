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

# ADDED (OOM mitigation): step 0's forward+backward+optimizer.step() alone used
# ~14.17 GiB of the card's 14.56 GiB, then OOM'd on step 1's forward pass.
# AdamW allocates its exp_avg/exp_avg_sq state (2x extra memory per trainable
# param) LAZILY, on the first optimizer.step() call -- so step 0 runs under a
# "no optimizer state yet" budget that step 1 never gets again. Gradient
# checkpointing trades compute for memory: instead of keeping every
# transformer layer's activations around for backward, it recomputes them
# on the fly, which cuts backward-pass memory substantially for a model this
# size. `enable_input_require_grads()` is the standard companion call needed
# when combining gradient checkpointing with a model whose input embeddings
# are frozen (as they are here, apart from LoRA) -- without it, checkpointing
# can silently stop gradients from flowing back through the LoRA-adapted
# layers at all.
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

dataset = OpenEMMADataset(dataroot=DATAROOT, version=VERSION)   # reads datasets/NuScenes, no intents_cache.json any more
# CHANGED (OOM mitigation): batch_size 4 -> 2. Per-step activation memory
# scales ~linearly with batch size, and this is the single biggest lever
# available without touching model architecture. GRAD_ACCUM_STEPS=2 below
# accumulates gradients over 2 micro-batches of size 2 before every
# optimizer.step(), so the *effective* batch size the model trains at (and
# the LoRA/head learning rates were tuned for) stays 4 -- only the peak
# memory of any single forward/backward drops.
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 2
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Two learning rates, exactly like TinyVLA's original recipe:
#   - a small one for the LoRA "patches" on the pretrained VLM (don't want to move it far)
#   - a bigger one for the brand-new diffusion head (it's starting from random, needs to move more)
lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "embed_out" not in n and "proj_to_action" not in n]
head_params = [p for n, p in model.named_parameters() if p.requires_grad and ("embed_out" in n or "proj_to_action" in n)]
optimizer = torch.optim.AdamW([
    # {"params": lora_params, "lr": 2e-4},
    # {"params": head_params, "lr": 2e-5},
    {"params": lora_params, "lr": 1e-5},
    {"params": head_params, "lr": 2e-4},
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

optimizer.zero_grad()

for epoch in range(EPOCHS):
    print(f"Epoch {epoch} running...!")
    for step, batch in enumerate(loader):
        frame_start = time.time()

        input_ids, images = preprocess_batch(batch)
        state = batch["state"].cuda()      # (B, 20)
        action = batch["action"].cuda()    # (B, 10, 2) -- GT future trajectories (diffusion regression target)
        is_pad = batch["is_pad"].cuda()    # (B, 10)

        # ADDED (OOM survivability across the WHOLE run, not just step 0->1):
        # the step-0->step-1 crash was a ONE-TIME jump -- AdamW allocates
        # exp_avg/exp_avg_sq once, on the first optimizer.step(), and that
        # memory is reused in place forever after, it does not keep growing
        # every step. So that specific mechanism cannot repeat later in
        # training. But total memory per step is NOT perfectly flat for the
        # rest of the run either: preprocess_batch() pads input_ids to the
        # LONGEST prompt in each specific batch (dynamic padding), so a
        # batch that happens to draw a longer raw_lang text needs more
        # activation memory than average; over thousands of steps across 5
        # epochs, plus ordinary CUDA allocator fragmentation (expandable_
        # segments helps but doesn't guarantee zero), a rare step can still
        # spike close to the ceiling this GPU is already running near. A
        # crash on step 4000 of 5 would otherwise lose the rest of that
        # epoch's progress. So: catch the OOM, free what we can, skip just
        # that one micro-batch, and keep the run alive -- this is a safety
        # net for rare spikes, not a fix for a batch size that's
        # structurally too big (if EVERY step OOMs, skipping won't help).
        try:
            out = model(input_ids=input_ids, images=images, states=state,
                        actions=action, is_pad=is_pad)
            loss = out["loss"] if isinstance(out, dict) else out.loss

            # ADDED (loss=nan guard): if a NaN/Inf loss still slips through
            # despite the curvature clip in utils.py (belt-and-suspenders --
            # there could be other degenerate samples we haven't seen yet),
            # NEVER call .backward()/optimizer.step() on it. AdamW's exp_avg /
            # exp_avg_sq are *cumulative* running averages -- one NaN gradient
            # poisons that state permanently, and every future update for that
            # parameter is NaN forever after, even once the bad batch is long
            # gone. Skipping the step (but not the frame-timing bookkeeping)
            # costs one batch of training and keeps the run alive; the printed
            # diagnostics show exactly which raw state/action values triggered
            # it, in case the clip needs to be tightened further.
            loss_is_finite = torch.isfinite(loss)
            if not loss_is_finite:
                print(f"epoch {epoch} step {step} | SKIPPED (loss={loss.item()}) | "
                      f"state min/max={state.min().item():.3f}/{state.max().item():.3f} | "
                      f"action min/max={action.min().item():.3f}/{action.max().item():.3f}")
                optimizer.zero_grad()
            else:
                # CHANGED (OOM mitigation, gradient accumulation): normalize by
                # GRAD_ACCUM_STEPS so the accumulated gradient over
                # GRAD_ACCUM_STEPS micro-batches matches the scale a single
                # batch_size=4 step would have produced -- this is what keeps
                # the effective batch size (and the tuned learning rates) at 4
                # even though each micro-batch here is only size 2.
                (loss / GRAD_ACCUM_STEPS).backward()
                if (step + 1) % GRAD_ACCUM_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()
        except torch.OutOfMemoryError as e:
            print(f"epoch {epoch} step {step} | OOM, SKIPPING this micro-batch "
                  f"(batch_size={input_ids.shape[0]}, padded_seq_len={input_ids.shape[1]}): {e}")
            optimizer.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

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
