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

ADDED (validation split + val_epoch_losses.json): dataset samples are now
split by WHOLE SCENE into a training set and a held-out validation set
(VAL_FRACTION of scenes, picked with a fixed seed so the split is
reproducible across runs) -- this split is NOT behind a flag, it always
runs. Splitting by scene, not by individual frame, matters because
consecutive frames in the same scene come from heavily overlapping
10-past/10-future windows -- a frame-level random split would put
near-duplicate windows on both sides and make the validation number
meaningless. After each epoch's training loop, the model is switched to
eval mode and run (no_grad, no backward, no optimizer step) over the
validation set once, using the exact same forward call and loss as
training. The resulting per-epoch average is written to
val_epoch_losses.json, mirroring the existing lp_epoch_loss.json format
exactly (same {"epoch-N": avg_loss} shape), so the two can be plotted
against each other directly (see the matplotlib block at the bottom of
this file, or the standalone plot_losses.py if you want to regenerate the
graph later without re-running training).

FIXED (NaN/Inf loss was silently poisoning the WHOLE epoch's average):
previously, when a step's loss was non-finite, the code printed a warning
and skipped backward()/optimizer.step() -- but did NOT skip the
`s_loss += loss.item()` / `s_t += 1` bookkeeping below it. Since
loss.item() is itself NaN/Inf in that branch, adding it to s_loss made
EVERY subsequent addition that epoch NaN too (NaN + anything = NaN), so
a single bad batch anywhere in an epoch would silently turn that whole
epoch's saved average into NaN. This is now handled the same way the OOM
branch already was: `continue` immediately after the warning, so a
non-finite step is excluded from the epoch average instead of poisoning
it. (The same fix is applied to the new validation loop below.)

ADDED (--use-intent / --no-intent CLI flag, default ON): whether each
sample's prompt includes a route-navigation intent line (see
route_intent.py / generate_intents.py / prompts.py) is now a command-line
choice instead of an always-on constant. Default is --use-intent (True),
which is "like previous" in the sense that this is the behavior that was
already shipped: INTENTS_PATH is looked up and threaded into every
prompt if generate_intents.py has produced it. Pass --no-intent to train
with the exact prompt behavior from before the intent feature existed at
all (no intent line, ever, regardless of whether the intents JSON
exists). This matters for a clean comparison: a checkpoint trained with
--no-intent should be evaluated with main.py's/eval_train_vs_test.py's
--no-intent too, or the prompt distribution at eval won't match training.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import random
import time
import torch
from torch.utils.data import DataLoader, Subset
from PIL import Image
import numpy as np

from build_model import build_openemma_tinyvla
from openemma_dataset import OpenEMMADataset
from llava_pythia.mm_utils import tokenizer_image_token
from llava_pythia.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from transformers import AutoTokenizer, CLIPImageProcessor


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--use-intent", dest="use_intent", action="store_true", default=True,
                         help="Thread a route-navigation intent (route_intent.py / generate_intents.py) "
                              "into each sample's prompt, same as prompts.py's intent= argument (default: on).")
    parser.add_argument("--no-intent", dest="use_intent", action="store_false",
                         help="Disable intent in the prompt -- reverts to the exact prompt text from "
                              "before the intent feature existed, regardless of whether an intents JSON is present.")
    return parser.parse_args()


args = parse_args()

PRETRAINED = "/kaggle/input/models/latheeshpoondla/llava-pythia/transformers/h/1/"   # swap in B or H later
DATAROOT = "/kaggle/working/nuscenes_test_root/"
VERSION = "v1.0-test"   # <-- training split (per project requirement)

# CHANGED: INTENTS_PATH is now gated by --use-intent/--no-intent (see this
# file's module docstring). If generate_intents.py has already produced this
# file for this exact dataroot/version, each sample's intent gets threaded
# into its prompt via openemma_dataset.py / prompts.py. If the file doesn't
# exist yet, OpenEMMADataset falls back to intent=None everywhere -- nothing
# else below needs to change either way. See openemma_dataset.py's and
# prompts.py's docstrings for the privileged-information caveat before
# relying on this for a real run.
INTENTS_PATH = "intents_v1.0-test.json" if args.use_intent else None

if args.use_intent:
    _found = os.path.exists(INTENTS_PATH)
    print(f"Intent ENABLED: looking up {INTENTS_PATH} "
          f"({'found' if _found else 'NOT FOUND -- every sample will fall back to no-intent prompts until generate_intents.py has been run for this dataroot/version'})")
else:
    print("Intent DISABLED (--no-intent): prompts built exactly as before the intent feature existed.")

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

dataset = OpenEMMADataset(dataroot=DATAROOT, version=VERSION, intents_path=INTENTS_PATH)   # reads datasets/NuScenes

# ADDED (scene-level train/val split, ALWAYS ON -- not a flag): group sample
# INDICES by scene, then hold out whole scenes for validation -- never
# individual frames -- so overlapping windows from the same scene can't leak
# across the split.
VAL_FRACTION = 0.1   # ~10% of scenes held out for validation
SPLIT_SEED = 42       # fixed, so re-running the script reproduces the same split
# NOTE: eval_train_vs_test.py reproduces this exact VAL_FRACTION/SPLIT_SEED
# pair to recover which scenes were actually trained on (vs held out) --
# if you change either constant here, update it there too.

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

# CHANGED (OOM mitigation): batch_size 4 -> 2. Per-step activation memory
# scales ~linearly with batch size, and this is the single biggest lever
# available without touching model architecture. GRAD_ACCUM_STEPS=2 below
# accumulates gradients over 2 micro-batches of size 2 before every
# optimizer.step(), so the *effective* batch size the model trains at (and
# the LoRA/head learning rates were tuned for) stays 4 -- only the peak
# memory of any single forward/backward drops.
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 2
loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Two learning rates, exactly like TinyVLA's original recipe:
#   - a small one for the LoRA "patches" on the pretrained VLM (don't want to move it far)
#   - a bigger one for the brand-new diffusion head (it's starting from random, needs to move more)
lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "embed_out" not in n and "proj_to_action" not in n]
head_params = [p for n, p in model.named_parameters() if p.requires_grad and ("embed_out" in n or "proj_to_action" in n)]
optimizer = torch.optim.AdamW([
    {"params": lora_params, "lr": 2e-4},
    {"params": head_params, "lr": 2e-5},
    # {"params": lora_params, "lr": 1e-5},
    # {"params": head_params, "lr": 2e-4},
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

EPOCHS = 20

# --- global frame counter + running average frame-processing time ---
global_frame_count = 0
total_process_time = 0.0

optimizer.zero_grad()

filename = "lp_epoch_loss.json"
val_filename = "val_epoch_losses.json"   # ADDED

# Create the files with an empty dictionary if they don't exist yet
losses = {}
if not os.path.exists(filename):
  with open(filename, "w") as file:
    json.dump(losses, file)

val_losses = {}   # ADDED
if not os.path.exists(val_filename):   # ADDED
  with open(val_filename, "w") as file:   # ADDED
    json.dump(val_losses, file)   # ADDED


def run_validation(epoch):
    """
    ADDED. One no-grad pass over val_loader, using the exact same forward
    call and loss as training (just no backward()/optimizer.step()), and
    the same finite-loss guard as the training loop below. Returns the
    average loss over the epoch, or None if every single validation step
    was skipped (OOM / non-finite) -- which would mean something is
    actually wrong, not just "no validation data."
    """
    model.eval()
    v_loss, v_t = 0.0, 0
    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            input_ids, images = preprocess_batch(batch)
            state = batch["state"].cuda()
            action = batch["action"].cuda()
            is_pad = batch["is_pad"].cuda()

            try:
                out = model(input_ids=input_ids, images=images, states=state,
                            actions=action, is_pad=is_pad)
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


for epoch in range(EPOCHS):
    print(f"Epoch {epoch} running...!")
    s_loss, s_t = 0, 0
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
                # FIXED: this used to fall through to the s_loss/s_t
                # bookkeeping below with a NaN/Inf loss.item(), which
                # permanently poisoned the rest of this epoch's average
                # (NaN + anything = NaN). Skip this step's bookkeeping too,
                # exactly like the OOM branch already does.
                continue
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

        s_loss += loss.item()
        s_t += 1

        if step % 20 == 0:
            print(f"epoch {epoch} step {step} | global_frame_count={global_frame_count} | "
                  f"loss={loss.item():.4f} | per_frame_time={per_frame_time:.3f}s | "
                  f"running_avg_frame_time={running_avg_frame_time:.3f}s")

    with open(filename, "r") as file:
        losses = json.load(file)

    # Update the dictionary
    losses[f"epoch-{epoch}"] = s_loss/s_t

    # Write back the fresh dictionary
    with open(filename, "w") as file:
        json.dump(losses, file, indent=4)

    # ADDED: one held-out validation pass per epoch, saved the same way.
    val_avg = run_validation(epoch)
    print(f"epoch {epoch} | validation loss = {val_avg}")
    with open(val_filename, "r") as file:
        val_losses = json.load(file)
    val_losses[f"epoch-{epoch}"] = val_avg
    with open(val_filename, "w") as file:
        json.dump(val_losses, file, indent=4)

    model.save_pretrained(f"openemma_tinyvla_epoch{epoch}")

print(f"Training done. Total frames processed: {global_frame_count}, "
      f"final running average frame-processing time: {total_process_time / max(global_frame_count, 1):.3f}s")

import matplotlib.pyplot as plt

with open(filename, "r") as file:
    losses = json.load(file)
with open(val_filename, "r") as file:   # ADDED
    val_losses = json.load(file)   # ADDED

plt.figure(figsize=(12, 6))
plt.plot(
    list(losses.keys()), list(losses.values()), marker="o", linestyle="-", label="train"
)
# ADDED: overlay validation loss on the same axes, same x labels, so the two
# curves are directly comparable at a glance.
plt.plot(
    list(val_losses.keys()), list(val_losses.values()), marker="s", linestyle="--", label="val"
)

# Add labels and title for readability
plt.xlabel("Epoch")
plt.ylabel("Loss Value")
plt.title("Training vs. Validation Loss per Epoch")
plt.xticks(rotation=45)  # Rotate epoch labels if they crowd together
plt.legend()
plt.grid(True)

# 1. SAVE FIRST (with dpi=300 for high resolution)
plt.savefig("ep_loss_graph.jpg", dpi=300, bbox_inches="tight")

# 2. SHOW LAST
plt.show()
