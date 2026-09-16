"""
train_openemma_tinyvla.py
Simple single-GPU training loop for OpenEMMA-TinyVLA.
"""
import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

from build_model import build_openemma_tinyvla
from openemma_dataset import OpenEMMADataset
from llava_pythia.mm_utils import tokenizer_image_token
from llava_pythia.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from transformers import AutoTokenizer, CLIPImageProcessor

PRETRAINED = "path/to/your/llava_pythia_S_checkpoint"   # swap in B or H later

model = build_openemma_tinyvla(PRETRAINED).cuda()
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED)
image_processor = CLIPImageProcessor.from_pretrained(PRETRAINED)

dataset = OpenEMMADataset()   # reads datasets/NuScenes + intents_cache.json
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
    return padded.cuda(), torch.stack(images_list).cuda()

EPOCHS = 5
for epoch in range(EPOCHS):
    for step, batch in enumerate(loader):
        input_ids, images = preprocess_batch(batch)
        state = batch["state"].cuda()      # (B, 20)
        action = batch["action"].cuda()    # (B, 10, 2)
        is_pad = batch["is_pad"].cuda()    # (B, 10)

        out = model(input_ids=input_ids, images=images, states=state,
                    actions=action, is_pad=is_pad)
        loss = out["loss"] if isinstance(out, dict) else out.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            print(f"epoch {epoch} step {step}: loss={loss.item():.4f}")

    model.save_pretrained(f"openemma_tinyvla_epoch{epoch}")