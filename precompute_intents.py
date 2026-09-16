"""
precompute_intents.py
Runs DescribeOrUpdateIntent() once over the whole nuScenes training set and
saves the results to disk, so the training script never has to call a
slow text-generating VLM during training.
"""
import os, json
from math import atan2
import numpy as np
from nuscenes import NuScenes
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import torch

OBS_LEN, FUT_LEN = 10, 10
TTL_LEN = OBS_LEN + FUT_LEN

# Reuse your existing, already-working Qwen model JUST for this one-time text step.
# (We are not throwing Qwen away — we're using it as a helper to label our training
#  data. The new LLaVA-Pythia model is the one that ends up in your final pipeline.)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/kaggle/input/models/qwen-lm/qwen2.5-vl/transformers/3b-instruct/2/",
    dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto")
processor = AutoProcessor.from_pretrained(
    "/kaggle/input/models/qwen-lm/qwen2.5-vl/transformers/3b-instruct/2/", use_fast=False)

def vlm_text(prompt, image_path):
    message = [{"role": "user", "content": [{"type": "image", "image": image_path},
                                              {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs = process_vision_info(message)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                        padding=True, return_tensors="pt").to(model.device)
    ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]

def describe_or_update_intent(image_path, prev_intent):
    if prev_intent is None:
        prompt = ("You are a autonomous driving labeller. ... describe the desired "
                   "intent of the ego car ...")
    else:
        prompt = (f"You are a autonomous driving labeller. ... Half a second ago your "
                   f"intent was to {prev_intent}. ... Explain your current intent: ")
    return vlm_text(prompt, image_path)

if __name__ == "__main__":
    nusc = NuScenes(version="v1.0-mini", dataroot="datasets/NuScenes")
    intents_cache = {}   # key: f"{scene_name}_{frame_idx}"  ->  intent text

    for scene in nusc.scene:
        name = scene["name"]
        tok = scene["first_sample_token"]
        images = []
        while True:
            sample = nusc.get("sample", tok)
            cam = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
            images.append(os.path.join(nusc.dataroot, cam["filename"]))
            if tok == scene["last_sample_token"]:
                break
            tok = sample["next"]

        if len(images) < TTL_LEN:
            continue

        prev_intent = None
        for i in range(len(images) - TTL_LEN):
            current_image = images[i + OBS_LEN - 1]
            prev_intent = describe_or_update_intent(current_image, prev_intent)
            intents_cache[f"{name}_{i}"] = prev_intent
            print(f"{name}_{i}: {prev_intent[:60]}...")

    with open("intents_cache.json", "w") as f:
        json.dump(intents_cache, f, indent=2)
    print(f"Saved {len(intents_cache)} intents to intents_cache.json")