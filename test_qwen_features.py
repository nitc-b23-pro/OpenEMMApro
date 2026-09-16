import argparse
import inspect
import os
import sys
from typing import Any

import torch

try:
    from qwen_vl_utils import process_vision_info
except Exception:
    print("WARNING: qwen_vl_utils not importable in this environment.", file=sys.stderr)
    process_vision_info = None

try:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
except Exception as exc:
    raise RuntimeError(
        "transformers / Qwen2.5-VL is not installed in the active environment. "
        "This script expects the same environment used by main.py."
    ) from exc


def get_message(prompt: str, image_path: str):
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt},
        ],
    }]


def build_prompt(intent_text: str, history_pairs):
    hist = ", ".join(f"[{x[0]:.1f},{x[1]:.1f}]" for x in history_pairs)
    return (
        f"You are a driving predictor. "
        f"The ego vehicle's previous 10 historical speed/curvature values are {hist}. "
        f"The current driving intent is: {intent_text}. "
        f"Briefly describe the scene and identify the most important cues for the next motion."
    )


def summarize_tensor(name: str, value: Any):
    if value is None:
        print(f"{name}: None")
        return

    if torch.is_tensor(value):
        print(f"{name}: shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device}")
        return

    if isinstance(value, (list, tuple)):
        print(f"{name}: type={type(value).__name__}, len={len(value)}")
        if len(value) > 0:
            first = value[0]
            if torch.is_tensor(first):
                print(f"  first element: shape={tuple(first.shape)}, dtype={first.dtype}, device={first.device}")
            else:
                print(f"  first element type={type(first).__name__}")
        return

    print(f"{name}: type={type(value).__name__}")


def inspect_output(label: str, outputs: Any):
    print(f"\n--- {label} ---")
    print(f"type={type(outputs).__name__}")

    if hasattr(outputs, "keys"):
        keys = list(outputs.keys())
        print(f"keys={keys}")
        for k in keys:
            summarize_tensor(k, outputs[k])
    else:
        print("No .keys() method on output object.")

    for attr_name in [
        "logits",
        "hidden_states",
        "image_hidden_states",
        "vision_hidden_states",
        "past_key_values",
        "attentions",
        "cross_attentions",
    ]:
        if hasattr(outputs, attr_name):
            val = getattr(outputs, attr_name)
            summarize_tensor(attr_name, val)


def inspect_model_architecture(model):
    print("\n=== MODEL ARCHITECTURE INSPECTION ===")
    for name in ["model", "visual", "vision_tower", "vision_model", "language_model"]:
        obj = getattr(model, name, None)
        if obj is not None:
            print(f"{name}: {type(obj).__name__}")
            try:
                print(f"  forward signature: {inspect.signature(obj.forward)}")
            except Exception:
                pass

    print("\n=== CANDIDATE FEATURE HOOKS ===")
    hooks = [
        ("model.model.visual", getattr(getattr(model, "model", None), "visual", None)),
        ("model.model.vision_tower", getattr(getattr(model, "model", None), "vision_tower", None)),
        ("model.visual", getattr(model, "visual", None)),
        ("model.vision_tower", getattr(model, "vision_tower", None)),
    ]
    for label, obj in hooks:
        if obj is not None:
            print(f"{label}: {type(obj).__name__}")


def main():
    parser = argparse.ArgumentParser(description="Single-sample Qwen2.5-VL feature probe without text generation.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the local Qwen2.5-VL checkpoint, matching main.py")
    parser.add_argument("--image-path", type=str, required=True, help="Path to a single front camera image used for the probe")
    parser.add_argument("--intent", type=str, required=True, help="Driving intent string to include in the prompt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image file not found: {args.image_path}")

    history = [
        [4.2, 0.0], [4.3, 0.2], [4.5, 0.3], [4.8, 0.4], [5.0, 0.5],
        [5.1, 0.4], [5.0, 0.3], [4.7, 0.1], [4.4, -0.1], [4.3, -0.2],
    ]

    print(f"Loading Qwen2.5-VL checkpoint from: {args.model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="sdpa",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False)
    model.eval()
    model.to(args.device)

    prompt = build_prompt(args.intent, history)
    message = get_message(prompt, args.image_path)
    text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(message)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(args.device)

    print("\n=== INPUT SUMMARY ===")
    for k, v in inputs.items():
        if torch.is_tensor(v):
            print(f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device}")

    print("\n=== FORWARD PASS: ONE SAMPLE, NO TEXT GENERATION ===")
    with torch.inference_mode():
        outputs = model(
            **inputs,
            return_dict=True,
            output_hidden_states=True,
            output_attentions=False,
            use_cache=False,
        )

    inspect_output("MODEL OUTPUT", outputs)
    inspect_model_architecture(model)

    print("\n=== FEATURE RECOMMENDATION ===")
    candidate_names = [
        "image_hidden_states",
        "vision_hidden_states",
        "hidden_states",
    ]
    selected = None
    for name in candidate_names:
        if hasattr(outputs, name) and getattr(outputs, name) is not None:
            selected = name
            break

    if selected is not None:
        val = getattr(outputs, selected)
        print(f"Best cached feature candidate: outputs.{selected}")
        summarize_tensor(selected, val)
    else:
        print("No direct image/visual hidden state was exposed by the model.forward() output.")
        print("Recommended feature hook: inspect the vision backbone and/or multimodal projector before generation, e.g. model.model.visual or model.model.vision_tower.")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
