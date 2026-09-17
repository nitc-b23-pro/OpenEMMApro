"""
build_model.py
Loads your pretrained LLaVA-Pythia checkpoint in plain text-generation mode.
No LoRA, no policy head -- this is a straight backbone swap.
"""
import torch
from llava_pythia.model.language_model.pythia.configuration_llava_pythia import LlavaPythiaConfig
from llava_pythia.model.language_model.pythia.llava_pythia import LlavaPythiaForCausalLM

def build_llava_pythia(pretrained_path, torch_dtype=torch.float16):
    config = LlavaPythiaConfig.from_pretrained(
        pretrained_path,
        trust_remote_code=True
    )

    # Normal language-generation mode.
    config.action_head_type = "fc"

    # Required by the current LLaVA-Pythia implementation,
    # even though it is not actually used by the FC head.
    config.action_dim = 0

    # Load in fp16 by default -- on a 14-15GB Kaggle T4, an H (1.3B) checkpoint
    # plus the CLIP vision tower plus YOLO3D easily eats the whole card in fp32.
    # Halving the weight footprint here is the cheapest way to buy back headroom
    # without touching the manual, no-KV-cache generation loop.
    model = LlavaPythiaForCausalLM.from_pretrained(
        pretrained_path,
        config=config,
        torch_dtype=torch_dtype
    )

    return model