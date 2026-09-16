"""
build_model.py
Loads your pretrained LLaVA-Pythia checkpoint (S, B, or H — no head yet),
attaches the diffusion policy head, and wraps it with LoRA.
"""
from llava_pythia.model.language_model.pythia.configuration_llava_pythia import LlavaPythiaConfig
from llava_pythia.model.language_model.pythia.llava_pythia import LlavaPythiaForCausalLM
from llava_pythia.llava_pythia_utils import find_all_linear_names
from peft import LoraConfig, get_peft_model

def build_openemma_tinyvla(pretrained_path):
    # 1. Load your existing pretrained LLaVA-Pythia's settings...
    config = LlavaPythiaConfig.from_pretrained(pretrained_path, trust_remote_code=True)

    # 2. ...and tell it: "you now also have a diffusion head, sized for OUR problem":
    config.action_head_type = "droid_diffusion"   # use the diffusion "sculptor" head
    config.action_dim = 2                          # each output step = [speed, curvature]
    config.state_dim  = 20                          # 10 history steps x 2 numbers, flattened
    config.chunk_size  = 10                          # predict 10 future steps
    config.concat = "None"                          # only ONE camera (front) -- no fusion needed

    # 3. Build the model. The vision tower + language model weights load from your
    #    checkpoint; the new head is randomly initialized (it didn't exist before).
    model = LlavaPythiaForCausalLM.from_pretrained(pretrained_path, config=config)

    # 4. Wrap it in LoRA: freeze almost everything, add small trainable "patches"
    #    to the vision tower ('vit') and language model ('llm').
    lora_config = LoraConfig(
        r=64, lora_alpha=256, lora_dropout=0.05, bias="none",
        target_modules=find_all_linear_names(model, print, lora_module="vit llm"),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # 5. The diffusion head is BRAND NEW -- LoRA doesn't touch it, so make sure
    #    it's fully trainable (not frozen).
    for name, param in model.named_parameters():
        if "embed_out" in name or "proj_to_action" in name:
            param.requires_grad = True

    return model