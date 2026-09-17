"""
build_model.py
Loads your pretrained LLaVA-Pythia checkpoint in plain text-generation mode.
No LoRA, no policy head -- this is a straight backbone swap.
"""
from llava_pythia.model.language_model.pythia.configuration_llava_pythia import LlavaPythiaConfig
from llava_pythia.model.language_model.pythia.llava_pythia import LlavaPythiaForCausalLM

def build_llava_pythia(pretrained_path):
    config = LlavaPythiaConfig.from_pretrained(
        pretrained_path,
        trust_remote_code=True
    )

    # Normal language-generation mode.
    config.action_head_type = "fc"

    # Required by the current LLaVA-Pythia implementation,
    # even though it is not actually used by the FC head.
    config.action_dim = 0

    model = LlavaPythiaForCausalLM.from_pretrained(
        pretrained_path,
        config=config
    )

    return model