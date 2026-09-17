"""
build_model.py
Loads your pretrained LLaVA-Pythia checkpoint (S, B, or H — no head yet),
attaches the diffusion policy head, and wraps it with LoRA.

Two use cases, one function:
  - Fresh training start:
        build_openemma_tinyvla(base_pretrained_path)
    -> brand-new, randomly-initialized LoRA adapters + brand-new,
       randomly-initialized diffusion head, ready to train.
  - Loading a checkpoint YOU already trained (for inference, or to resume
    training):
        build_openemma_tinyvla(base_pretrained_path, trained_checkpoint_path=...)
    -> loads the SAVED LoRA weights + SAVED diffusion head weights from
       trained_checkpoint_path, on top of the same base VLM.

FIXED (two real bugs found while explaining the training/inference flow):

1. LoraConfig now sets modules_to_save=["embed_out", "proj_to_action"].
   Without this, PeftModel.save_pretrained() -- which is exactly what
   train_openemma_tinyvla.py calls every epoch -- only writes parameters
   whose name contains "lora_". The diffusion head (`embed_out`, the
   ConditionalUnet1D -- tens of millions of freshly-trained parameters)
   has no "lora_" in its name, so it was being SILENTLY DROPPED from every
   saved checkpoint, even though `requires_grad=True` was correctly set on
   it during training and it was genuinely being trained. modules_to_save
   tells PEFT to treat these modules as fully trainable+saved+reloadable,
   alongside the LoRA deltas. (find_all_linear_names() already excludes
   both names from the LoRA target list, so there's no conflict.)

2. main.py used to call this function with only ONE path and no way to
   load a previously-trained checkpoint: every call went through
   get_peft_model(...), which ALWAYS attaches brand-new, randomly
   initialized LoRA adapters and a brand-new, randomly initialized
   diffusion head. Calling it at inference time this way silently gave you
   an untrained LoRA + untrained head on top of the trained-looking base
   path name -- none of the actual training you did was ever loaded back
   in. The new `trained_checkpoint_path` argument fixes this: when given,
   it uses peft.PeftModel.from_pretrained(...) to load your ACTUAL saved
   adapter + head weights instead of re-randomizing them.
"""
# COMPAT SHIM (must run before anything imports `diffusers`): diffusers'
# top-level __init__.py unconditionally imports its dynamic-pipeline-loading
# module, which does `from huggingface_hub import cached_download` -- a
# function huggingface_hub has deprecated/removed. We never use that dynamic-
# pipeline feature (only DDPMScheduler / DDIMScheduler / EMAModel), so instead
# of chasing a "compatible" diffusers/huggingface_hub/transformers version
# triangle (risking a much bigger breaking change, like transformers 4.x->5.x),
# just alias the missing name back in before diffusers is ever imported. The
# import chain that triggers this is: build_model.py -> llava_pythia.py ->
# policy_heads.models -> droid_unet_diffusion.py -> diffusers.
import huggingface_hub
if not hasattr(huggingface_hub, "cached_download"):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download

from llava_pythia.model.language_model.pythia.configuration_llava_pythia import LlavaPythiaConfig
from llava_pythia.model.language_model.pythia.llava_pythia import LlavaPythiaForCausalLM
from llava_pythia.llava_pythia_utils import find_all_linear_names
from peft import LoraConfig, get_peft_model, PeftModel
import torch


def build_openemma_tinyvla(pretrained_path, trained_checkpoint_path=None):
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
    #
    #    FIXED (OOM risk on Kaggle's 14-16GB GPUs): this used to load with no
    #    torch_dtype (defaults to fp32, i.e. 4 bytes/param) and no attn_implementation
    #    (defaults to eager attention, which materializes the full attention-score
    #    matrix for every layer so it can be used in the backward pass). Together with
    #    LoRA's own gradients + AdamW optimizer state (2 extra copies) for every
    #    trainable parameter, that's exactly the shape of the "CUDA out of memory"
    #    crash you hit before. Loading the FROZEN base in fp16 halves its footprint
    #    (this is the large, dominant chunk of memory, since LoRA+the diffusion head
    #    are comparatively small); attn_implementation="sdpa" uses a fused/flash-style
    #    kernel that avoids materializing that full attention matrix. Trainable
    #    parameters (LoRA deltas + the diffusion head) are explicitly upcast back to
    #    fp32 right after they're created below -- fp16 gradients/optimizer state on
    #    the actively-trained parameters is a common source of unstable/NaN training,
    #    so only the FROZEN weights stay fp16.
    try:
        model = LlavaPythiaForCausalLM.from_pretrained(
            pretrained_path, config=config, torch_dtype=torch.float16, attn_implementation="sdpa"
        )
    except (TypeError, ValueError) as e:
        print(f"[build_openemma_tinyvla] attn_implementation='sdpa' not supported here ({e}); "
              f"falling back to default attention (still fp16).")
        model = LlavaPythiaForCausalLM.from_pretrained(
            pretrained_path, config=config, torch_dtype=torch.float16
        )

    # 4. Wrap it in LoRA: freeze almost everything, add small trainable "patches"
    #    to the vision tower ('vit') and language model ('llm'). modules_to_save
    #    makes sure the brand-new diffusion head is tracked (and saved/reloaded)
    #    by PEFT too, not just LoRA-adapted.
    lora_config = LoraConfig(
        r=64, lora_alpha=256, lora_dropout=0.05, bias="none",
        target_modules=find_all_linear_names(model, print, lora_module="vit llm"),
        modules_to_save=["embed_out", "proj_to_action"],
        task_type="CAUSAL_LM",
    )

    if trained_checkpoint_path is None:
        # Fresh training start: attach brand-new, randomly-initialized LoRA
        # adapters + diffusion head on top of the base VLM weights.
        model = get_peft_model(model, lora_config)

        # The diffusion head is BRAND NEW -- LoRA doesn't touch it, so make
        # sure it's fully trainable (not frozen).
        for name, param in model.named_parameters():
            if "embed_out" in name or "proj_to_action" in name:
                param.requires_grad = True

        # Upcast every TRAINABLE parameter (LoRA deltas + diffusion head) to fp32.
        # The frozen base stays fp16 (that's where the memory saving comes from);
        # only what's actually being optimized needs fp32 precision for stability.
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()
    else:
        # Loading a checkpoint YOU already trained: load the SAVED LoRA
        # weights + SAVED diffusion head weights from trained_checkpoint_path
        # instead of re-randomizing them.
        model = PeftModel.from_pretrained(model, trained_checkpoint_path, is_trainable=False)

    return model
