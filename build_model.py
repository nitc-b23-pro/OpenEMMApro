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

    # FIXED (loss=nan root cause): confirmed via a [NaN DEBUG] instrumentation
    # pass that ~108 embed_out parameters are ALREADY non-finite immediately
    # after from_pretrained() returns -- before LoRA, before the fp32 upcast,
    # before any forward pass -- and every single one is an nn.Conv1d bias or
    # an nn.GroupNorm weight/bias. Not one nn.Linear (combine, cond_encoder,
    # diffusion_step_encoder) was affected.
    #
    # Root cause: from_pretrained()'s low-memory loading path allocates every
    # parameter as raw, uninitialized memory (torch.empty()) and only fills in
    # real values for (a) keys found in the checkpoint, and (b) "missing"
    # keys whose module type GPTNeoXPreTrainedModel._init_weights() knows how
    # to handle -- which is only nn.Linear, nn.Embedding, and nn.LayerNorm
    # (confirmed by reading transformers' actual source). embed_out
    # (ConditionalUnet1D) is a brand-new module type this base class has
    # never seen, built almost entirely out of nn.Conv1d/nn.GroupNorm --
    # neither is in that isinstance() chain, so their tensors were simply
    # never written to: raw uninitialized memory, which very often decodes
    # as NaN/Inf when read as floats. This is exactly why the corruption was
    # 100%, deterministic, and present on step 0 before any training ever
    # happened -- it was never a computed value, just garbage bits nothing
    # had written real numbers into.
    #
    # Fix: explicitly reset every Conv1d/ConvTranspose1d/GroupNorm inside the
    # newly-built head using PyTorch's own standard reset_parameters() --
    # the exact call every one of these layers normally runs in its own
    # __init__, which from_pretrained's fast-init path skipped because it
    # doesn't recognize these module types. This must happen BEFORE
    # get_peft_model() below, so the clean values are what gets deep-copied
    # into modules_to_save's trainable copy.
    reinitialized = []
    for module_name, submodule in model.embed_out.named_modules():
        if isinstance(submodule, (torch.nn.Conv1d, torch.nn.ConvTranspose1d, torch.nn.GroupNorm)):
            submodule.reset_parameters()
            reinitialized.append(module_name)
    print(f"[build_openemma_tinyvla] reset_parameters() re-ran on {len(reinitialized)} "
          f"Conv1d/ConvTranspose1d/GroupNorm submodules inside embed_out (fixing the "
          f"uninitialized-memory NaN bug from from_pretrained's fast-init path).")

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

        # ADDED (loss=nan investigation, checkpoint #0): every training step so
        # far has produced a noise_pred tensor that is 100% NaN -- not a few
        # extreme values, ALL of them, from the very first forward pass before
        # any training has happened. That pattern (unconditional, from step 0,
        # regardless of how ordinary the input data is) smells like the
        # diffusion head's weights are ALREADY corrupted right out of
        # __init__/post_init(), before a single batch is ever seen -- rather
        # than a computation that corrupts otherwise-clean weights during the
        # forward pass. This checks that directly, once, at model-build time:
        # if any embed_out/proj_to_action parameter is already NaN/Inf here,
        # we've found the actual bug (almost certainly in how
        # LlavaPythiaForCausalLM.post_init() re-initializes this module, since
        # post_init() runs its generic HF weight-init AFTER ConditionalUnet1D's
        # own __init__ already gave every layer sane values). If nothing
        # prints here, the weights start clean and the corruption is
        # happening during the forward pass itself instead.
        bad_params = []
        for name, param in model.named_parameters():
            if ("embed_out" in name or "proj_to_action" in name) and not torch.isfinite(param).all():
                bad_params.append(name)
        if bad_params:
            print(f"[NaN DEBUG] {len(bad_params)} embed_out/proj_to_action parameter(s) "
                  f"are ALREADY non-finite immediately after model construction "
                  f"(before any forward pass): {bad_params}")
        else:
            print("[NaN DEBUG] all embed_out/proj_to_action parameters are finite "
                  "immediately after model construction.")
    else:
        # Loading a checkpoint YOU already trained: load the SAVED LoRA
        # weights + SAVED diffusion head weights from trained_checkpoint_path
        # instead of re-randomizing them.
        model = PeftModel.from_pretrained(model, trained_checkpoint_path, is_trainable=False)

        # FIXED (inference dtype mismatch): the base model above is loaded with
        # torch_dtype=torch.float16, so PeftModel.from_pretrained() allocates the
        # LoRA / embed_out / proj_to_action destination tensors as fp16 BEFORE it
        # copies the checkpoint's saved values into them. Since training upcast
        # these same parameters to fp32 (see the `if trained_checkpoint_path is
        # None:` branch above), the saved checkpoint values are fp32 -- but the
        # in-place copy into fp16-allocated destination tensors silently downcasts
        # them back to fp16. is_trainable=False also sets requires_grad=False on
        # every parameter here, so (unlike the fresh-training branch) we can't
        # filter by requires_grad -- we match by parameter name instead. Without
        # this, forward_diffusion_head()'s unconditional `hidden_states.float()`
        # produces an fp32 input that hits norm_after_pool's still-fp16
        # LayerNorm weight/bias, crashing with "expected scalar type Float but
        # found Half" during inference (mirror image of the training-side dtype
        # bugs fixed earlier).
        upcasted = []
        for name, param in model.named_parameters():
            if "lora_" in name or "embed_out" in name or "proj_to_action" in name:
                param.data = param.data.float()
                upcasted.append(name)
        print(f"[build_openemma_tinyvla] upcast {len(upcasted)} lora_/embed_out/"
              f"proj_to_action parameter(s) back to fp32 after loading the trained "
              f"checkpoint (fixing the inference-time fp16/fp32 LayerNorm dtype "
              f"mismatch).")

    return model
