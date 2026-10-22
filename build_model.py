"""
build_model_qwen.py
Same diffusion-policy-head + LoRA recipe as build_model.py, swapped onto a
Qwen2.5-VL-3B-Instruct backbone instead of LLaVA-Pythia. Branch:
kaggle-openvlm.

Why a separate file instead of a flag in build_model.py
---------------------------------------------------------
LlavaPythiaForCausalLM (build_model.py's backbone) is a from-scratch HF
model class that bakes the diffusion head IN as a config-driven attribute
of the class itself (see llava_pythia.py's __init__: `if config.
action_head_type == 'droid_diffusion': self.embed_out = ConditionalUnet1D
(...)`). Qwen2_5_VLForConditionalGeneration is a real, unmodified
transformers class we are not going to subclass/vendor just to bolt a
head onto it the same way. Instead, this file keeps the diffusion head
OUTSIDE the HF model entirely, as a sibling nn.Module
(OpenEMMAQwenPolicy, below) that owns a LoRA-wrapped Qwen2_5_VL instance
plus the SAME policy_heads.models.ConditionalUnet1D used on the
LLaVA-Pythia backend (identical hyperparameters: action_dim=2,
state_dim=20, chunk_size=10 future steps) and runs the SAME DDIM
diffusion process (100 train timesteps, squaredcos_cap_v2, epsilon
prediction, 10 inference steps) copied from llava_pythia.py's
forward_diffusion_head.

This sidesteps two of build_model.py's three real bugs, for free:
  - No `modules_to_save` hack needed. PeftModel.save_pretrained() here
    only ever has to know about LoRA deltas on the Qwen backbone --
    embed_out/proj_to_action live outside that PeftModel and are saved/
    loaded by OpenEMMAQwenPolicy.save_pretrained()/the loader in
    build_openemma_qwen() below, via plain torch.save/load_state_dict.
  - No from_pretrained-fast-init uninitialized-memory NaN risk. embed_out
    is a bare `ConditionalUnet1D()` constructed directly in Python (see
    __init__ below) -- every Conv1d/GroupNorm/Linear inside it runs its
    own normal nn.Module __init__ the moment it's created, never routed
    through HF's `from_pretrained()` fast-init path at all (that path --
    and specifically GPTNeoXPreTrainedModel._init_weights() not knowing
    about Conv1d/GroupNorm -- is what corrupted LLaVA-Pythia's head; it
    is simply not in the call path here).

What IS still replicated here (same root cause, not backbone-specific):
  - Frozen base in fp16 (+ attn_implementation="sdpa", falling back to
    default attention if unsupported) to fit a ~3B-param VLM in a Kaggle
    GPU's memory budget; every TRAINABLE parameter (LoRA deltas + the
    diffusion head) explicitly upcast to fp32 right after creation, for
    the same optimizer-stability reason build_model.py upcasts
    LLaVA-Pythia's trainable parameters.
  - The inference-time dtype fix: loading a trained checkpoint allocates
    LoRA destination tensors as fp16 (matching the fp16 base) BEFORE
    copying the saved fp32 values in, silently downcasting them -- fixed
    the same way, by upcasting lora_* parameters back to fp32 right after
    PeftModel.from_pretrained() returns.
"""
# COMPAT SHIM (see build_model.py for the same shim + explanation): must
# run before `diffusers` is ever imported (policy_heads.models ->
# droid_unet_diffusion.py -> diffusers).
import huggingface_hub
if not hasattr(huggingface_hub, "cached_download"):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download

import os

import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from peft import LoraConfig, PeftModel, get_peft_model
from policy_heads.models import ConditionalUnet1D
from transformers import Qwen2_5_VLForConditionalGeneration

ACTION_DIM = 2    # [speed, curvature] per step -- same as the LLaVA-Pythia backend
STATE_DIM = 20    # 10 history steps x 2 numbers, flattened -- same as the LLaVA-Pythia backend
CHUNK_SIZE = 10   # predict 10 future steps -- same as the LLaVA-Pythia backend


def find_all_linear_names_qwen(model):
    """
    Qwen2.5-VL analogue of llava_pythia_utils.find_all_linear_names, with
    the same "vit llm" policy build_model.py uses (both the vision tower
    and the language model get LoRA-adapted) -- for Qwen2.5-VL that's
    every nn.Linear in both `model.visual` (the ViT + patch-merger) and
    the language decoder stack, which is everything EXCEPT the final
    `lm_head` (excluded for the same "needed for 16-bit" reason
    find_all_linear_names excludes it -- LoRA-adapting the tied output
    projection on a 151k-token vocab is also just not useful here, since
    this project never generates text with this model at all).
    """
    names = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name == "lm_head" or name.endswith(".lm_head"):
            continue
        names.append(name)
    return sorted(names)


class OpenEMMAQwenPolicy(nn.Module):
    """
    LoRA-wrapped Qwen2.5-VL-3B-Instruct backbone + the diffusion policy
    head, as two sibling attributes of a plain nn.Module (see this file's
    docstring for why, vs. the LLaVA-Pythia backend's bolt-it-onto-the-
    HF-class design). Plain nn.Module also means .cuda()/.eval()/.train()/
    .named_parameters() all come for free -- only the HF-PreTrainedModel-
    style methods train_openemma_qwen.py calls (gradient_checkpointing_
    enable, enable_input_require_grads) need an explicit passthrough.
    """

    def __init__(self, qwen, hidden_size):
        super().__init__()
        self.qwen = qwen   # Qwen2_5_VLForConditionalGeneration, LoRA-wrapped (get_peft_model) or loaded (PeftModel.from_pretrained)
        self.hidden_size = hidden_size
        self.action_dim = ACTION_DIM
        self.num_queries = CHUNK_SIZE
        self.noise_samples = 1
        self.num_inference_timesteps = 10

        self.proj_to_action = nn.Identity()
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=100,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="epsilon",
        )
        # (B, L, hidden_size) global_cond in, (B, chunk_size, action_dim) out --
        # identical construction to build_model.py's embed_out, just with
        # Qwen's hidden_size (2048 for the 3B checkpoint) instead of
        # LLaVA-Pythia's.
        self.embed_out = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=hidden_size,
            state_dim=STATE_DIM,
        )

    # -- HF-PreTrainedModel-style passthroughs train_openemma_qwen.py calls --
    def gradient_checkpointing_enable(self, **kwargs):
        self.qwen.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self):
        self.qwen.enable_input_require_grads()

    def get_backbone_hidden_states(self, input_ids, attention_mask, pixel_values, image_grid_thw):
        """
        One Qwen2.5-VL forward pass, under the same fp16 autocast
        llava_pythia.py's forward() uses around its own backbone call
        (same rationale: the frozen backbone is loaded hard-cast fp16 for
        memory; autocast keeps LayerNorm/softmax-like ops numerically
        safe without needing fp32 "master" weights). Returns the LAST
        decoder layer's hidden states, (B, L, hidden_size) -- the Qwen
        analogue of llava_pythia.py's `hidden_states = outputs[0]`.

        Deliberately calls the FULL Qwen2_5_VLForConditionalGeneration.
        forward() (with output_hidden_states=True) rather than reaching
        into its internals (e.g. calling `.model(...)` directly): the
        image-token/embedding fusion (splicing the ViT's patch features
        into the text embedding sequence at the <image> token positions)
        happens inside THIS class's own forward(), not in a separate
        "prepare inputs" step the way LLaVA's prepare_inputs_labels_for_
        multimodal is -- going through the public forward() is the only
        way to get that fusion for free and stay correct across
        transformers versions, instead of re-deriving Qwen's own
        rope/mrope + scatter logic here.
        """
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
            outputs = self.qwen(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
        return outputs.hidden_states[-1]

    def forward_diffusion_head(self, actions, hidden_states, states, is_pad):
        """
        Byte-for-byte the same diffusion process as llava_pythia.py's
        forward_diffusion_head (same scheduler config, same noise/
        timestep sampling, same loss) -- copied rather than imported
        since this class does not subclass LlavaPythiaForCausalLM, and
        the two backends are meant to stay independently editable.
        """
        hidden_states = hidden_states.float()

        if actions is not None:   # training time
            B = actions.size(0)
            actions = actions[:, :self.num_queries]
            is_pad = is_pad[:, :self.num_queries]
            num_noise_samples = self.noise_samples

            noise = torch.randn([num_noise_samples] + list(actions.shape), device=actions.device, dtype=actions.dtype)
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, (B,), device=actions.device
            ).long()

            noisy_actions = torch.cat([
                self.noise_scheduler.add_noise(actions, noise[i], timesteps) for i in range(len(noise))
            ], dim=0)
            noisy_actions = noisy_actions.to(dtype=actions.dtype)
            assert hidden_states.ndim == 3

            hidden_states = hidden_states.repeat(num_noise_samples, 1, 1)
            timesteps_r = timesteps.repeat(num_noise_samples)
            is_pad = is_pad.repeat(num_noise_samples, 1)
            states = states.repeat(num_noise_samples, 1)

            noise_pred = self.embed_out(noisy_actions, timesteps_r, global_cond=hidden_states, states=states)

            noise = noise.view(noise.size(0) * noise.size(1), *noise.size()[2:])
            loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction="none")
            loss = (loss * ~is_pad.unsqueeze(-1)).mean()
            return {"loss": loss}
        else:   # inference time
            B = 1
            Tp = self.num_queries
            action_dim = self.action_dim

            noisy_action = torch.randn((B, Tp, action_dim)).cuda()
            naction = noisy_action.to(dtype=hidden_states.dtype)
            self.noise_scheduler.set_timesteps(self.num_inference_timesteps)

            for k in self.noise_scheduler.timesteps:
                noise_pred = self.embed_out(naction, k, global_cond=hidden_states, states=states)
                naction = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample

            return naction

    def forward(self, input_ids, attention_mask=None, pixel_values=None, image_grid_thw=None,
                states=None, actions=None, is_pad=None, eval=False, **_unused_processor_keys):
        # **_unused_processor_keys: defensive catch-all for any extra key the
        # processor's output dict might carry (e.g. a video-related key if a
        # future transformers version adds one even for image-only input) --
        # train_openemma_qwen.py/main_qwen.py call this as model(**inputs,
        # states=..., ...), and inputs is whatever AutoProcessor.__call__
        # returns, not a fixed, version-pinned key set.
        hidden_states = self.get_backbone_hidden_states(input_ids, attention_mask, pixel_values, image_grid_thw)
        if not eval:
            return self.forward_diffusion_head(actions, hidden_states, states, is_pad)
        else:
            return self.forward_diffusion_head(None, hidden_states, states, is_pad)

    def save_pretrained(self, path):
        """
        Mirrors model.save_pretrained(...) being called every epoch in
        train_openemma_tinyvla.py/train_openemma_qwen.py. Two pieces,
        saved separately since they're separate objects here (see this
        file's docstring): the PEFT adapter (LoRA deltas on self.qwen,
        via its own save_pretrained -- writes adapter_config.json +
        adapter_model.safetensors, nothing else needed since there's no
        modules_to_save on this side), and the diffusion head's plain
        state_dict (embed_out + proj_to_action) as one extra file.
        """
        os.makedirs(path, exist_ok=True)
        self.qwen.save_pretrained(path)
        torch.save(
            {"embed_out": self.embed_out.state_dict(), "proj_to_action": self.proj_to_action.state_dict()},
            os.path.join(path, "diffusion_head.pt"),
        )


def build_openemma_qwen(pretrained_path, trained_checkpoint_path=None):
    """
    Two use cases, one function -- same contract as build_openemma_tinyvla:
      - Fresh training start: build_openemma_qwen(pretrained_path)
        -> brand-new LoRA adapters + brand-new diffusion head.
      - Loading a checkpoint YOU already trained (inference, or resuming):
        build_openemma_qwen(pretrained_path, trained_checkpoint_path=...)
        -> loads the SAVED LoRA weights + SAVED diffusion head weights
           from trained_checkpoint_path, on top of the same base VLM.
    """
    try:
        qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_path, torch_dtype=torch.float16, attn_implementation="sdpa",
        )
    except (TypeError, ValueError) as e:
        print(f"[build_openemma_qwen] attn_implementation='sdpa' not supported here ({e}); "
              f"falling back to default attention (still fp16).")
        qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrained_path, torch_dtype=torch.float16)

    hidden_size = qwen.config.hidden_size   # 2048 for Qwen2.5-VL-3B-Instruct

    lora_config = LoraConfig(
        r=64, lora_alpha=256, lora_dropout=0.05, bias="none",
        target_modules=find_all_linear_names_qwen(qwen),
        task_type="CAUSAL_LM",
    )

    if trained_checkpoint_path is None:
        qwen = get_peft_model(qwen, lora_config)

        # Upcast every TRAINABLE parameter (LoRA deltas) to fp32 -- same
        # optimizer-stability reason as build_model.py. The diffusion head
        # built inside OpenEMMAQwenPolicy below is constructed fresh, in
        # plain fp32 (nn.Module's default dtype), so it needs no separate
        # upcast here the way LLaVA-Pythia's embed_out (materialized via
        # from_pretrained) did.
        for name, param in qwen.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        model = OpenEMMAQwenPolicy(qwen, hidden_size)

        bad = [n for n, p in model.embed_out.named_parameters() if not torch.isfinite(p).all()]
        if bad:
            # Not expected to ever fire here (see this file's docstring --
            # embed_out is never routed through from_pretrained's fast-init
            # path), kept as cheap insurance against the exact bug class
            # that DID hit the LLaVA-Pythia backend.
            print(f"[build_openemma_qwen] WARNING: {len(bad)} embed_out parameter(s) "
                  f"non-finite immediately after construction: {bad}")
    else:
        qwen = PeftModel.from_pretrained(qwen, trained_checkpoint_path, is_trainable=False)

        # FIXED (inference dtype mismatch, same root cause as build_model.py):
        # the base model is fp16, so PeftModel.from_pretrained() allocates the
        # LoRA destination tensors as fp16 before copying the checkpoint's
        # fp32-saved values in, silently downcasting them. Upcast back to fp32
        # by name since is_trainable=False means every param's requires_grad
        # is False here, so we can't filter on that the way the fresh-training
        # branch does.
        upcasted = []
        for name, param in qwen.named_parameters():
            if "lora_" in name:
                param.data = param.data.float()
                upcasted.append(name)
        print(f"[build_openemma_qwen] upcast {len(upcasted)} lora_ parameter(s) back to fp32 "
              f"after loading the trained checkpoint.")

        model = OpenEMMAQwenPolicy(qwen, hidden_size)

        head_ckpt_path = os.path.join(trained_checkpoint_path, "diffusion_head.pt")
        head_ckpt = torch.load(head_ckpt_path, map_location="cpu")
        model.embed_out.load_state_dict(head_ckpt["embed_out"])
        model.proj_to_action.load_state_dict(head_ckpt["proj_to_action"])
        model.embed_out = model.embed_out.float()

    return model
