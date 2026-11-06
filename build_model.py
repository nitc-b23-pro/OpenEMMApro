"""
build_model.py
LLaVA-Pythia backbone + LoRA + a diffusion policy head for OpenEMMA local
trajectory prediction. The head lives outside the LoRA-wrapped backbone as
a sibling module (OpenEMMAPolicy), so its weights are saved/loaded
separately from the backbone's LoRA adapter.

BACKBONE-SWAP NOTE (kaggle-lp branch): this mirrors the Qwen2.5-VL branch's
build_model.py almost line for line -- OpenEMMAPolicy (the diffusion head,
DDIMScheduler config, and critically the percentile-based action/state
normalization fix from commit 69e604b / claude/diffusion-head-shortcut-
learning-diagnosis.md) is copied UNCHANGED. Only the backbone loading
(_load_backbone), LoRA target-module discovery (find_linear_names), and
OpenEMMAPolicy._hidden_states()/forward()'s call signature differ, because
LLaVA-Pythia's own forward() takes `images` (a single pre-processed image
tensor) instead of Qwen's `pixel_values`/`image_grid_thw` pair.

WHY THIS BACKBONE IS LOADED WITH action_head_type="external": the ORIGINAL
LLaVA-Pythia pipeline (still on branch kaggle-tiny-ft) builds its diffusion
head INSIDE LlavaPythiaForCausalLM itself (action_head_type="droid_diffusion"),
diffusing RAW, unnormalized [speed, curvature*100] values through a
DDIMScheduler(clip_sample=True) that silently clamps every sample to
[-1, 1]. That is the exact bug diagnosed and fixed on the Qwen branch
(commit 69e604b) -- loss looks fine, but ADE plateaus near-identically
regardless of backbone or config, because ~30%+ of predictions were pinned
to the clip boundary. Reviving the LLaVA-Pythia backbone from before that
fix existed would silently reintroduce it. Instead, llava_pythia.py (the
vendored model file) has been patched with a small "external" action-head
sentinel: LlavaPythiaForCausalLM is used purely as a hidden-states
producer (embed_out/proj_to_action become inert nn.Identity() placeholders
inside it, never trained or called), and OpenEMMAPolicy below -- the SAME
already-proven, already-normalized diffusion head used on the Qwen branch
-- owns the actual diffusion process. See llava_pythia.py's __init__ and
forward() for the two small patched branches.
"""
import os

import numpy as np

import huggingface_hub
# diffusers' top-level import chain reaches for a helper huggingface_hub
# has since removed; this project never uses the feature that needs it.
if not hasattr(huggingface_hub, "cached_download"):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download

import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from peft import LoraConfig, PeftModel, get_peft_model
from policy_heads.models import ConditionalUnet1D

from llava_pythia.model.language_model.pythia.configuration_llava_pythia import LlavaPythiaConfig
from llava_pythia.model.language_model.pythia.llava_pythia import LlavaPythiaForCausalLM
from llava_pythia.llava_pythia_utils import find_all_linear_names

ACTION_DIM = 2    # [speed, curvature] per predicted step
STATE_DIM = 20    # 10 history steps x 2 numbers, flattened
CHUNK_SIZE = 10   # predicted future steps

# Robust percentile bounds (not literal min/max) for normalizing
# [speed, curvature*100] to ~[-1, 1] before diffusing. Percentiles rather
# than min/max because EstimateCurvatureFromTrajectory (utils.py) divides by
# near-zero path length for a near-stationary vehicle and can produce huge
# curvature outliers (observed directly: curvature*100 = +-200 for a frame
# where speed ~ 0) that would otherwise dominate a literal min/max and crush
# every ordinary turn into a sliver near zero after normalization.
ACTION_STATS_LOW_PCT = 1.0
ACTION_STATS_HIGH_PCT = 99.0


def compute_action_stats(dataset, low_pct=ACTION_STATS_LOW_PCT, high_pct=ACTION_STATS_HIGH_PCT):
    """
    Per-dimension (speed, curvature*100) normalization bounds computed from
    a training dataset (an OpenEMMADataset, or anything with a `.samples`
    list of dicts holding "obs"/"fut" arrays). `states` and `actions` are
    the identical [speed, curvature*100] representation (see
    openemma_dataset.py), so one set of bounds -- computed by pooling every
    observed AND future step across the whole training set -- normalizes
    both consistently. Returns (low, high), each shape (ACTION_DIM,).

    Backbone-agnostic -- copied verbatim from the Qwen branch.
    """
    all_vals = np.concatenate(
        [np.concatenate([s["obs"], s["fut"]], axis=0) for s in dataset.samples], axis=0
    )  # (N, ACTION_DIM) pooled [speed, curvature*100] pairs
    low = np.percentile(all_vals, low_pct, axis=0).astype(np.float32)
    high = np.percentile(all_vals, high_pct, axis=0).astype(np.float32)
    return low, high


def find_linear_names(model):
    """
    LoRA target-module names for the LLaVA-Pythia backbone: both the ViT
    and the Pythia/GPT-NeoX decoder's nn.Linear layers. lora_module=
    "vit llm" matches the original LLaVA-Pythia project's own setting (see
    claude/project-context.md: "LoRA patches at both the ViT and the LLM
    modules") -- unlike the Qwen branch, which deliberately EXCLUDES the
    vision tower from LoRA to fit a 3B model's memory budget (see Qwen
    build_model.py's find_linear_names docstring). LLaVA-Pythia is ~1.3B
    params and fits fp16 comfortably on a 14-16GB GPU, so there's no need
    to make that same tradeoff here -- this matches the proven original
    LLaVA-Pythia recipe instead. find_all_linear_names() already excludes
    embed_out/proj_to_action/mm_projector/vision_resampler/lm_head itself.
    """
    return find_all_linear_names(model, print, lora_module="vit llm")


class OpenEMMAPolicy(nn.Module):
    """LoRA-wrapped LLaVA-Pythia backbone + a diffusion policy head.

    Identical in substance to the Qwen2.5-VL branch's OpenEMMAPolicy --
    same ConditionalUnet1D head, same DDIMScheduler config, same
    normalize()/denormalize()/forward()/save_pretrained() logic (commit
    69e604b). Only _hidden_states() and forward()'s call signature differ,
    to match LLaVA-Pythia's own forward(), which takes `images` instead of
    Qwen's `pixel_values`/`image_grid_thw` pair. There is no 4-bit-
    quantization branch here (unlike the Qwen version): LLaVA-Pythia is
    small enough (~1.3B params, ~2.6GB fp16) that plain fp16 loading is
    the proven, sufficient approach -- matching how this backbone was
    always loaded in the original (pre-Qwen) pipeline.
    """

    def __init__(self, backbone, hidden_size):
        super().__init__()
        self.backbone = backbone
        self.action_dim = ACTION_DIM
        self.num_queries = CHUNK_SIZE
        self.num_inference_timesteps = 10

        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=100,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            clip_sample_range=1.0,
            set_alpha_to_one=True,
            prediction_type="epsilon",
        )
        self.embed_out = ConditionalUnet1D(
            input_dim=ACTION_DIM, global_cond_dim=hidden_size, state_dim=STATE_DIM
        )

        # clip_sample=True clamps every DDIM step's sample estimate to
        # +-clip_sample_range (1.0) -- a standard diffusion-policy
        # assumption that the data being diffused is already normalized to
        # roughly that range. `states`/`actions` here are raw physical
        # units ([speed in m/s, curvature*100], curvature*100 alone can run
        # to +-200+), so without normalizing first, that clamp isn't a
        # helpful prior, it's a hard ceiling miles below the real target
        # scale -- see this file's module docstring for the full history.
        # set_action_stats() (called by train_openemma.py right after
        # building the dataset, or by build_openemma() when loading a
        # checkpoint) must be called before forward() will run.
        self.register_buffer("action_low", torch.zeros(ACTION_DIM), persistent=False)
        self.register_buffer("action_high", torch.ones(ACTION_DIM), persistent=False)
        self._stats_set = False

    def set_action_stats(self, low, high):
        self.action_low = torch.as_tensor(low, dtype=torch.float32)
        self.action_high = torch.as_tensor(high, dtype=torch.float32)
        self._stats_set = True

    def _normalize(self, x):
        """Physical [speed, curvature*100] units -> ~[-1, 1], per dimension."""
        low = self.action_low.to(device=x.device, dtype=x.dtype)
        high = self.action_high.to(device=x.device, dtype=x.dtype)
        span = (high - low).clamp_min(1e-6)
        return ((x - low) / span) * 2 - 1

    def _denormalize(self, x):
        """~[-1, 1] -> physical [speed, curvature*100] units, per dimension."""
        low = self.action_low.to(device=x.device, dtype=x.dtype)
        high = self.action_high.to(device=x.device, dtype=x.dtype)
        span = (high - low).clamp_min(1e-6)
        return (x + 1) / 2 * span + low

    def gradient_checkpointing_enable(self, **kwargs):
        kwargs.setdefault("gradient_checkpointing_kwargs", {"use_reentrant": False})
        self.backbone.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self):
        self.backbone.enable_input_require_grads()

    def _hidden_states(self, input_ids, attention_mask, images):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
            out = self.backbone(
                input_ids=input_ids, attention_mask=attention_mask, images=images,
                output_hidden_states=True, return_dict=True, use_cache=False,
            )
        return out.hidden_states[-1]

    def forward(self, input_ids, attention_mask=None, images=None,
                states=None, actions=None, eval=False, **_unused):
        if not self._stats_set:
            raise RuntimeError(
                "OpenEMMAPolicy has no normalization stats -- call set_action_stats(low, high) "
                "right after building a fresh model (train_openemma.py does this once, right after "
                "constructing the dataset), or load a checkpoint through build_openemma(), which "
                "restores them from action_stats.pt."
            )
        hidden_states = self._hidden_states(input_ids, attention_mask, images).float()

        # `states` arrives flattened (B, STATE_DIM) -- reshape to (B, steps,
        # ACTION_DIM) to normalize per-dimension like `actions`, then flatten
        # back since embed_out expects the flattened form.
        states_steps = states.reshape(states.size(0), -1, self.action_dim)
        cond_states = self._normalize(states_steps).reshape(states.size(0), -1)

        if not eval:
            actions = actions[:, :self.num_queries]
            norm_actions = self._normalize(actions)
            noise = torch.randn_like(norm_actions)
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, (actions.size(0),), device=actions.device
            ).long()
            noisy_actions = self.noise_scheduler.add_noise(norm_actions, noise, timesteps).to(norm_actions.dtype)

            noise_pred = self.embed_out(noisy_actions, timesteps, global_cond=hidden_states, states=cond_states)
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            return {"loss": loss}

        naction = torch.randn(
            (1, self.num_queries, self.action_dim), device=hidden_states.device, dtype=hidden_states.dtype
        )
        self.noise_scheduler.set_timesteps(self.num_inference_timesteps)
        for k in self.noise_scheduler.timesteps:
            noise_pred = self.embed_out(naction, k, global_cond=hidden_states, states=cond_states)
            naction = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample
        # naction is in normalized [-1, 1] space -- convert back to physical
        # [speed, curvature*100] units before handing it to callers, which
        # all expect (and always have expected) absolute units.
        return self._denormalize(naction)

    def save_pretrained(self, path):
        if not self._stats_set:
            raise RuntimeError(
                "Refusing to save a checkpoint with no normalization stats set -- call "
                "set_action_stats(...) before training (see train_openemma.py)."
            )
        os.makedirs(path, exist_ok=True)
        self.backbone.save_pretrained(path)
        torch.save(self.embed_out.state_dict(), os.path.join(path, "diffusion_head.pt"))
        torch.save(
            {"action_low": self.action_low.cpu(), "action_high": self.action_high.cpu()},
            os.path.join(path, "action_stats.pt"),
        )


def _load_backbone(pretrained_path):
    config = LlavaPythiaConfig.from_pretrained(pretrained_path, trust_remote_code=True)
    # action_head_type="external": see this file's module docstring and
    # llava_pythia.py's patched __init__/forward -- this backbone is used
    # purely as a hidden-states producer, never its own native
    # (unnormalized) diffusion head.
    config.action_head_type = "external"
    config.action_dim = ACTION_DIM
    config.state_dim = STATE_DIM
    config.chunk_size = CHUNK_SIZE
    config.concat = "None"   # single front camera only -- no multi-view fusion

    try:
        return LlavaPythiaForCausalLM.from_pretrained(
            pretrained_path, config=config, torch_dtype=torch.float16, attn_implementation="sdpa",
        )
    except (TypeError, ValueError) as e:
        print(f"[build_model] attn_implementation='sdpa' not supported here ({e}); "
              f"falling back to default attention (still fp16).")
        return LlavaPythiaForCausalLM.from_pretrained(pretrained_path, config=config, torch_dtype=torch.float16)


def build_openemma(pretrained_path, trained_checkpoint_path=None):
    """
    Fresh training start: build_openemma(pretrained_path) -> brand-new
    LoRA adapters + brand-new diffusion head.

    Loading a trained checkpoint: build_openemma(pretrained_path,
    trained_checkpoint_path=...) -> loads the saved LoRA + diffusion head
    weights on top of the same base backbone.

    NOTE: the original LLaVA-Pythia pipeline needed an explicit
    reset_parameters() pass here to fix a real from_pretrained()
    fast-init bug (Conv1d/GroupNorm params inside its native diffusion
    head left as uninitialized memory -- see claude/project-context.md,
    bug #6). That bug doesn't apply here: embed_out is loaded as an inert
    nn.Identity() placeholder (action_head_type="external", see
    llava_pythia.py), which has no such parameters, and the REAL diffusion
    head (OpenEMMAPolicy.embed_out, a ConditionalUnet1D) is constructed
    directly via its own __init__ below, never through from_pretrained()'s
    fast-init path -- the exact same reasoning that already let the Qwen
    branch skip this fix.
    """
    backbone = _load_backbone(pretrained_path)
    hidden_size = backbone.config.hidden_size

    lora_config = LoraConfig(
        r=64, lora_alpha=256, lora_dropout=0.05, bias="none",
        target_modules=find_linear_names(backbone), task_type="CAUSAL_LM",
    )

    if trained_checkpoint_path is None:
        backbone = get_peft_model(backbone, lora_config)
        backbone.print_trainable_parameters()
        for p in backbone.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        return OpenEMMAPolicy(backbone, hidden_size)

    backbone = PeftModel.from_pretrained(backbone, trained_checkpoint_path, is_trainable=False)
    for name, p in backbone.named_parameters():
        if "lora_" in name:
            p.data = p.data.float()

    model = OpenEMMAPolicy(backbone, hidden_size)
    head_state = torch.load(os.path.join(trained_checkpoint_path, "diffusion_head.pt"), map_location="cpu")
    model.embed_out.load_state_dict(head_state)
    model.embed_out = model.embed_out.float()

    stats_path = os.path.join(trained_checkpoint_path, "action_stats.pt")
    if not os.path.exists(stats_path):
        raise RuntimeError(
            f"{trained_checkpoint_path} has no action_stats.pt -- this checkpoint predates "
            "action/state normalization and cannot be evaluated correctly with the current code "
            "(forward() now normalizes before diffusing and denormalizes the output, which requires "
            "the exact bounds the checkpoint was trained with -- there's no safe default to fall "
            "back to). Re-train from scratch on the current code."
        )
    stats = torch.load(stats_path, map_location="cpu")
    model.set_action_stats(stats["action_low"], stats["action_high"])
    return model
