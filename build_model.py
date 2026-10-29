"""
build_model.py
Qwen2.5-VL-3B-Instruct backbone + LoRA + a diffusion policy head for
OpenEMMA local trajectory prediction. The head lives outside the LoRA-
wrapped backbone as a sibling module (OpenEMMAPolicy), so its weights are
saved/loaded separately from the backbone's LoRA adapter.
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
from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

try:
    import bitsandbytes  # noqa: F401
    from peft import prepare_model_for_kbit_training
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False

# Load the frozen backbone in 4-bit (QLoRA-style) when bitsandbytes is
# available -- the backbone's own weights never need to update, so the
# precision loss only affects the (frozen) forward pass, not what LoRA
# learns. On a GPU with ~15GB this is the difference between fitting and
# not: a 3B-param fp16 backbone alone is already ~7GB, before any LoRA
# deltas, optimizer state, or activations. Set OPENEMMA_USE_4BIT=0 to force
# fp16 loading instead (e.g. for debugging numerics).
USE_4BIT = BNB_AVAILABLE and os.environ.get("OPENEMMA_USE_4BIT", "1") != "0"

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
    """
    all_vals = np.concatenate(
        [np.concatenate([s["obs"], s["fut"]], axis=0) for s in dataset.samples], axis=0
    )  # (N, ACTION_DIM) pooled [speed, curvature*100] pairs
    low = np.percentile(all_vals, low_pct, axis=0).astype(np.float32)
    high = np.percentile(all_vals, high_pct, axis=0).astype(np.float32)
    return low, high


def find_linear_names(model):
    """
    nn.Linear names in the language-model decoder only. The vision tower
    (`visual.*`) is deliberately excluded: none of its own weights need to
    update for this task, and keeping it out of LoRA means none of its
    inputs require grad, so PyTorch skips building a backward graph
    through it entirely instead of retaining activations for every ViT
    layer. That's the difference between fitting on a 15GB GPU and not.
    """
    return sorted(
        name for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and not name.endswith("lm_head")
        and "visual" not in name.split(".")
    )


class OpenEMMAPolicy(nn.Module):
    """LoRA-wrapped Qwen2.5-VL backbone + a diffusion policy head."""

    def __init__(self, backbone, hidden_size, backbone_is_quantized=False):
        super().__init__()
        self.backbone = backbone
        # A 4-bit-quantized backbone is already placed on its device via
        # device_map at load time; transformers raises if you then call
        # .cuda()/.to() on it again. So when quantized, .cuda()/.to() only
        # move this policy's own (unquantized) diffusion head.
        self._backbone_is_quantized = backbone_is_quantized
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
        # scale: every sample gets crushed into [-1, 1] regardless of what
        # it should have been, independent of image/history conditioning.
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

    def cuda(self, device=None):
        if self._backbone_is_quantized:
            self.embed_out = self.embed_out.cuda(device)
            return self
        return super().cuda(device)

    def to(self, *args, **kwargs):
        if self._backbone_is_quantized:
            self.embed_out = self.embed_out.to(*args, **kwargs)
            return self
        return super().to(*args, **kwargs)

    def _hidden_states(self, input_ids, attention_mask, pixel_values, image_grid_thw):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
            out = self.backbone(
                input_ids=input_ids, attention_mask=attention_mask,
                pixel_values=pixel_values, image_grid_thw=image_grid_thw,
                output_hidden_states=True, return_dict=True, use_cache=False,
            )
        return out.hidden_states[-1]

    def forward(self, input_ids, attention_mask=None, pixel_values=None, image_grid_thw=None,
                states=None, actions=None, eval=False, **_unused):
        if not self._stats_set:
            raise RuntimeError(
                "OpenEMMAPolicy has no normalization stats -- call set_action_stats(low, high) "
                "right after building a fresh model (train_openemma.py does this once, right after "
                "constructing the dataset), or load a checkpoint through build_openemma(), which "
                "restores them from action_stats.pt."
            )
        hidden_states = self._hidden_states(input_ids, attention_mask, pixel_values, image_grid_thw).float()

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
    load_kwargs = dict(
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        ),
        device_map={"": 0} if torch.cuda.is_available() else "cpu",
    ) if USE_4BIT else dict(torch_dtype=torch.float16)

    try:
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_path, attn_implementation="sdpa", **load_kwargs,
        )
    except (TypeError, ValueError):
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrained_path, **load_kwargs)


def build_openemma(pretrained_path, trained_checkpoint_path=None):
    """
    Fresh training start: build_openemma(pretrained_path) -> brand-new
    LoRA adapters + brand-new diffusion head.

    Loading a trained checkpoint: build_openemma(pretrained_path,
    trained_checkpoint_path=...) -> loads the saved LoRA + diffusion head
    weights on top of the same base backbone.
    """
    backbone = _load_backbone(pretrained_path)
    if USE_4BIT:
        backbone = prepare_model_for_kbit_training(backbone, use_gradient_checkpointing=False)

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
        return OpenEMMAPolicy(backbone, hidden_size, backbone_is_quantized=USE_4BIT)

    backbone = PeftModel.from_pretrained(backbone, trained_checkpoint_path, is_trainable=False)
    for name, p in backbone.named_parameters():
        if "lora_" in name:
            p.data = p.data.float()

    model = OpenEMMAPolicy(backbone, hidden_size, backbone_is_quantized=USE_4BIT)
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
