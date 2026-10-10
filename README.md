# OpenEMMApro — Open Source End to End Multimodal Model for Autonomous Driving

End-to-end Vision‑Language‑Action pipeline for autonomous-driving trajectory prediction. A fork/derivative of **OpenEMMA** (open-source reimplementation of Waymo's **EMMA** — arXiv:2412.15208) that replaces OpenEMMA's free-text chain-of-thought + regex-parsed numbers with a **diffusion policy head** bolted onto a **LLaVA‑Pythia** VLM backbone, fine-tuned with **LoRA** — so the model goes straight from (image, motion history) to (future speed, curvature) numbers with no text generation and no parsing step anywhere in the loop.

## Why this exists

EMMA's own framing (quoted directly from the paper, and the philosophy this project inherits) is that hand-designed interfaces between perception, planning, and control become a ceiling on generalization, and that splitting a driving stack into separately-trained modules means the system can never be jointly optimized toward the real objective. EMMA's fix is to put everything inside one function, `O_trajectory = G(T_intent, T_ego, V)`, with no module boundaries in between.

OpenEMMA reproduces that idea without EMMA's compute budget by doing pure prompting over an off-the-shelf MLLM — no fine-tuning — but it reintroduces exactly the kind of fragile hand-designed interface EMMA was trying to remove: the model writes a sentence containing the predicted numbers, and a regex scrapes them back out, with retry loops when the format doesn't parse. **OpenEMMA‑TinyVLA's whole point is removing that one remaining interface**: swap the output head from free-text generation to a diffusion policy head that reads the backbone's hidden states directly and emits exactly 10 × (speed, curvature) numbers, every time, with one backbone forward pass instead of an autoregressive text loop.

## What changed, relative to OpenEMMA

- **Removed** the YOLO3D bounding-box module (external, bolted-on, orthogonal to the actual prediction path — kept in the inference script only for optional visualization overlays).
- **Replaced the backbone** with LLaVA‑Pythia (1.3B params, from the TinyVLA project).
- **Added LoRA** (`peft`, r=64, α=256, dropout=0.05) on both the ViT and the Pythia/GPT‑NeoX decoder's linear layers.
- **Replaced the text-generation head with a Diffusion Policy Head** (`ConditionalUnet1D`, "droid_diffusion", from TinyVLA's `policy_heads`), trained full-parameter (not LoRA) — matching TinyVLA's own split of LoRA-tuned VLM + full-parameter diffusion head.
- **No chain-of-thought text anymore.** Since the model never generates text at all, it can't produce OpenEMMA's Scene Description / Critical Object / Intent Description sub-steps. Instead, a single shared prompt (`prompts.py`) is written to implicitly motivate that same reasoning inside the backbone's hidden states, without ever emitting it as text.
- **Route intent, re-added — but geometry-only.** An earlier VLM-generated "describe the maneuver" intent was replaced with a deterministic, map-free "turn-by-turn navigator" phrase (`route_intent.py` / `generate_intents.py`) derived purely from the ego vehicle's own recorded path curvature and speed — never from scene content, nearby agents, or critical-object reasoning. It reads like Google Maps announcing "next left turn," not a dashcam narrator describing traffic.

## Architecture

```
front camera image ──┐
                      ├─► LLaVA-Pythia (ViT + GPT-NeoX, LoRA r=64/α=256) ──► hidden states
10×[speed,curv] hist ─┘        │
prompt (+ optional intent) ────┘
                                                 │
                                                 ▼
                              ConditionalUnet1D diffusion policy head
                              (DDIM, 100 train / 10 inference steps,
                               FiLM-conditioned U-Net, state_dim=20,
                               action_dim=2, chunk_size=10)
                                                 │
                                                 ▼
                              10 × [speed, curvature] (future)
                                                 │
                                                 ▼
                    kinematic bicycle-model integration → (x, y) trajectory
```

- **Diffusion process**: `diffusers.DDIMScheduler`, 100 training timesteps, `squaredcos_cap_v2` schedule, `prediction_type="epsilon"`. Training adds noise to the GT action chunk at a random timestep and regresses the injected noise (MSE, masked by padding). Inference starts from pure Gaussian noise and denoises over 10 re-spaced DDIM steps.
- **Conditioning**: backbone hidden states are mean-pooled, layer-normed, concatenated with the 20-dim motion-history state vector, and linearly projected back to hidden size; this plus a sinusoidal timestep embedding conditions every residual block in the U-Net via FiLM.
- **Output → trajectory**: the head's (speed, curvature) outputs are integrated via the same bicycle-model math as upstream OpenEMMA (heading from ∫curvature·speed, then velocity components, then position) — never regressed directly as (x, y).
- **Training split**: LoRA params at `2e-4`, diffusion-head params at `2e-5` (two optimizer param groups — the head starts from nothing and needs a higher relative learning rate than the already-pretrained LoRA adapters, though see *Known limitations* below).

## Repository layout

| File | Role |
|---|---|
| `openemma_dataset.py` | nuScenes → PyTorch `Dataset`. Builds `(image_path, prompt, history[10,2], future[10,2])` samples; scene-level train/val split; optional precomputed-intent lookup; `iter_scene_motion()` is the single shared scene-walk + curvature/speed computation reused by every other script below. |
| `route_intent.py` | Deterministic, map-free "turn-by-turn navigator" intent classifier — pure curvature/speed geometry, no image, no VLM, no GPU. |
| `generate_intents.py` | Offline pass: runs `route_intent.py` over every sample window in a dataroot/version and writes a `{sample_id: intent_string}` JSON cache. |
| `prompts.py` | The one shared prompt builder used identically by training and inference. |
| `main.py` | Inference entrypoint: loads a trained checkpoint, runs it over nuScenes scenes, computes ADE@1s/2s/3s, writes overlay videos/plots. |
| `train_openemma_tinyvla.py` | Training loop: LoRA + diffusion-head fine-tuning, scene-level train/val split, class-balanced sampling, periodic ADE eval, NaN-loss guard, per-epoch loss logging. |
| `eval_train_vs_test.py` | Runs inference + ADE evaluation twice — on scenes the model trained on vs. genuinely unseen scenes — and plots the gap as an overfitting signal. |
| `scene_diagnostics.py` | Zero-GPU diagnostic: joins nuScenes' own scene descriptions + per-scene motion stats against an ADE results file to check whether harder (higher-curvature / higher-speed-variance) scenes are the ones with worse ADE. |
| `plot_losses.py` | Standalone re-plot of train/val loss per epoch from the JSON files `train_openemma_tinyvla.py` writes, without re-running training. |
| `_test_consistency.py` | Unit test for `route_intent_for_window` against synthetic straight/left/right/mixed scenes. |
| `resume_faangpath.tex` | *(unrelated personal document — not part of the pipeline.)* |

## Setup

```bash
pip install torch torchvision diffusers transformers peft nuscenes-devkit opencv-python matplotlib pillow
```

Also needs the vendored `llava-pythia/` package (TinyVLA's LLaVA‑Pythia implementation, with local patches for current-`transformers` compatibility) and `openemma/YOLO3D/` (only required if running with visualization overlays). Point `BASE_PRETRAINED` / `PRETRAINED` (top of `main.py` / `train_openemma_tinyvla.py`) at your local LLaVA‑Pythia checkpoint directory.

## Usage

**Generate route-intent cache** (optional — `--no-intent` skips this and drops the intent line from the prompt):

```bash
python generate_intents.py --dataroot <train_dataroot> --version v1.0-test --output intents_v1.0-test.json
python generate_intents.py --dataroot <mini_dataroot>  --version v1.0-mini --output intents_v1.0-mini.json
```

**Train**:

```bash
python train_openemma_tinyvla.py                    # LoRA + diffusion head, intent on, class-balanced sampling
python train_openemma_tinyvla.py --no-intent --no-balance-classes
```

**Run inference + ADE evaluation**:

```bash
python main.py --epoch <checkpoint_dir> --dataroot <dataroot> --version v1.0-mini
```

**Train-vs-unseen-scene comparison**:

```bash
python eval_train_vs_test.py --epoch <checkpoint_dir> \
    --train-dataroot <train_dataroot> --train-version v1.0-test \
    --test-dataroot <mini_dataroot> --test-version v1.0-mini
```

**Re-plot losses / diagnose hard scenes** (no GPU needed):

```bash
python plot_losses.py
python scene_diagnostics.py --dataroot <dataroot> --version <version> --ade-results <ade_results.jsonl>
```

## Dataset

Trained on **nuScenes v1.0-test** (150 scenes) — usable here because ground truth comes only from `ego_pose`, never from the withheld `sample_annotation` labels. Evaluated on **nuScenes-mini** (10 scenes), held out entirely. Each sample is a sliding window: 10 past + 10 future `[speed, curvature]` steps (0.5s apart) plus the front-camera image at the window boundary.

## Results

**Earlier OpenEMMA-method reproduction** (nuScenes-mini, 10 scenes, CoT + text-generation pipeline):

| Model | ADE@1s | ADE@2s | ADE@3s | avg ADE |
|---|---|---|---|---|
| LLaVA-7B | 1.90 | 1.82 | 2.50 | 2.07 |
| Qwen2.5-VL-3B | 0.87 | 1.52 | 2.19 | 1.53 |

(~1 min 30 s/frame — the main practical motivation, alongside output reliability, for moving to a diffusion head.)

**OpenEMMA‑TinyVLA (no‑CoT, diffusion head)**, nuScenes-mini, 10 scenes, 5 epochs of fine-tuning:

| | ADE@1s | ADE@2s | ADE@3s | avg ADE | failure % |
|---|---|---|---|---|---|
| **Mean over 10 scenes** | 3.809 | 6.407 | 9.054 | 6.423 | 3.33 |

Currently worse than the CoT baselines above. Suspected causes: only 5 training epochs; the diffusion head is trained entirely from scratch (no prior exposure to driving data) while LoRA only has to nudge an already-strong pretrained representation — a training-balance problem between the two; only 150 training scenes; and nuScenes provides no ground-truth navigation intent, so unlike OpenEMMA's self-generated CoT intent, this pipeline currently has no explicit representation of *where the vehicle is headed*, only how it has moved so far.

## Known limitations / next steps

- Train for more epochs and on the full nuScenes dataset rather than a 150-scene slice.
- Rebalance the LoRA-vs-diffusion-head learning-rate split (likely slower for LoRA, higher for the head, given the head trains from scratch).
- No ground-truth route/navigation intent is available from nuScenes; the geometry-only `route_intent.py` intent is a *partial* substitute, derived only from where the car already went, not from an actual planned route.

## Fixed bugs worth knowing about

- `LlavaPythiaForCausalLM` ships with no working `.generate()` — required a manual decoding loop before the diffusion head replaced text output entirely.
- `LoraConfig` needs `modules_to_save=["embed_out", "proj_to_action"]` or the diffusion head is silently dropped from saved checkpoints.
- Loading a trained checkpoint must use `PeftModel.from_pretrained(...)`, not a fresh `get_peft_model(...)` call, or inference silently runs on random untrained weights.
- **NaN loss root cause**: `from_pretrained()`'s fast-init path leaves `Conv1d`/`GroupNorm` params inside the new diffusion head as raw uninitialized memory (`GPTNeoXPreTrainedModel._init_weights()` doesn't know those layer types) — fixed by explicitly calling `reset_parameters()` on them right after load, before `get_peft_model()`.
- fp16/fp32 dtype mismatches after loading a trained LoRA/head checkpoint onto an fp16 base — fixed by upcasting `lora_`/`embed_out`/`proj_to_action` params back to fp32 post-load.
- OOM on 14–16GB GPUs — fp16 base weights, `attn_implementation="sdpa"`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, gradient checkpointing, batch size 2 with grad-accum 2, and an OOM try/except that skips a bad micro-batch rather than killing the run.
- A non-finite-loss guard in the training loop skips (no backward/step) any batch whose loss isn't finite, since one NaN gradient otherwise permanently poisons AdamW's running moment estimates.

## References

- EMMA: End-to-End Multimodal Model for Autonomous Driving — arXiv:2410.23262
- OpenEMMA — arXiv:2412.15208
- TinyVLA — arXiv:2409.12514
