# OpenEMMApro — OpenEMMA‑TinyVLA

Fork of [**OpenEMMA**](https://github.com/taco-group/OpenEMMA) (open-source reimplementation of Waymo's **EMMA** — [arXiv:2412.15208](https://arxiv.org/abs/2412.15208)) that replaces OpenEMMA's free-text chain-of-thought + regex-parsed numbers with a **diffusion policy head** bolted onto a **LLaVA‑Pythia** VLM backbone (from [TinyVLA](https://arxiv.org/abs/2409.12514)), fine-tuned with **LoRA**. The model goes straight from (image, motion history) to (future speed, curvature) numbers — no text generation, no parsing step, anywhere in the loop.

> This README describes the current, heavily modified pipeline (`main.py`, `train_openemma_tinyvla.py`, `build_model.py`, …). The project started as a direct clone of upstream OpenEMMA; essentially nothing of the original installation/usage instructions below still applies as written — see *Credit* at the bottom for the original project and citation.

## Why this exists

EMMA's own framing is that hand-designed interfaces between perception, planning, and control become a ceiling on generalization, and that splitting a driving stack into separately-trained modules means the system can never be jointly optimized toward the real objective. EMMA's fix is to put everything inside one function, `O_trajectory = G(T_intent, T_ego, V)`, with no module boundaries in between.

OpenEMMA reproduces that idea without EMMA's compute budget by doing pure prompting over an off-the-shelf MLLM — no fine-tuning — but it reintroduces exactly the kind of fragile hand-designed interface EMMA was trying to remove: the model writes a sentence containing the predicted numbers, and a regex scrapes them back out, with retry loops when the format doesn't parse. **OpenEMMA‑TinyVLA's whole point is removing that one remaining interface**: swap the output head from free-text generation to a diffusion policy head that reads the backbone's hidden states directly and emits exactly 10 × (speed, curvature) numbers every time, with one backbone forward pass instead of an autoregressive text loop.

## What changed, relative to upstream OpenEMMA

- **Removed** the YOLO3D bounding-box module from the core pipeline as an external, bolted-on dependency — a genuinely unified end-to-end model shouldn't depend on an external auxiliary detector (kept in `openemma/YOLO3D/` only for optional visualization overlays in `main.py`).
- **Replaced the backbone** with LLaVA‑Pythia (1.3B params, from TinyVLA) instead of GPT‑4o / LLaVA‑1.6‑Mistral / Llama‑3.2‑Vision / Qwen2‑VL.
- **Added LoRA** (`peft`, r=64, α=256, dropout=0.05) on both the ViT and the Pythia/GPT‑NeoX decoder's linear layers.
- **Replaced the text-generation head with a Diffusion Policy Head** (`ConditionalUnet1D`, `action_head_type="droid_diffusion"`, from TinyVLA's `policy_heads/`), trained full-parameter (not LoRA) — matching TinyVLA's own split of LoRA-tuned VLM + full-parameter diffusion head.
- **No chain-of-thought text anymore.** Since the model never generates text at all, it can't produce OpenEMMA's Scene Description / Critical Object / Intent Description sub-steps. Instead, a single shared prompt (`prompts.py`) is written to implicitly motivate that same reasoning inside the backbone's hidden states, without ever emitting it as text.
- **Route intent, re-added — but geometry-only.** An earlier VLM-generated "describe the maneuver" intent was replaced with a deterministic, map-free "turn-by-turn navigator" phrase (`route_intent.py` / `generate_intents.py`) derived purely from the ego vehicle's own recorded path curvature and speed — never from scene content, nearby agents, or critical-object reasoning. It reads like Google Maps announcing "next left turn," not a dashcam narrator describing traffic.
- **GPT‑4 API dependency removed** — no OpenAI key needed; everything runs locally against the LLaVA‑Pythia checkpoint.

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

- **Diffusion process** (`build_model.py` / `llava_pythia.py`'s `forward_diffusion_head`): `diffusers.DDIMScheduler`, 100 training timesteps, `squaredcos_cap_v2` schedule, `prediction_type="epsilon"`. Training adds noise to the GT action chunk at a random timestep and regresses the injected noise (MSE, masked by padding). Inference starts from pure Gaussian noise and denoises over 10 re-spaced DDIM steps.
- **Conditioning**: backbone hidden states are mean-pooled, layer-normed, concatenated with the 20-dim motion-history state vector, and linearly projected back to hidden size; this plus a sinusoidal timestep embedding conditions every residual block in the U-Net via FiLM.
- **Output → trajectory**: the head's (speed, curvature) outputs are integrated via the same bicycle-model math as upstream OpenEMMA (heading from ∫curvature·speed, then velocity components, then position) — never regressed directly as (x, y).
- **Two learning-rate groups**: LoRA params at `2e-4`, diffusion-head params at `2e-5` — see *Known limitations* for why this split is itself under suspicion as a cause of the current result gap.

## Repository layout

| File | Role |
|---|---|
| `build_model.py` | Constructs the LLaVA‑Pythia + LoRA + diffusion-head model; two paths — fresh training init vs. loading a trained checkpoint (`PeftModel.from_pretrained`). |
| `openemma_dataset.py` | nuScenes → PyTorch `Dataset`. Builds `(image_path, prompt, history[10,2], future[10,2])` samples; scene-level train/val split; optional precomputed-intent lookup; `iter_scene_motion()` is the single shared scene-walk + curvature/speed computation reused by every script below. |
| `route_intent.py` | Deterministic, map-free "turn-by-turn navigator" intent classifier — pure curvature/speed geometry, no image, no VLM, no GPU. |
| `generate_intents.py` | Offline pass: runs `route_intent.py` over every sample window in a dataroot/version and writes a `{sample_id: intent_string}` JSON cache. |
| `prompts.py` | The one shared prompt builder used identically by training and inference. |
| `utils.py` | Curvature/trajectory math (bicycle-model integration) and plotting/overlay helpers. |
| `main.py` | Inference entrypoint: loads a trained checkpoint, runs it over nuScenes scenes, computes ADE@1s/2s/3s, writes overlay videos/plots. |
| `train_openemma_tinyvla.py` | Training loop: LoRA + diffusion-head fine-tuning, scene-level train/val split, class-balanced sampling, periodic ADE eval, NaN-loss guard, per-epoch loss logging. |
| `eval_train_vs_test.py` | Runs inference + ADE evaluation twice — on scenes the model trained on vs. genuinely unseen scenes — and plots the gap as an overfitting signal. |
| `scene_diagnostics.py` | Zero-GPU diagnostic: joins nuScenes' own scene descriptions + per-scene motion stats against an ADE results file to check whether harder (higher-curvature / higher-speed-variance) scenes are the ones with worse ADE. |
| `plot_losses.py` | Standalone re-plot of train/val loss per epoch from the JSON files `train_openemma_tinyvla.py` writes, without re-running training. |
| `extract_nuscenes.py` | (upstream) nuScenes archive extraction helper. |
| `llava-pythia/`, `policy_heads/` | Vendored TinyVLA packages (LLaVA‑Pythia model code, diffusion policy head implementation) — locally patched for current-`transformers` compatibility. |
| `openemma/YOLO3D/` | Vendored YOLO-based 3D detector — optional, visualization-only in this pipeline. |
| `assests/` | Reference papers (EMMA, OpenEMMA, TinyVLA, VLA survey), design-rationale guides (`openemma_guide.md`, `openemma_tinyvla_guide.md`), Kaggle notebooks, and saved checkpoints/results. |
| `Claude outputs/` | Compiled IEEE-format project report (`openemma_tinyvla_report.pdf` / `.tex`). |

## Setup

`requirements.txt` reflects the **original** OpenEMMA baseline (Qwen/LLaVA/YOLO3D path) and is missing packages the TinyVLA path needs. Install both:

```bash
pip install -r requirements.txt
pip install peft diffusers nuscenes-devkit
```

Also needs the vendored `llava-pythia/` and `policy_heads/` packages already in this repo (with local patches for current-`transformers` compatibility). Point `BASE_PRETRAINED` / `PRETRAINED` (top of `main.py` / `train_openemma_tinyvla.py`) at your local LLaVA‑Pythia checkpoint directory — no OpenAI API key is needed for this path.

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

Currently worse than the CoT baselines above. The full per-scene table, methodology math, and discussion are in `Claude outputs/openemma_tinyvla_report.pdf`.

## Known limitations / next steps

- Train for more epochs and on the full nuScenes dataset rather than a 150-scene slice.
- Rebalance the LoRA-vs-diffusion-head learning-rate split (likely slower for LoRA, higher for the head, given the head trains from scratch while LoRA only has to nudge an already-strong pretrained representation).
- No ground-truth route/navigation intent is available from nuScenes; the geometry-only `route_intent.py` intent is a *partial* substitute, derived only from where the car already went, not from an actual planned route.

## Fixed bugs worth knowing about

- `LlavaPythiaForCausalLM` ships with no working `.generate()` — required a manual decoding loop before the diffusion head replaced text output entirely.
- `LoraConfig` needs `modules_to_save=["embed_out", "proj_to_action"]` or the diffusion head is silently dropped from saved checkpoints.
- Loading a trained checkpoint must use `PeftModel.from_pretrained(...)`, not a fresh `get_peft_model(...)` call, or inference silently runs on random untrained weights.
- **NaN loss root cause**: `from_pretrained()`'s fast-init path leaves `Conv1d`/`GroupNorm` params inside the new diffusion head as raw uninitialized memory (`GPTNeoXPreTrainedModel._init_weights()` doesn't know those layer types) — fixed by explicitly calling `reset_parameters()` on them right after load, before `get_peft_model()`.
- fp16/fp32 dtype mismatches after loading a trained LoRA/head checkpoint onto an fp16 base — fixed by upcasting `lora_`/`embed_out`/`proj_to_action` params back to fp32 post-load.
- OOM on 14–16GB GPUs (Kaggle) — fp16 base weights, `attn_implementation="sdpa"`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, gradient checkpointing, batch size 2 with grad-accum 2, and an OOM try/except that skips a bad micro-batch rather than killing the run.
- A non-finite-loss guard in the training loop skips (no backward/step) any batch whose loss isn't finite, since one NaN gradient otherwise permanently poisons AdamW's running moment estimates.

## Credit

This repository began as a clone of [**OpenEMMA**](https://github.com/taco-group/OpenEMMA) (Xing, Qian, Wang, Hua, Tian, Zhou, Tu — arXiv:2412.15208) and is built on ideas from Waymo's **EMMA** (arXiv:2410.23262) and **TinyVLA** (arXiv:2409.12514). If this work is useful to you, please cite OpenEMMA:

```bibtex
@article{openemma,
	author = {Xing, Shuo and Qian, Chengyuan and Wang, Yuping and Hua, Hongyuan and Tian, Kexin and Zhou, Yang and Tu, Zhengzhong},
	title = {OpenEMMA: Open-Source Multimodal Model for End-to-End Autonomous Driving},
	journal = {arXiv},
	year = {2024},
	month = dec,
	eprint = {2412.15208},
	doi = {10.48550/arXiv.2412.15208}
}
```

Released under the Apache 2.0 license (see `LICENSE`), inherited from upstream OpenEMMA.
