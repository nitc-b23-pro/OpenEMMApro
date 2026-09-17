"""
llava_pythia_inference.py
A minimal, manual text-generation loop for LlavaPythiaForCausalLM, replacing
Qwen's model.generate() call.

FIX 1 (chat template): the previous version fed the model a raw
"<image>\n{prompt}" string with no conversation formatting. LlavaPythiaForCausalLM
was instruction-tuned by TinyVLA using LLaVA's standard "USER: ... ASSISTANT:"
chat template (llava_pythia/conversation.py -> conv_pythia). Without the
trailing "ASSISTANT:" cue the model has no learned signal to switch from
"reading" to "answering".

FIX 2 (this version, speed): the loop used to call
    model(input_ids=generated_ids, images=image_tensor)
passing the FULL growing sequence every step, with no `past_key_values` and no
`use_cache`. Look at prepare_inputs_labels_for_multimodal() in llava_arch.py --
it only skips re-encoding the image when `input_ids.shape[1] == 1` (i.e. you're
feeding just the newest token and relying on a KV cache for everything before
it). Because the old loop never did that, EVERY one of up to max_new_tokens
steps was: re-running the full CLIP/SigLIP vision tower on the image from
scratch, AND re-running full (uncached) self-attention over the entire
sequence generated so far. That is the actual reason one frame was taking so
long -- it is not "no KV-cache" in the abstract, it's "re-encoding the image
end-to-end at every single decoding step", which for a real ViT is far more
expensive than the GPT-NeoX text steps themselves. GPTNeoXModel (imported
straight from `transformers`) already understands `use_cache`/`past_key_values`
-- prepare_inputs_labels_for_multimodal() was already written to short-circuit
correctly for cached decoding, it just wasn't being used.

This version does real incremental decoding: prefill once (image + full
prompt), then feed only the newest token on every later step. If your
`transformers` version on Kaggle returns a `Cache` object instead of the
legacy tuple-of-tuples `past_key_values` (a real risk -- see openemma_guide.md's
own warning about llava_pythia being written against transformers==4.37.1),
the cached path will raise, and this falls back to the old, slow, always-
correct-but-much-slower recompute-everything loop rather than silently
producing wrong output.
"""
import torch
from PIL import Image
from llava_pythia.mm_utils import tokenizer_image_token
from llava_pythia.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava_pythia.conversation import conv_templates


def build_prompt(prompt_text, conv_mode="pythia"):
    """Wrap raw task text in the USER/ASSISTANT template the checkpoint was
    actually trained on, ending right at 'ASSISTANT:' so generation continues
    from there (do NOT append <|endoftext|> here -- that only belongs in
    TinyVLA's action-head path, where no text is ever generated afterwards)."""
    conv = conv_templates[conv_mode].copy()
    inp = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def preprocess_image(image_path, image_processor, model):
    """Public helper: preprocess a frame's image ONCE and reuse the returned
    tensor across all of that frame's CoT sub-prompts (Scene/Object/Intent/
    Motion, plus GenerateMotion's up-to-3 retries) instead of re-opening and
    re-running the image processor on the same JPEG for every single call."""
    return _prepare_image_tensor(image_path, image_processor, model)


def _prepare_image_tensor(image_path_or_tensor, image_processor, model):
    """Accepts either a path (backward compatible) or an already-preprocessed
    tensor (new -- lets callers encode an image ONCE per frame and reuse it
    across the 4 CoT sub-prompts instead of re-opening + re-preprocessing the
    same JPEG on every one of Scene/Object/Intent/Motion + retries)."""
    if isinstance(image_path_or_tensor, torch.Tensor):
        return image_path_or_tensor.to(device="cuda", dtype=model.dtype)
    img = Image.open(image_path_or_tensor).convert("RGB")
    image_tensor = image_processor.preprocess(img, return_tensors="pt")["pixel_values"]
    return image_tensor.to(device="cuda", dtype=model.dtype)


def _generate_with_cache(input_ids, image_tensor, model, eos_id, max_new_tokens):
    """Real incremental decoding: prefill once, then feed one new token per step."""
    with torch.inference_mode():
        # Prefill: full prompt (with the image token) + image, once.
        out = model(input_ids=input_ids, images=image_tensor, use_cache=True)
        past_key_values = out.past_key_values
        next_token_logits = out.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        generated = [next_token]
        dummy_attn = torch.ones((input_ids.shape[0], 1), dtype=input_ids.dtype, device=input_ids.device)

        for _ in range(max_new_tokens - 1):
            if eos_id is not None and next_token.item() == eos_id:
                break
            # input_ids.shape[1] == 1 here -> prepare_inputs_labels_for_multimodal
            # short-circuits and does NOT re-touch the vision tower or re-embed
            # anything already in past_key_values.
            out = model(input_ids=next_token, images=image_tensor,
                        past_key_values=past_key_values, attention_mask=dummy_attn,
                        use_cache=True)
            past_key_values = out.past_key_values
            next_token_logits = out.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated.append(next_token)

    return torch.cat(generated, dim=1)


def _generate_slow_no_cache(input_ids, image_tensor, model, eos_id, max_new_tokens):
    """Original fallback: recomputes the full forward pass every step. Only used
    if the cached path above fails (e.g. a transformers version whose GPTNeoX
    attention returns a Cache object incompatible with this repo's tuple-based
    past_key_values indexing in prepare_inputs_labels_for_multimodal)."""
    generated_ids = input_ids
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=generated_ids, images=image_tensor)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            if eos_id is not None and next_token.item() == eos_id:
                break
    return generated_ids[:, input_ids.shape[1]:]


def llava_pythia_generate(prompt_text, image_path, model, tokenizer, image_processor,
                           max_new_tokens=128, conv_mode="pythia", _warned=[]):
    prompt = build_prompt(prompt_text, conv_mode=conv_mode)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                       return_tensors="pt").unsqueeze(0).cuda()

    image_tensor = _prepare_image_tensor(image_path, image_processor, model)

    # The tokenizer's eos_token_id should already correspond to "<|endoftext|>"
    # (GPT-NeoX/Pythia's native EOS), which is also conv_pythia's sep2 -- i.e.
    # the token the model was trained to emit at the end of its ASSISTANT turn.
    eos_id = tokenizer.eos_token_id

    try:
        new_tokens = _generate_with_cache(input_ids, image_tensor, model, eos_id, max_new_tokens)
    except Exception as e:
        if not _warned:
            print(f"[llava_pythia_generate] Cached decoding failed ({e!r}); "
                  f"falling back to the slow no-cache loop for the rest of this run. "
                  f"This usually means your transformers version's GPTNeoX returns a "
                  f"Cache object instead of a legacy tuple for past_key_values.")
            _warned.append(True)
        new_tokens = _generate_slow_no_cache(input_ids, image_tensor, model, eos_id, max_new_tokens)

    return tokenizer.decode(new_tokens[0], skip_special_tokens=True)
