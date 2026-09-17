"""
llava_pythia_inference.py
A minimal, manual text-generation loop for LlavaPythiaForCausalLM, replacing
Qwen's model.generate() call. Recomputes the full forward pass every step
(no KV-cache) -- slower than a real .generate(), but simple and correct.

FIX (2026-09-17): the previous version fed the model a raw
"<image>\n{prompt}" string with no conversation formatting. LlavaPythiaForCausalLM
was instruction-tuned by TinyVLA using LLaVA's standard "USER: ... ASSISTANT:"
chat template (see llava_pythia/conversation.py -> conv_pythia, and TinyVLA's
own eval_real_franka.py, which always builds the prompt through
conv_templates[...].get_prompt()). Without that template -- specifically,
without the trailing "ASSISTANT:" cue -- the model has no learned signal that
it should now produce an answer instead of continuing the user's text, which
is exactly why it was echoing the prompt's own bracket-format placeholder or
copying the historical numbers back out: those are the most probable
"continuations" of an unfinished-looking instruction, not a bug in the model
weights themselves.
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


def llava_pythia_generate(prompt_text, image_path, model, tokenizer, image_processor,
                           max_new_tokens=128, conv_mode="pythia"):
    prompt = build_prompt(prompt_text, conv_mode=conv_mode)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                       return_tensors="pt").unsqueeze(0).cuda()

    img = Image.open(image_path).convert("RGB")
    image_tensor = image_processor.preprocess(img, return_tensors="pt")["pixel_values"].cuda()

    # The tokenizer's eos_token_id should already correspond to "<|endoftext|>"
    # (GPT-NeoX/Pythia's native EOS), which is also conv_pythia's sep2 -- i.e.
    # the token the model was trained to emit at the end of its ASSISTANT turn.
    eos_id = tokenizer.eos_token_id

    generated_ids = input_ids
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=generated_ids, images=image_tensor)
            next_token_logits = outputs.logits[:, -1, :]          # scores for the next word
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # pick the best one
            generated_ids = torch.cat([generated_ids, next_token], dim=1)       # add it to the sentence
            if eos_id is not None and next_token.item() == eos_id:
                break

    new_tokens = generated_ids[0, input_ids.shape[1]:]   # strip off the prompt, keep only the reply
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
