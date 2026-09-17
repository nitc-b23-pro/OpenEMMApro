"""
llava_pythia_inference.py
A minimal, manual text-generation loop for LlavaPythiaForCausalLM, replacing
Qwen's model.generate() call. Recomputes the full forward pass every step
(no KV-cache) -- slower than a real .generate(), but simple and correct.
"""
import torch
from PIL import Image
from llava_pythia.mm_utils import tokenizer_image_token
from llava_pythia.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX

def llava_pythia_generate(prompt_text, image_path, model, tokenizer, image_processor,
                           max_new_tokens=128):
    prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                       return_tensors="pt").unsqueeze(0).cuda()

    img = Image.open(image_path).convert("RGB")
    image_tensor = image_processor.preprocess(img, return_tensors="pt")["pixel_values"].cuda()

    generated_ids = input_ids
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=generated_ids, images=image_tensor)
            next_token_logits = outputs.logits[:, -1, :]          # scores for the next word
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # pick the best one
            generated_ids = torch.cat([generated_ids, next_token], dim=1)       # add it to the sentence
            if next_token.item() == tokenizer.eos_token_id:
                break

    new_tokens = generated_ids[0, input_ids.shape[1]:]   # strip off the prompt, keep only the reply
    return tokenizer.decode(new_tokens, skip_special_tokens=True)