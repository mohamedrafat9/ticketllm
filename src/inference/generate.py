from src.inference.decoding import select_next_token
from src.inference.decoding import apply_temperature, apply_top_k
from src.inference.load_model import get_model_device
from src.inference.tokenizer import tokenize
import torch


def generate_text(model, tokenizer, prompt, max_length=50, temperature=1.0,top_p=None, top_k=None,max_new_tokens=50):
    device = get_model_device(model)
    inputs = tokenize(prompt, tokenizer, device)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    for _ in range(max_new_tokens):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_logits = apply_temperature(next_token_logits, temperature)
        next_token_logits = apply_top_k(next_token_logits, top_k) if top_k is not None else next_token_logits
        # next_token_logits = apply_top_p(next_token_logits, top_p) if top_p
        next_token = select_next_token(next_token_logits, temperature)
        if(tokenizer.eos_token_id is not None
            and next_token.item() == tokenizer.eos_token_id):
            break
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0],
                                                                1), device=device)], dim=-1)

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)