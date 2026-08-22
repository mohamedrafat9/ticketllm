from src.inference.decoding import select_next_token
from src.inference.decoding import apply_temperature, apply_top_k
from src.inference.load_model import get_model_device
from src.inference.tokenizer import tokenize
import torch


def generate_text(
    model,
    tokenizer,
    prompt,
    max_length=50,
    temperature=1.0,
    top_p=None,
    top_k=None,
    max_new_tokens=50
):
    device = get_model_device(model)

    inputs = tokenize(prompt, tokenizer, device)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    generated_tokens = []

    with torch.inference_mode():

        # First forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True
        )

        past_key_values = outputs.past_key_values

        next_token_logits = outputs.logits[:, -1, :]

        # Temperature
        next_token_logits = apply_temperature(
            next_token_logits,
            temperature
        )

        # Top-k
        if top_k is not None:
            next_token_logits = apply_top_k(
                next_token_logits,
                top_k
            )

        # Select token
        next_token = select_next_token(
            next_token_logits,
            1.0
        )

        generated_tokens.append(next_token)

        # Stop if EOS
        if (
            tokenizer.eos_token_id is not None
            and next_token.item() == tokenizer.eos_token_id
        ):
            return tokenizer.decode(
                next_token[0],
                skip_special_tokens=True
            )

        # Generate remaining tokens
        for _ in range(max_new_tokens - 1):

            # Only pass the newly generated token
            input_ids = next_token

            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        device=device,
                        dtype=attention_mask.dtype
                    )
                ],
                dim=-1
            )

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True
            )

            past_key_values = outputs.past_key_values

            next_token_logits = outputs.logits[:, -1, :]

            next_token_logits = apply_temperature(
                next_token_logits,
                temperature
            )

            if top_k is not None:
                next_token_logits = apply_top_k(
                    next_token_logits,
                    top_k
                )

            next_token = select_next_token(
                next_token_logits,
                1.0
            )

            generated_tokens.append(next_token)

            if (
                tokenizer.eos_token_id is not None
                and next_token.item() == tokenizer.eos_token_id
            ):
                break

    # Decode ONLY generated tokens
    generated_ids = torch.cat(
        generated_tokens,
        dim=-1
    )

    return tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True
    )