import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "gpt2"  
def load_model(model_name=MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name,device_map="auto", torch_dtype='auto')
    return tokenizer, model
def get_model_device(model):
    return model.get_input_embeddings().weight.device

def tokenize(text,tokenizer,device):
    ids = tokenizer(text, return_tensors="pt").to(device)
    return ids

def get_logits(id, model):
    logits=model(id)
    return logits

def apply_temperature(logits, temperature):
    if temperature <= 0:
        return logits
    else:
        return logits / temperature

def apply_top_k(logits, k):
    if k <= 0:
        return logits
    else:
        k=min(k, logits.size(-1))  # Ensure k does not exceed the vocabulary size
        top_k_values, _ = torch.topk(logits, k)
        treshold=top_k_values[:, -1].unsqueeze(-1)  # Get the k-th largest value
        logits=logits.masked_fill(
            logits < treshold, float('-inf')
        )
        return logits
# def apply_top_p(logits, p):

def select_next_token(logits, temperature=1.0):
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)
    else:
        probabilities =torch.softmax(logits,dim=-1)
        next_token =torch.multinomial(probabilities,num_samples=1)
        return next_token


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

if __name__ == "__main__":
    tokenizer, model = load_model()
    prompt = "we will live today"
    generated_text = generate_text(model, tokenizer, prompt, max_length=50, temperature=1.0, top_k=50)
    print(generated_text)
        


           