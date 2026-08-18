import torch

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