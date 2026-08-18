def tokenize(text,tokenizer,device):
    ids = tokenizer(text, return_tensors="pt").to(device)
    return ids