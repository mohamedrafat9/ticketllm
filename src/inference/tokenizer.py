def tokenize(text,tokenizer,device):
    ids = tokenizer(text, return_tensors="pt").to(device)
    return ids

# prompt = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True
# )

# print(prompt)