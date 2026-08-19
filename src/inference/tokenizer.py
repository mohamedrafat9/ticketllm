# def tokenize(text,tokenizer,device):
#     ids = tokenizer(text, return_tensors="pt").to(device)
#     return ids

# # prompt = tokenizer.apply_chat_template(
# #     messages,
# #     tokenize=False,
# #     add_generation_prompt=True
# # )

# # print(prompt)
def tokenize(messages, tokenizer, device):

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(device)

    return inputs