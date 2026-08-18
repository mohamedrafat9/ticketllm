from src.inference.load_model import load_model
from src.inference.generate import generate_text
user_message = "Explain software engineering in simple terms."
prompt = f"""
You are a helpful assistant.
Please provide a detailed explanation of the concept of software engineering,
User message: {user_message}
"""
if __name__ == "__main__": 
    tokenizer, model = load_model() 
    generated_text = generate_text(model,
                                tokenizer, 
                                prompt, 
                                max_length=50,
                                temperature=1.0, 
                                top_k=50) 
    print(generated_text)