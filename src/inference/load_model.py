from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config import MODEL_NAME

def load_model(model_name=MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name,device_map="auto", torch_dtype='auto')
    return tokenizer, model
def get_model_device(model):
    return model.get_input_embeddings().weight.device