import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.infrence.model import load_model
from structured.generator import StructuredGeneration
from structured.schemas   import TicketOutput
from prompt.classifer import build_ticket_classifier



def main() : 
    tokenizer, model = load_model()

    generator = StructuredGeneration(model, tokenizer)

    user_query = """Your support team was very helpful, thank you!"""

    prompt = build_ticket_classifier(user_query)
    
    result = generator.generate(prompt, TicketOutput)

    print(result)


if __name__ == '__main__' :
    main()