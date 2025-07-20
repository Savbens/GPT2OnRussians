import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_continuation(model, tokenizer, input_text, max_length=256, temperature=0.7):
    model.eval()
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            no_repeat_ngram_size=2
        )
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    continuation = generated_text[len(input_text):].strip()
    return continuation

def main():
    model_path = "./gpt2-text-generation/final"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loading model and tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(device)
    test_inputs = [
        "Однажды в далеком лесу",
        "Маша пошла в школу и",
        "Кот сидел на подоконнике и",
        "Весна пришла, и",
        "Старик сидел у окна и",
        "Once upon a time in a far forest",
        "Mary went to school and",
        "The cat was sitting on the windowsill and",
        "Spring came, and",
        "The old man sat by the window and"
    ]
    print("\nTesting model with example inputs...")
    for i, text in enumerate(test_inputs, 1):
        print(f"\nExample {i}:")
        print(f"Input: {text}")
        continuation = generate_continuation(model, tokenizer, text)
        print(f"Continuation: {continuation}")
        print(f"Full text: {text} {continuation}")

if __name__ == "__main__":
    main() 