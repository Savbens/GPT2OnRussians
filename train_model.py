import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json
from tqdm import tqdm

class TextGenerationDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=256):
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                self.data.append(obj['original'])
                self.data.append(obj['translation'])
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': encodings['input_ids'].squeeze()
        }

def prepare_dataset(jsonl_path, tokenizer, max_length=256, batch_size=32):
    print("Loading dataset...")
    dataset = TextGenerationDataset(jsonl_path, tokenizer, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def main():
    model_name = "gpt2"
    save_dir = "./gpt2-text-generation"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(save_dir, exist_ok=True)
    print("Loading tokenizer and model...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.to(device)
    dataloader = prepare_dataset(
        jsonl_path="translated_tiny_stories.jsonl",
        tokenizer=tokenizer,
        max_length=256,
        batch_size=8
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    print("Starting training...")
    model.train()
    num_epochs = 20
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix({'loss': loss.item()})
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        model.save_pretrained(os.path.join(save_dir, f"checkpoint-{epoch + 1}"))
        tokenizer.save_pretrained(os.path.join(save_dir, f"checkpoint-{epoch + 1}"))
    model.save_pretrained(os.path.join(save_dir, "final"))
    tokenizer.save_pretrained(os.path.join(save_dir, "final"))
    print("Training completed!")

if __name__ == "__main__":
    main()

