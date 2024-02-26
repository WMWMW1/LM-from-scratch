import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import TransformerModel
from dataset import TextDataset
from tokenizer import build_vocab, encode_text, decode_tokens
import json

# Function to extract text from a JSON file
def extract_text_from_json(json_file_path):
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)
            text_array = [item['text'] for item in data]
            return text_array
    except FileNotFoundError:
        print("File not found.")
        return []

# Load text data from JSON file
json_file_path = "training_data.json"  # Replace with the path to your JSON file
# json_file_path = "overfit_test.json"  # Replace with the path to your JSON file

texts = extract_text_from_json(json_file_path)
print(texts)

# Build vocabulary
vocab = build_vocab(texts)
reverse_vocab = {index: token for token, index in vocab.items()}

# Parameters
VOCAB_SIZE = len(vocab)
EMBED_SIZE = 256
NUM_HEADS = 8
HIDDEN_DIM = 200
NUM_LAYERS =4
BATCH_SIZE = 8
NUM_EPOCHS = 1

# Data preparation
train_texts, dev_texts = train_test_split(texts, test_size=0.2)  # Split data into 80% train and 20% dev
train_dataset = TextDataset(train_texts, vocab)
dev_dataset = TextDataset(dev_texts, vocab)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize model
model = TransformerModel(VOCAB_SIZE, EMBED_SIZE, NUM_HEADS, HIDDEN_DIM, NUM_LAYERS)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)  # Move model to the device

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss(reduction='none')
optimizer = optim.Adam(model.parameters())

# Training loop
for epoch in range(NUM_EPOCHS):
    # Train
    model.train()
    for batch in train_dataloader:
        optimizer.zero_grad()
        model_input, mask = [item.to(device) for item in batch]
        target = model_input[:, 1:]  # Assuming target is the next token in sequence
        mask = mask[:, 1:]  # Adjust mask accordingly
        model_output = model(model_input[:, :-1]).transpose(1, 2)
        loss = loss_fn(model_output, target)
        loss = (loss * mask).sum() / mask.sum()
        loss.backward()
        optimizer.step()
    print(f'Training    - Epoch {epoch+1}, Loss: {loss.item()}')

    # Evaluate on development set
    model.eval()
    total_dev_loss = 0.0
    with torch.no_grad():
        for batch in dev_dataloader:
            model_input, mask = [item.to(device) for item in batch]
            target = model_input[:, 1:]
            mask = mask[:, 1:]
            model_output = model(model_input[:, :-1]).transpose(1, 2)
            loss = loss_fn(model_output, target)
            loss = (loss * mask).sum() / mask.sum()
            total_dev_loss += loss.item()
    avg_dev_loss = total_dev_loss / len(dev_dataloader)
    print(f'Development - Epoch {epoch+1}, Loss: {avg_dev_loss}')

# Function to generate text
def generate_text(model, start_text, vocab, reverse_vocab, stop_words, max_length=50, temperature=1.0, top_k=50):
    model.eval()
    encoded_input = encode_text(start_text, vocab)
    input_tensor = torch.tensor([encoded_input], dtype=torch.long).to(device)
    generated_tokens = []
    with torch.no_grad():
        for _ in range(max_length):
            output = model(input_tensor)
            logits = output[0, -1, :] / temperature
            probabilities = F.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probabilities, top_k)
            top_probs = top_probs / top_probs.sum()
            sampled_token_id = top_indices[torch.multinomial(top_probs, 1)]
            # print(sampled_token_id[0])
            # print(reverse_vocab[sampled_token_id.item()])

            if reverse_vocab[sampled_token_id.item()] in stop_words:
                break
            generated_tokens.append(sampled_token_id.item())
            input_tensor = torch.cat((input_tensor, sampled_token_id.unsqueeze(0)), dim=1)
    generated_text = decode_tokens(generated_tokens, reverse_vocab)
    return generated_text

# Stop words set
stop_words = {"<EOS>"}

# Inference loop
temperature = 1
top_k = 3
while True:
    start_text = input("Please enter something: ")
    if start_text == "exit()":
        break
    generated_text = generate_text(model, start_text, vocab, reverse_vocab, stop_words, max_length=50, temperature=temperature, top_k=top_k)
    print("Generated Text:", generated_text)
