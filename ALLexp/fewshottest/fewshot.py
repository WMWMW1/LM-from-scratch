import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from transformers import Trainer, TrainingArguments

from transformers import GPT2Tokenizer

# 初始化GPT-2分词器
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# # 验证分词器是否正常工作
# test_text = "AND"
# encoded_input = tokenizer.encode(test_text, return_tensors="pt")
# decoded_output = tokenizer.decode(encoded_input[0])

# print(encoded_input, decoded_output)
class model(nn.Module):
    def __init__(self, num_tokens=4, d_model=64, nhead=2, dim_feedforward=256, num_layers=1):
        super(model, self).__init__()
        self.token_emb = nn.Embedding(num_tokens, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_tokens)

    def forward(self, src):
        src = self.token_emb(src)  # (Batch, Seq, Embed)
        src = src.permute(1, 0, 2)  # (Seq, Batch, Embed) for PyTorch Transformer
        output = self.transformer_encoder(src)
        output = output.permute(1, 0, 2)  # Back to (Batch, Seq, Embed)
        output = self.fc(output)
        return output[:, -1, :]  # Only return the last output
