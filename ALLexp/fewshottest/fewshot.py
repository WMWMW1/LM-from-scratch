import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

class Tokenizer:
    def __init__(self, token_to_idx, pad_token='<pad>', unk_token='<unk>'):
        self.token_to_idx = token_to_idx
        self.idx_to_token = {idx: token for token, idx in token_to_idx.items()}
        self.pad = pad_token
        self.unk = unk_token
        # 确保<pad>和<unk>标记被添加到字典中
        if self.pad not in self.token_to_idx:
            self.token_to_idx[self.pad] = len(self.token_to_idx)
        if self.unk not in self.token_to_idx:
            self.token_to_idx[self.unk] = len(self.token_to_idx)
        
    def encode(self, sequence):
        # 使用.get方法返回未知字符的索引，如果字符不在字典中
        return [self.token_to_idx.get(token, self.token_to_idx[self.unk]) for token in sequence]

    def decode(self, indices):
        # 解码时，未知索引将被忽略或替换为<unk>
        return [self.idx_to_token.get(idx, self.unk) for idx in indices]

    
class ExternalSequenceDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer):
        self.sequences = [tokenizer.encode(sequence) for sequence in sequences]
        self.labels = [tokenizer.encode(label)[0] for label in labels]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

# 模型定义
class SimpleTransformerModel(nn.Module):
    def __init__(self, num_tokens=4, d_model=64, nhead=2, dim_feedforward=256, num_layers=1):
        super(SimpleTransformerModel, self).__init__()
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

# 训练函数
def train(model, data_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for src, tgt in data_loader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src)
        loss = loss_fn(output, tgt)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def load_data(tokenizer):
    # 假设这些是我们的原始序列
    raw_sequences = ['ABC', 'BCD', 'CDA', 'DAB']
    
    # 将序列和标签分别处理
    sequences = []
    labels = []
    
    # 对每个序列生成输入和标签
    for seq in raw_sequences:
        for i in range(1, len(seq)):
            input_seq = seq[:i] + tokenizer.pad * (len(seq) - i)  # 填充到相同长度
            target_label = seq[i]
            sequences.append(input_seq)
            labels.append(target_label)
            
    return sequences, labels


def human_test_loop(model, tokenizer, device):
    print("Enter a sequence of 'A', 'B', 'C', 'D' or type 'exit' to quit:")
    while True:
        user_input = input("Input: ")
        if user_input.lower() == 'exit':
            break

        input_tensor = torch.tensor([tokenizer.encode(user_input)], dtype=torch.long).to(device)
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            predicted_index = output.argmax(dim=1).item()
            predicted_token = tokenizer.decode([predicted_index])
            print("Predicted next token:", ''.join(predicted_token))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义tokenizer
    token_to_idx = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3,
    'AND': 4, 'OR': 5, 'XOR': 6, 'NOT': 7, 'XNOR': 8, '(': 9, ')': 10
}

    tokenizer = Tokenizer(token_to_idx)
    
    sequences, labels = load_data(tokenizer)
    dataset = ExternalSequenceDataset(sequences, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    model = SimpleTransformerModel().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    



    
    epochs = 1
    for epoch in range(epochs):
        loss = train(model, dataloader, loss_fn, optimizer, device)
        print(f'Epoch {epoch+1}, Loss: {loss}')
        print(epoch)

        
    human_test_loop(model, tokenizer, device)


if __name__ == '__main__':
    main()
