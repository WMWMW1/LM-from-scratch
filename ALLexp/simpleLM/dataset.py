import torch
from torch.utils.data import Dataset
from tokenizer import encode_text

class TextDataset(Dataset):
    def __init__(self, texts, vocab, max_length=None):
        self.vocab = vocab
        # Ensure <EOS> token is in the vocab
        if "<EOS>" not in vocab:
            raise ValueError("Vocabulary must include an '<EOS>' token")
        self.encoded_texts = [encode_text(text, vocab) + [self.vocab["<EOS>"]] for text in texts]
        self.max_length = max_length if max_length is not None else max(len(t) for t in self.encoded_texts)

    def __len__(self):
        return len(self.encoded_texts)

    def __getitem__(self, idx):
        encoded_text = self.encoded_texts[idx]
        padded_text, mask = self.pad_sequence(encoded_text)
        return torch.tensor(padded_text, dtype=torch.long), torch.tensor(mask, dtype=torch.long)
    
    def pad_sequence(self, encoded_text):
        padded_length = self.max_length
        if len(encoded_text) > padded_length:
            # If the text with <EOS> is longer than max_length, truncate it before adding <EOS>
            encoded_text = encoded_text[:padded_length-1] + [self.vocab["<EOS>"]]
        padding_length = padded_length - len(encoded_text)
        padded_text = encoded_text + [self.vocab["<PAD>"]] * padding_length
        mask = [1] * len(encoded_text) + [0] * padding_length  # 1s for actual data and <EOS>, 0s for padding
        return padded_text, mask
