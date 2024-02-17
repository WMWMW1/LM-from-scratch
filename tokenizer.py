import re
from collections import Counter

def basic_tokenizer(text):
    """基本的文本分词逻辑，根据空格和标点符号分割文本"""
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return tokens
def build_vocab(texts, max_vocab_size=10000, min_freq=1):
    """构建词汇表"""
    token_freqs = Counter()
    for text in texts:
        tokens = basic_tokenizer(text)
        token_freqs.update(tokens)
    
    # Add <EOS> token to the vocabulary
    vocab = {"<PAD>": 0, "<UNK>": 1, "<EOS>": 2}
    for token, freq in token_freqs.most_common(max_vocab_size - len(vocab)):
        if freq >= min_freq:
            vocab[token] = len(vocab)
    
    return vocab


def encode_text(text, vocab):
    """将文本编码为词汇表索引的列表，包括句子结束标记"""
    tokens = basic_tokenizer(text)
    encoded = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    # Append <EOS> token at the end of the encoded sequence
    encoded.append(vocab["<EOS>"])
    return encoded


def decode_tokens(tokens, reverse_vocab):
    """将词汇表索引的列表解码为文本"""
    return " ".join([reverse_vocab[token] for token in tokens])