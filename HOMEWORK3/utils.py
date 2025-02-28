# utils.py
import torch
import numpy as np
from sklearn.model_selection import train_test_split

def load_text(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def build_vocab(text):
    chars = sorted(list(set(text)))
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}
    return char_to_ix, ix_to_char

class CharDataset(torch.utils.data.Dataset):
    def __init__(self, text, seq_length):
        self.seq_length = seq_length
        self.text = text
        self.char_to_ix, self.ix_to_char = build_vocab(text)
        self.vocab_size = len(self.char_to_ix)
        self.data = []
        self.labels = []
        # Create sequences: each sample is a sequence of length seq_length with a label (next char)
        for i in range(len(text) - seq_length):
            seq = text[i:i + seq_length]
            label = text[i + seq_length]
            self.data.append([self.char_to_ix[ch] for ch in seq])
            self.labels.append(self.char_to_ix[label])
        self.data = torch.tensor(self.data, dtype=torch.long)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def get_dataloaders(text, seq_length, batch_size, test_size=0.2):
    dataset = CharDataset(text, seq_length)
    # Split indices for train and validation
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_size * dataset_size))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, dataset.char_to_ix, dataset.ix_to_char, dataset.vocab_size

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
