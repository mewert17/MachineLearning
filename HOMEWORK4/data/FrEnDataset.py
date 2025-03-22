import torch
from torch.utils.data import Dataset
from data.dataset import Vocabulary, tokenize_and_pad  # reuse your normalized vocab/tokenizer

class FrEnDataset(Dataset):
    def __init__(self, pairs):
        self.fr_vocab = Vocabulary()
        self.en_vocab = Vocabulary()
        self.pairs = []

        for fr, en in pairs:
            self.fr_vocab.add_sentence(fr)
            self.en_vocab.add_sentence(en)
            self.pairs.append((fr, en))

        self.fr_tokens = tokenize_and_pad([p[0] for p in self.pairs], self.fr_vocab)
        self.en_tokens = tokenize_and_pad([p[1] for p in self.pairs], self.en_vocab)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.fr_tokens[idx], self.en_tokens[idx]
