import torch
from torch.utils.data import Dataset
import string

def normalize_string(s):
    # Lowercase, trim, and remove punctuation
    s = s.lower().strip()
    return s.translate(str.maketrans('', '', string.punctuation))

class Vocabulary:
    def __init__(self):
        # Initialize special tokens: <PAD>, <SOS>, <EOS>, and <UNK>
        self.word2index = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.word_count = {}
        self.n_words = 4  # Starting index for new words
    
    def add_sentence(self, sentence):
        # Normalize and then split the sentence
        sentence = normalize_string(sentence)
        for word in sentence.split():
            self.add_word(word)
    
    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.word_count[word] = 1
            self.n_words += 1
        else:
            self.word_count[word] += 1

def tokenize_and_pad(sentences, vocab):
    # Normalize all sentences first
    normalized_sentences = [normalize_string(s) for s in sentences]
    max_length = max(len(s.split()) for s in normalized_sentences) + 2  # +2 for <SOS> and <EOS>
    tokenized_sentences = []
    for sentence in normalized_sentences:
        tokens = [vocab.word2index["<SOS>"]] + \
                 [vocab.word2index.get(word, vocab.word2index["<UNK>"]) for word in sentence.split()] + \
                 [vocab.word2index["<EOS>"]]
        padded_tokens = tokens + [vocab.word2index["<PAD>"]] * (max_length - len(tokens))
        tokenized_sentences.append(padded_tokens)
    return torch.tensor(tokenized_sentences, dtype=torch.long)

class EngFrDataset(Dataset):
    def __init__(self, pairs):
        self.eng_vocab = Vocabulary()
        self.fr_vocab = Vocabulary()
        self.pairs = []
        
        # First pass: build vocabularies
        for eng, fr in pairs:
            self.eng_vocab.add_sentence(eng)
            self.fr_vocab.add_sentence(fr)
            self.pairs.append((eng, fr))
        
        # Second pass: tokenize and pad
        self.eng_sentences = [pair[0] for pair in self.pairs]
        self.fr_sentences = [pair[1] for pair in self.pairs]
        self.eng_tokens = tokenize_and_pad(self.eng_sentences, self.eng_vocab)
        self.fr_tokens = tokenize_and_pad(self.fr_sentences, self.fr_vocab)
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        return self.eng_tokens[idx], self.fr_tokens[idx]