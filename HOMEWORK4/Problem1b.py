import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# English→French pairs
english_to_french = [
    ("I am cold", "J'ai froid"),
    ("You are tired", "Tu es fatigué"),
    ("He is hungry", "Il a faim"),
    ("She is happy", "Elle est heureuse"),
    ("We are friends", "Nous sommes amis"),
    ("They are students", "Ils sont étudiants")
]

# Special tokens
SOS_token = 0
EOS_token = 1

# Build character vocabulary
all_chars = set()
for en, fr in english_to_french:
    all_chars.update(en)
    all_chars.update(fr)

char_to_index = {"SOS": SOS_token, "EOS": EOS_token}
index_to_char = {SOS_token: "<SOS>", EOS_token: "<EOS>"}
for i, ch in enumerate(sorted(all_chars), start=2):
    char_to_index[ch] = i
    index_to_char[i] = ch

# Dataset
class TranslationDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        en, fr = self.pairs[idx]
        en_idxs = [char_to_index[c] for c in en] + [EOS_token]
        fr_idxs = [char_to_index[c] for c in fr] + [EOS_token]
        return torch.tensor(en_idxs, dtype=torch.long), torch.tensor(fr_idxs, dtype=torch.long)

# Encoder
class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, x, hidden):
        output, hidden = self.gru(self.embedding(x).view(1,1,-1), hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1,1,self.hidden_size, device=device)

# Decoder
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        output, hidden = self.gru(self.embedding(x).view(1,1,-1), hidden)
        return self.softmax(self.out(output[0])), hidden

# Train function
def train(en_tensor, fr_tensor, encoder, decoder, enc_opt, dec_opt, criterion):
    encoder_hidden = encoder.initHidden()
    enc_opt.zero_grad(); dec_opt.zero_grad()

    loss = 0
    # Encode
    for ei in range(en_tensor.size(0)):
        _, encoder_hidden = encoder(en_tensor[ei].unsqueeze(0).to(device), encoder_hidden)
    # Decode
    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden
    for di in range(fr_tensor.size(0)):
        output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = output.topk(1)
        decoder_input = fr_tensor[di].unsqueeze(0).to(device)  # teacher forcing
        loss += criterion(output, fr_tensor[di].unsqueeze(0).to(device))
        if decoder_input.item() == EOS_token:
            break

    loss.backward()
    enc_opt.step(); dec_opt.step()
    return loss.item()/fr_tensor.size(0)

# Evaluate
def evaluate_sentence(sentence, encoder, decoder):
    with torch.no_grad():
        tensor = torch.tensor([char_to_index[c] for c in sentence] + [EOS_token], device=device)
        hidden = encoder.initHidden()
        for i in range(tensor.size(0)):
            _, hidden = encoder(tensor[i].unsqueeze(0), hidden)
        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoded = []
        for _ in range(100):
            output, hidden = decoder(decoder_input, hidden)
            topi = output.argmax(1)
            if topi.item() == EOS_token:
                break
            decoded.append(index_to_char[topi.item()])
            decoder_input = topi.unsqueeze(0)
        return "".join(decoded)

# Hyperparams
hidden_size = 256
lr = 0.005
n_epochs = 100

# Data setup
dataset = TranslationDataset(english_to_french)
train_size = int(0.8*len(dataset))
train_ds, val_ds = random_split(dataset, [train_size, len(dataset)-train_size])
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)

# Model init
vocab_size = len(char_to_index)
encoder = EncoderRNN(vocab_size, hidden_size).to(device)
decoder = DecoderRNN(hidden_size, vocab_size).to(device)
enc_opt = optim.SGD(encoder.parameters(), lr=lr)
dec_opt = optim.SGD(decoder.parameters(), lr=lr)
criterion = nn.NLLLoss()

# Training
for epoch in range(1, n_epochs+1):
    total_loss = 0
    for en, fr in train_loader:
        total_loss += train(en.squeeze(), fr.squeeze(), encoder, decoder, enc_opt, dec_opt, criterion)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Avg Loss: {total_loss/len(train_loader):.4f}")

# Qualitative Evaluation
print("\nQualitative Results:")
for en, fr in english_to_french:
    print(f"{en} -> {evaluate_sentence(en, encoder, decoder)}")
