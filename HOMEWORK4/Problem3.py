import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# --- Device setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --- Special tokens ---
SOS_token = 0
EOS_token = 1

# --- Build character vocabulary from your French↔English pairs ---
french_to_english = [
    ("J'ai froid", "I am cold"),
    ("Tu es fatigué", "You are tired"),
    ("Il a faim", "He is hungry"),
    ("Elle est heureuse", "She is happy"),
    ("Nous sommes amis", "We are friends"),
    ("Ils sont étudiants", "They are students")
]

all_chars = set()
for fr, en in french_to_english:
    all_chars.update(fr)
    all_chars.update(en)

char_to_index = {"SOS": SOS_token, "EOS": EOS_token}
index_to_char = {SOS_token: "<SOS>", EOS_token: "<EOS>"}
for i, ch in enumerate(sorted(all_chars), start=2):
    char_to_index[ch] = i
    index_to_char[i] = ch

# --- Dataset class ---
class TranslationDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        fr, en = self.pairs[idx]
        fr_idxs = [char_to_index[c] for c in fr] + [EOS_token]
        en_idxs = [char_to_index[c] for c in en] + [EOS_token]
        return torch.tensor(fr_idxs, dtype=torch.long), torch.tensor(en_idxs, dtype=torch.long)

# --- Encoder and Decoder ---
class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, x, hidden):
        out, hidden = self.gru(self.embedding(x).view(1,1,-1), hidden)
        return out, hidden

    def initHidden(self):
        return torch.zeros(1,1,self.gru.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, vocab_size, dropout_p=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.attn = nn.Linear(hidden_size*2, 100)
        self.attn_combine = nn.Linear(hidden_size*2, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input).view(1,1,-1))
        attn_weights = torch.softmax(self.attn(torch.cat((embedded[0], hidden[0]),1)), dim=1)
        context = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        rnn_input = self.attn_combine(torch.cat((embedded[0], context[0]),1)).unsqueeze(0)
        output, hidden = self.gru(rnn_input, hidden)
        return self.log_softmax(self.out(output[0])), hidden, attn_weights

# --- Training and Evaluation Helpers ---
def train(input_tensor, target_tensor, encoder, decoder, enc_opt, dec_opt, criterion):
    encoder_hidden = encoder.initHidden()
    enc_opt.zero_grad(); dec_opt.zero_grad()

    encoder_outputs = torch.zeros(100, encoder.gru.hidden_size, device=device)
    for i in range(input_tensor.size(0)):
        _, encoder_hidden = encoder(input_tensor[i].unsqueeze(0).to(device), encoder_hidden)
        encoder_outputs[i] = encoder_hidden[0]

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden
    loss = 0
    for i in range(target_tensor.size(0)):
        output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
        loss += criterion(output, target_tensor[i].unsqueeze(0).to(device))
        decoder_input = target_tensor[i].unsqueeze(0).to(device)

        if decoder_input.item() == EOS_token:
            break

    loss.backward()
    enc_opt.step(); dec_opt.step()
    return loss.item()/target_tensor.size(0)

def evaluate_sentence(sentence, encoder, decoder):
    with torch.no_grad():
        input_tensor = torch.tensor([char_to_index[c] for c in sentence] + [EOS_token], device=device)
        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(100, encoder.gru.hidden_size, device=device)
        for i in range(input_tensor.size(0)):
            _, encoder_hidden = encoder(input_tensor[i].unsqueeze(0), encoder_hidden)
            encoder_outputs[i] = encoder_hidden[0]

        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden
        decoded = []
        for _ in range(100):
            output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topi = output.argmax(1)
            if topi.item() == EOS_token:
                break
            decoded.append(index_to_char[topi.item()])
            decoder_input = topi.unsqueeze(0)

        return "".join(decoded)

# --- Main training loop ---
dataset = TranslationDataset(french_to_english)
train_size = int(0.8*len(dataset))
train_ds, val_ds = random_split(dataset, [train_size, len(dataset)-train_size])
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)

encoder = EncoderRNN(len(char_to_index), 256).to(device)
decoder = AttnDecoderRNN(256, len(char_to_index)).to(device)
enc_opt = optim.SGD(encoder.parameters(), lr=0.005)
dec_opt = optim.SGD(decoder.parameters(), lr=0.005)
criterion = nn.NLLLoss()

for epoch in range(1, 101):
    total_loss = 0
    for inp, tgt in train_loader:
        total_loss += train(inp.squeeze(), tgt.squeeze(), encoder, decoder, enc_opt, dec_opt, criterion)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Avg Loss: {total_loss/len(train_loader):.4f}")

print("\nQualitative Evaluation:")
for fr, _ in french_to_english:
    print(f"{fr} -> {evaluate_sentence(fr, encoder, decoder)}")
