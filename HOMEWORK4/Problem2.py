import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import DataLoader, random_split

from data.dataset import EngFrDataset, english_to_french
from models.seq2seq_attention import EncoderGRU, DecoderGRUWithAttention as DecoderGRU

# Hyperparameters
EPOCHS = 100
BATCH_SIZE = 16
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 2
DROPOUT = 0.1
TEACHER_FORCING_RATIO = 1.0
LEARNING_RATE = 0.001
CLIP = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(encoder, decoder, loader, enc_opt, dec_opt, criterion, dataset):
    encoder.train()
    decoder.train()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for eng_batch, fr_batch in loader:
        eng_batch, fr_batch = eng_batch.to(device), fr_batch.to(device)
        enc_opt.zero_grad(); dec_opt.zero_grad()

        batch_size = eng_batch.size(0)
        enc_hidden = encoder.init_hidden(batch_size, device)
        enc_out, enc_hidden = encoder(eng_batch, enc_hidden)

        dec_hidden = enc_hidden
        dec_input = torch.full((batch_size,1),
                               dataset.fr_vocab.word2index["<SOS>"],
                               dtype=torch.long, device=device)

        loss = 0.0
        max_len = fr_batch.size(1)
        for t in range(1, max_len):
            output, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_out)
            target = fr_batch[:,t]
            loss += criterion(output, target)

            _, pred = output.max(1)
            mask = (target != dataset.fr_vocab.word2index["<PAD>"])
            total_correct += (pred[mask]==target[mask]).sum().item()
            total_tokens += mask.sum().item()

            dec_input = (target.unsqueeze(1)
                         if random.random() < TEACHER_FORCING_RATIO
                         else pred.unsqueeze(1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), CLIP)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), CLIP)
        enc_opt.step(); dec_opt.step()

        total_loss += loss.item()

    avg_loss = total_loss / (len(loader)*(max_len-1))
    accuracy = total_correct / total_tokens
    return avg_loss, accuracy

def evaluate(encoder, decoder, loader, criterion, dataset):
    encoder.eval(); decoder.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for eng_batch, fr_batch in loader:
            eng_batch, fr_batch = eng_batch.to(device), fr_batch.to(device)
            batch_size = eng_batch.size(0)
            enc_hidden = encoder.init_hidden(batch_size, device)
            enc_out, enc_hidden = encoder(eng_batch, enc_hidden)

            dec_hidden = enc_hidden
            dec_input = torch.full((batch_size,1),
                                   dataset.fr_vocab.word2index["<SOS>"],
                                   dtype=torch.long, device=device)

            loss = 0.0
            max_len = fr_batch.size(1)
            for t in range(1, max_len):
                output, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_out)
                target = fr_batch[:,t]
                loss += criterion(output, target)

                _, pred = output.max(1)
                mask = (target != dataset.fr_vocab.word2index["<PAD>"])
                total_correct += (pred[mask]==target[mask]).sum().item()
                total_tokens += mask.sum().item()

                dec_input = pred.unsqueeze(1)

            total_loss += loss.item()

    avg_loss = total_loss / (len(loader)*(max_len-1))
    accuracy = total_correct / total_tokens
    return avg_loss, accuracy

def translate(sentence, encoder, decoder, dataset, max_length=50):
    encoder.eval(); decoder.eval()
    with torch.no_grad():
        tokens = sentence.lower().split()
        indices = [dataset.eng_vocab.word2index.get(w, dataset.eng_vocab.word2index["<UNK>"]) for w in tokens]
        indices = [dataset.eng_vocab.word2index["<SOS>"]] + indices + [dataset.eng_vocab.word2index["<EOS>"]]
        tensor = torch.tensor(indices, dtype=torch.long, device=device).unsqueeze(0)

        enc_hidden = encoder.init_hidden(1, device)
        enc_out, enc_hidden = encoder(tensor, enc_hidden)

        dec_hidden = enc_hidden
        dec_input = torch.tensor([[dataset.fr_vocab.word2index["<SOS>"]]], device=device)

        result = []
        for _ in range(max_length):
            output, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_out)
            topi = output.argmax(1)
            if topi.item() == dataset.fr_vocab.word2index["<EOS>"]:
                break
            result.append(dataset.fr_vocab.index2word[topi.item()])
            dec_input = topi.unsqueeze(1)

        return " ".join(result)

def main():
    dataset = EngFrDataset(english_to_french)
    train_size = int(0.8*len(dataset))
    train_ds, val_ds = random_split(dataset, [train_size, len(dataset)-train_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

    encoder = EncoderGRU(dataset.eng_vocab.n_words, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
    decoder = DecoderGRU(dataset.fr_vocab.n_words, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)

    criterion = nn.NLLLoss(ignore_index=dataset.fr_vocab.word2index["<PAD>"], reduction="sum")
    enc_opt = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
    dec_opt = optim.Adam(decoder.parameters(), lr=LEARNING_RATE)

    best_val = float('inf')
    for epoch in range(1, EPOCHS+1):
        tr_loss, tr_acc = train_one_epoch(encoder, decoder, train_loader, enc_opt, dec_opt, criterion, dataset)
        val_loss, val_acc = evaluate(encoder, decoder, val_loader, criterion, dataset)
        print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(encoder.state_dict(), "checkpoints/encoder_attn.pth")
            torch.save(decoder.state_dict(), "checkpoints/decoder_attn.pth")

    print("\nQualitative Evaluation:")
    for sent in ["I am cold", "She is happy", "They are students"]:
        print(f"{sent} -> {translate(sent, encoder, decoder, dataset)}")

if __name__ == "__main__":
    main()
