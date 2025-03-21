import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import DataLoader, random_split

# Import dataset and english_to_french data
from data.dataset import EngFrDataset, english_to_french
# Import the GRU encoder and decoder from your models folder
from models.seq2seq_gru import EncoderGRU, DecoderGRU

# -----------------------------
# HYPERPARAMETERS & SETTINGS
# -----------------------------
EPOCHS = 20
BATCH_SIZE = 32
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 2
DROPOUT = 0.3  # If training is unstable, try setting this to 0.0
TEACHER_FORCING_RATIO = 0.5
LEARNING_RATE = 1e-4  # Lower learning rate can help avoid NaNs
CLIP = 5.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(encoder, decoder, dataloader, enc_opt, dec_opt, criterion, dataset):
    encoder.train()
    decoder.train()
    
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    
    for eng_batch, fr_batch in dataloader:
        eng_batch = eng_batch.to(device)
        fr_batch = fr_batch.to(device)
        
        enc_opt.zero_grad()
        dec_opt.zero_grad()
        
        batch_size = eng_batch.size(0)
        # Encode input sentences
        encoder_hidden = encoder.init_hidden(batch_size, device)
        encoder_outputs, encoder_hidden = encoder(eng_batch, encoder_hidden)
        
        # Decoder uses encoder's final hidden state
        decoder_hidden = encoder_hidden
        decoder_input = torch.tensor(
            [dataset.fr_vocab.word2index["<SOS>"]] * batch_size,
            device=device
        ).unsqueeze(1)  # shape: [batch_size, 1]
        
        max_len = fr_batch.size(1)
        loss = 0.0
        
        # Loop over each time step (starting at t=1, since index 0 is <SOS>)
        for t in range(1, max_len):
            output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            target = fr_batch[:, t]
            
            loss += criterion(output, target)
            
            # Compute token-level accuracy (ignoring <PAD> tokens)
            _, predicted = output.max(1)
            mask = (target != dataset.fr_vocab.word2index["<PAD>"])
            total_correct += (predicted[mask] == target[mask]).sum().item()
            total_tokens += mask.sum().item()
            
            # Teacher forcing decision
            if random.random() < TEACHER_FORCING_RATIO:
                decoder_input = target.unsqueeze(1)
            else:
                decoder_input = predicted.unsqueeze(1)
        
        loss.backward()
        # Clip gradients to help prevent exploding gradients and NaNs
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), CLIP)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), CLIP)
        enc_opt.step()
        dec_opt.step()
        
        total_loss += loss.item() / (max_len - 1)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    return avg_loss, accuracy

def evaluate(encoder, decoder, dataloader, criterion, dataset):
    encoder.eval()
    decoder.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for eng_batch, fr_batch in dataloader:
            eng_batch = eng_batch.to(device)
            fr_batch = fr_batch.to(device)
            
            batch_size = eng_batch.size(0)
            encoder_hidden = encoder.init_hidden(batch_size, device)
            encoder_outputs, encoder_hidden = encoder(eng_batch, encoder_hidden)
            
            decoder_hidden = encoder_hidden
            decoder_input = torch.tensor(
                [dataset.fr_vocab.word2index["<SOS>"]] * batch_size,
                device=device
            ).unsqueeze(1)
            max_len = fr_batch.size(1)
            loss = 0.0
            
            for t in range(1, max_len):
                output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                target = fr_batch[:, t]
                loss += criterion(output, target)
                
                _, predicted = output.max(1)
                mask = (target != dataset.fr_vocab.word2index["<PAD>"])
                total_correct += (predicted[mask] == target[mask]).sum().item()
                total_tokens += mask.sum().item()
                
                decoder_input = predicted.unsqueeze(1)
            
            total_loss += loss.item() / (max_len - 1)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    return avg_loss, accuracy

def translate_sentence(sentence, encoder, decoder, dataset, max_length=50):
    """
    Translate an English sentence into French using the trained encoder-decoder model.
    """
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        # Tokenize the sentence and convert words to indices
        words = sentence.lower().strip().split()
        input_indices = [dataset.eng_vocab.word2index.get(w, dataset.eng_vocab.word2index["<UNK>"]) for w in words]
        input_indices = [dataset.eng_vocab.word2index["<SOS>"]] + input_indices + [dataset.eng_vocab.word2index["<EOS>"]]
        input_tensor = torch.tensor(input_indices, dtype=torch.long, device=device).unsqueeze(0)
        
        encoder_hidden = encoder.init_hidden(1, device)
        encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)
        decoder_hidden = encoder_hidden
        decoder_input = torch.tensor([dataset.fr_vocab.word2index["<SOS>"]], device=device).unsqueeze(1)
        
        translated_words = []
        for _ in range(max_length):
            output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            _, topi = output.topk(1)
            # Do not add an extra dimension here—keep the shape as [1, 1]
            decoder_input = topi.detach()
            next_word = topi.squeeze().item()
            if next_word == dataset.fr_vocab.word2index["<EOS>"]:
                break
            translated_words.append(dataset.fr_vocab.index2word[next_word])
        
        return " ".join(translated_words)

def main():
    # Load dataset using the provided english_to_french list
    dataset = EngFrDataset(english_to_french)
    
    # Split dataset into 80% training and 20% validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize encoder and decoder models
    encoder = EncoderGRU(
        input_size=dataset.eng_vocab.n_words,
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout_p=DROPOUT
    ).to(device)
    
    decoder = DecoderGRU(
        output_size=dataset.fr_vocab.n_words,
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout_p=DROPOUT
    ).to(device)
    
    # Define loss function and optimizers
    criterion = nn.NLLLoss(ignore_index=dataset.fr_vocab.word2index["<PAD>"])
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(encoder, decoder, train_loader,
                                                encoder_optimizer, decoder_optimizer,
                                                criterion, dataset)
        val_loss, val_acc = evaluate(encoder, decoder, val_loader, criterion, dataset)
        
        print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(encoder.state_dict(), "checkpoints/encoder_best.pth")
            torch.save(decoder.state_dict(), "checkpoints/decoder_best.pth")
            print("  --> Best model saved.")
    
    # Qualitative evaluation
    test_sentences = [
        "I am cold",
        "She is happy",
        "They are students"
    ]
    
    print("\nQualitative Evaluation:")
    for sentence in test_sentences:
        translation = translate_sentence(sentence, encoder, decoder, dataset)
        print(f"{sentence} -> {translation}")

if __name__ == "__main__":
    main()
