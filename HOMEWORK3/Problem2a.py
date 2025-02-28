import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import requests
import time

##########################
# Data Loader for Tiny Shakespeare
##########################

# Step 1: Download the dataset
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = requests.get(url)
text = response.text  # Entire text data

# Step 2: Prepare the dataset given a sequence length
def prepare_shakespeare_data(sequence_length):
    # Build character vocabulary
    chars = sorted(list(set(text)))
    char_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_char = {i: ch for i, ch in enumerate(chars)}
    
    # Encode text into integers
    encoded_text = [char_to_int[ch] for ch in text]
    
    # Create sequences and targets
    sequences = []
    targets = []
    for i in range(len(encoded_text) - sequence_length):
        seq = encoded_text[i:i+sequence_length]
        target = encoded_text[i+sequence_length]
        sequences.append(seq)
        targets.append(target)
    
    # Convert lists to PyTorch tensors
    sequences = torch.tensor(sequences, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return sequences, targets, char_to_int, int_to_char, len(chars)

# Step 3: Define a Dataset class
class CharDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index], self.targets[index]

def get_dataloaders(sequence_length, batch_size, test_split=0.2):
    sequences, targets, char_to_int, int_to_char, vocab_size = prepare_shakespeare_data(sequence_length)
    dataset = CharDataset(sequences, targets)
    
    # Split indices
    dataset_size = len(dataset)
    split = int(np.floor(test_split * dataset_size))
    indices = np.arange(dataset_size)
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, char_to_int, int_to_char, vocab_size

##########################
# Model Definitions
##########################

class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers=1):
        super(CharLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        # Use output of the last time step
        out = self.fc(out[:, -1, :])
        return out, hidden

class CharGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers=1):
        super(CharGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.gru(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

##########################
# Training Function
##########################

def train_shakespeare(model_type, sequence_length, epochs=30, batch_size=128, lr=0.001, device='cpu'):
    # Get dataloaders
    train_loader, val_loader, char_to_int, int_to_char, vocab_size = get_dataloaders(sequence_length, batch_size)
    
    # Hyperparameters for model
    embed_dim = 64
    hidden_size = 256  # You can adjust this as a hyperparameter
    num_layers = 2     # Adjust the number of hidden layers
    
    # Initialize model based on model_type
    if model_type == "LSTM":
        model = CharLSTM(vocab_size, embed_dim, hidden_size, num_layers).to(device)
    elif model_type == "GRU":
        model = CharGRU(vocab_size, embed_dim, hidden_size, num_layers).to(device)
    else:
        raise ValueError("Model type must be either 'LSTM' or 'GRU'")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # For logging metrics
    total_start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs, _ = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs, _ = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total
        
        print(f"Epoch {epoch}/{epochs} | Seq Len {sequence_length} | {model_type} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
    
    total_training_time = time.time() - total_start_time
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    metrics = {
        "final_train_loss": avg_train_loss,
        "final_val_loss": avg_val_loss,
        "final_val_accuracy": val_accuracy,
        "total_training_time": total_training_time,
        "model_parameters": model_params
    }
    return metrics, char_to_int, int_to_char

##########################
# Experiment Runner
##########################

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Experiment with sequence lengths 20 and 30 for both LSTM and GRU
    for model_type in ["LSTM", "GRU"]:
        for seq_length in [20, 30]:
            print(f"Training {model_type} with sequence length {seq_length}")
            metrics, _, _ = train_shakespeare(model_type, seq_length, epochs=30, batch_size=128, lr=0.001, device=device)
            print(f"Results for {model_type} (Seq {seq_length}): {metrics}")
    
    # Extended experiment: Increase sequence length to 50 for one model (example with LSTM)
    print("Training LSTM with sequence length 50")
    metrics, _, _ = train_shakespeare("LSTM", 50, epochs=30, batch_size=128, lr=0.001, device=device)
    print(f"Results for LSTM (Seq 50): {metrics}")
