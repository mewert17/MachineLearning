# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from models.rnn_models import CharRNN, CharLSTM, CharGRU
from utils import load_text, get_dataloaders, count_parameters

def train_model(model_type="RNN", seq_length=10, epochs=50, batch_size=32, lr=0.001, device='cpu'):
    # Load text from data file
    text = load_text("data/sequence.txt")
    train_loader, val_loader, char_to_ix, ix_to_char, vocab_size = get_dataloaders(text, seq_length, batch_size)
    
    # Model hyperparameters
    embed_dim = 64
    hidden_size = 128
    num_layers = 1  # Adjust if desired
    
    # Initialize model
    if model_type == "RNN":
        model = CharRNN(vocab_size, embed_dim, hidden_size, num_layers).to(device)
    elif model_type == "LSTM":
        model = CharLSTM(vocab_size, embed_dim, hidden_size, num_layers).to(device)
    elif model_type == "GRU":
        model = CharGRU(vocab_size, embed_dim, hidden_size, num_layers).to(device)
    else:
        raise ValueError("Invalid model type!")
    
    print(f"{model_type} model has {count_parameters(model)} parameters.")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Create directories for outputs if they don't exist
    checkpoint_dir = "outputs/checkpoints"
    log_dir = "outputs/logs"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Open log file to save training metrics
    log_file = os.path.join(log_dir, f"{model_type}_seq{seq_length}.csv")
    with open(log_file, "w") as f:
        f.write("epoch,train_loss,val_loss,val_accuracy,epoch_time\n")
    
    start_time = time.time()
    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        batch_count = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs, _ = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            batch_count += 1
        train_loss /= batch_count
        
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
        val_loss /= len(val_loader)
        val_accuracy = correct / total
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Time: {epoch_time:.2f}s")
        
        # Log metrics to CSV file
        with open(log_file, "a") as f:
            f.write(f"{epoch},{train_loss},{val_loss},{val_accuracy},{epoch_time}\n")
        
        # Save checkpoint every 10 epochs or at the final epoch
        if epoch % 10 == 0 or epoch == epochs:
            checkpoint_path = os.path.join(checkpoint_dir, f"{model_type}_seq{seq_length}_epoch{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
    
    total_time = time.time() - start_time
    metrics = {
        "final_train_loss": train_loss,
        "final_val_loss": val_loss,
        "final_val_accuracy": val_accuracy,
        "total_training_time": total_time
    }
    return metrics, char_to_ix, ix_to_char

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics, char_to_ix, ix_to_char = train_model(model_type="RNN", seq_length=10, epochs=50, batch_size=32, lr=0.001, device=device)
    print(metrics)
