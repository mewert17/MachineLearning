import matplotlib.pyplot as plt
import csv
import os

def load_log(log_file):
    epochs, train_loss, val_loss, val_acc = [], [], [], []
    with open(log_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            val_acc.append(float(row["val_accuracy"]))
    return epochs, train_loss, val_loss, val_acc

def plot_metrics(model_type, seq_length, save_dir="outputs/plots"):
    log_file = f"outputs/logs/{model_type}_seq{seq_length}.csv"
    if not os.path.exists(log_file):
        print(f"Log file {log_file} not found!")
        return
    epochs, train_loss, val_loss, val_acc = load_log(log_file)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train Loss", marker="o")
    plt.plot(epochs, val_loss, label="Val Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_type} (Seq={seq_length}) Loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_acc, label="Val Accuracy", marker="o", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_type} (Seq={seq_length}) Accuracy")
    plt.legend()
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{model_type}_seq{seq_length}.png"))
    plt.show()

if __name__ == "__main__":
    # Adjust model types and sequence lengths as needed
    for model_type in ["LSTM", "GRU"]:
        for seq_length in [20, 30, 50]:
            plot_metrics(model_type, seq_length)
