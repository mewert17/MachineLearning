# Problem1a.py
from train import train_model
import torch
import csv
import os

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_types = ["RNN", "LSTM", "GRU"]
    seq_lengths = [10, 20, 30]
    results = []

    for model_type in model_types:
        for seq_length in seq_lengths:
            print(f"Training {model_type} with sequence length {seq_length}")
            metrics, _, _ = train_model(model_type=model_type, seq_length=seq_length, epochs=50, batch_size=32, lr=0.001, device=device)
            result = {
                "model_type": model_type,
                "seq_length": seq_length,
                "final_train_loss": metrics["final_train_loss"],
                "final_val_loss": metrics["final_val_loss"],
                "final_val_accuracy": metrics["final_val_accuracy"],
                "total_training_time": metrics["total_training_time"]
            }
            results.append(result)
    
    # Print summary of results
    print("Experiment Results:")
    for res in results:
        print(res)
    
    # Save results to a CSV file in outputs/logs
    log_dir = "outputs/logs"
    os.makedirs(log_dir, exist_ok=True)
    summary_file = os.path.join(log_dir, "experiment_summary.csv")
    with open(summary_file, "w", newline="") as csvfile:
        fieldnames = ["model_type", "seq_length", "final_train_loss", "final_val_loss", "final_val_accuracy", "total_training_time"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            writer.writerow(res)

if __name__ == "__main__":
    main()
