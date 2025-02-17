import torch
import matplotlib.pyplot as plt

# Load training results
checkpoint_cifar10 = torch.load("models/alexnet_cifar10_no_dropout.pth")
checkpoint_cifar100 = torch.load("models/alexnet_cifar100_no_dropout.pth")

train_loss_no_dropout = checkpoint_cifar10["train_loss"]
val_loss_no_dropout = checkpoint_cifar10["val_loss"]
val_accuracy_no_dropout = checkpoint_cifar10["val_accuracy"]

train_loss_no_dropout_100 = checkpoint_cifar100["train_loss"]
val_loss_no_dropout_100 = checkpoint_cifar100["val_loss"]
val_accuracy_no_dropout_100 = checkpoint_cifar100["val_accuracy"]

# Function to plot results
def plot_results(train_loss, val_loss, val_accuracy, dataset_name):
    plt.figure()
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{dataset_name} - Training & Validation Loss (No Dropout)')
    plt.legend()
    plt.savefig(f'models/{dataset_name}_AlexNet_Loss_NoDropout.png')
    plt.show()

    plt.figure()
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{dataset_name} - Validation Accuracy (No Dropout)')
    plt.legend()
    plt.savefig(f'models/{dataset_name}_AlexNet_Accuracy_NoDropout.png')
    plt.show()

# Generate and save plots
plot_results(train_loss_no_dropout, val_loss_no_dropout, val_accuracy_no_dropout, "CIFAR-10")
plot_results(train_loss_no_dropout_100, val_loss_no_dropout_100, val_accuracy_no_dropout_100, "CIFAR-100")

print("✅ No-Dropout plots saved in 'models/' folder.")
