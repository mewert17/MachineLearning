import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt

# ✅ Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ✅ Data Preprocessing & Augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ✅ Prevent Redundant Dataset Checks
cifar10_path = "./data/cifar-10-batches-py"
cifar100_path = "./data/cifar-100-python"

if not os.path.exists(cifar10_path):
    trainset_cifar10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset_cifar10 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
else:
    trainset_cifar10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    testset_cifar10 = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

if not os.path.exists(cifar100_path):
    trainset_cifar100 = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    testset_cifar100 = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
else:
    trainset_cifar100 = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=transform)
    testset_cifar100 = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform)

trainloader_cifar10 = torch.utils.data.DataLoader(trainset_cifar10, batch_size=64, shuffle=True, num_workers=2)
testloader_cifar10 = torch.utils.data.DataLoader(testset_cifar10, batch_size=64, shuffle=False, num_workers=2)

trainloader_cifar100 = torch.utils.data.DataLoader(trainset_cifar100, batch_size=64, shuffle=True, num_workers=2)
testloader_cifar100 = torch.utils.data.DataLoader(testset_cifar100, batch_size=64, shuffle=False, num_workers=2)

# ✅ Modified AlexNet for CIFAR-10 and CIFAR-100
class AlexNet_Modified(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(AlexNet_Modified, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 192, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ✅ Training Function
def train_model(model, trainloader, testloader, num_epochs=50, learning_rate=0.001, save_path="alexnet.pth"):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loss_list, val_loss_list, val_accuracy_list = [], [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss_list.append(running_loss / len(trainloader))

        model.eval()
        running_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss_list.append(running_loss / len(testloader))
        val_accuracy = 100 * correct / total
        val_accuracy_list.append(val_accuracy)

        print(f'Epoch {epoch+1}: Training Loss = {train_loss_list[-1]:.4f}, Validation Loss = {val_loss_list[-1]:.4f}, Accuracy = {val_accuracy:.2f}%')

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), f"models/{save_path}")
    print(f"✅ Model saved at models/{save_path}")

    return train_loss_list, val_loss_list, val_accuracy_list

# ✅ Plot Training Loss & Accuracy
def plot_results(train_loss, val_loss, val_accuracy, dataset_name):
    plt.figure()
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{dataset_name} - Training & Validation Loss')
    plt.legend()
    plt.savefig(f'models/{dataset_name}_AlexNet_Loss.png')
    plt.show()

    plt.figure()
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{dataset_name} - Validation Accuracy')
    plt.legend()
    plt.savefig(f'models/{dataset_name}_AlexNet_Accuracy.png')
    plt.show()

# ✅ Run Training for Both Datasets
def main():
    print("🚀 Training AlexNet on CIFAR-10...")
    model_cifar10 = AlexNet_Modified(num_classes=10, dropout_rate=0.5)
    train_loss_cifar10, val_loss_cifar10, val_accuracy_cifar10 = train_model(model_cifar10, trainloader_cifar10, testloader_cifar10, num_epochs=50, save_path="alexnet_cifar10.pth")
    plot_results(train_loss_cifar10, val_loss_cifar10, val_accuracy_cifar10, "CIFAR-10")

    print("\n🚀 Training AlexNet on CIFAR-100...")
    model_cifar100 = AlexNet_Modified(num_classes=100, dropout_rate=0.5)
    train_loss_cifar100, val_loss_cifar100, val_accuracy_cifar100 = train_model(model_cifar100, trainloader_cifar100, testloader_cifar100, num_epochs=50, save_path="alexnet_cifar100.pth")
    plot_results(train_loss_cifar100, val_loss_cifar100, val_accuracy_cifar100, "CIFAR-100")

    total_params = sum(p.numel() for p in model_cifar10.parameters())
    print(f"Total number of parameters in modified AlexNet: {total_params}")

if __name__ == "__main__":
    main()
