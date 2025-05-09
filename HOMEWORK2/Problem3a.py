import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import sys
from contextlib import redirect_stdout

# ----------------------------
# Device Check
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------
# Data Preprocessing & Loading
# ----------------------------
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Suppress download logs if you already have the datasets
with open(os.devnull, 'w') as fnull:
    with redirect_stdout(fnull):
        trainset_cifar10 = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=False, transform=transform_train)
        testset_cifar10 = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=False, transform=transform_test)
        trainset_cifar100 = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=False, transform=transform_train)
        testset_cifar100 = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=False, transform=transform_test)

trainloader_cifar10 = torch.utils.data.DataLoader(
    trainset_cifar10, batch_size=128, shuffle=True, num_workers=2)
testloader_cifar10 = torch.utils.data.DataLoader(
    testset_cifar10, batch_size=100, shuffle=False, num_workers=2)

trainloader_cifar100 = torch.utils.data.DataLoader(
    trainset_cifar100, batch_size=128, shuffle=True, num_workers=2)
testloader_cifar100 = torch.utils.data.DataLoader(
    testset_cifar100, batch_size=100, shuffle=False, num_workers=2)

# ----------------------------
# ResNet-11 Definition
# ----------------------------

class BasicBlock(nn.Module):
    """
    A basic residual block with two 3x3 conv layers and a skip connection.
    We optionally add dropout in between.
    """
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection if input and output have different dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU(inplace=True)(out)
        
        out = self.dropout(out)  # dropout between conv1 and conv2
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip connection
        out += self.shortcut(x)
        out = nn.ReLU(inplace=True)(out)
        return out

class ResNet11(nn.Module):
    """
    Example ResNet-11 architecture:
      - Initial conv layer + BN + ReLU
      - 4 BasicBlock "stages" (each has 2 conv layers) => total 1 + (2*4) + final FC = 1 + 8 + 1 = 10
        (We add one more FC or layer if needed to strictly call it "11" from lecture.)
      - For CIFAR, we do global average pooling at the end.
    """
    def __init__(self, block=BasicBlock, num_classes=10, dropout_rate=0.0):
        super(ResNet11, self).__init__()
        self.in_channels = 64
        
        # Initial conv
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Layers
        self.layer1 = self._make_layer(block, 64, num_blocks=1, stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(block, 128, num_blocks=1, stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(block, 256, num_blocks=1, stride=2, dropout_rate=dropout_rate)
        self.layer4 = self._make_layer(block, 512, num_blocks=1, stride=2, dropout_rate=dropout_rate)
        
        # Extra block or conv to bring total to "ResNet-11"
        # (If your lecture version differs, adapt as needed.)
        self.extra_block = self._make_layer(block, 512, num_blocks=1, stride=1, dropout_rate=dropout_rate)
        
        # Global average pooling & FC
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block, out_channels, num_blocks, stride, dropout_rate):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s, dropout_rate=dropout_rate))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU(inplace=True)(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.extra_block(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def get_resnet11(num_classes=10, dropout_rate=0.0):
    return ResNet11(num_classes=num_classes, dropout_rate=dropout_rate)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ----------------------------
# Training & Plotting
# ----------------------------
def train_model(model, trainloader, testloader, num_epochs=50, learning_rate=0.001, save_path="resnet11.pth"):
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
        avg_train_loss = running_loss / len(trainloader)
        train_loss_list.append(avg_train_loss)
        
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        avg_val_loss = running_loss / len(testloader)
        accuracy = 100. * correct / total
        
        val_loss_list.append(avg_val_loss)
        val_accuracy_list.append(accuracy)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f} "
              f"Val Loss: {avg_val_loss:.4f} "
              f"Val Acc: {accuracy:.2f}%")
    
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print("Model saved to", save_path)
    
    return train_loss_list, val_loss_list, val_accuracy_list

def plot_results(train_loss, val_loss, val_accuracy, model_name):
    epochs = range(1, len(train_loss) + 1)
    plt.figure()
    plt.plot(epochs, train_loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Loss")
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(epochs, val_accuracy, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{model_name} Accuracy")
    plt.legend()
    plt.show()

# ----------------------------
# Main: Train ResNet-11 on CIFAR-10 and CIFAR-100
# ----------------------------
def main():
    # ----- CIFAR-10 -----
    print("\n=== ResNet-11 on CIFAR-10 (No Dropout) ===")
    model_c10 = get_resnet11(num_classes=10, dropout_rate=0.0)
    print("Number of parameters:", count_parameters(model_c10))
    train_loss_c10, val_loss_c10, val_acc_c10 = train_model(
        model_c10, trainloader_cifar10, testloader_cifar10,
        num_epochs=30, learning_rate=0.001, save_path="models/resnet11_cifar10_no_dropout.pth"
    )
    plot_results(train_loss_c10, val_loss_c10, val_acc_c10, "ResNet-11 CIFAR-10 (No Dropout)")
    
    print("\n=== ResNet-11 on CIFAR-10 (Dropout) ===")
    model_c10_drop = get_resnet11(num_classes=10, dropout_rate=0.3)
    train_loss_c10_drop, val_loss_c10_drop, val_acc_c10_drop = train_model(
        model_c10_drop, trainloader_cifar10, testloader_cifar10,
        num_epochs=30, learning_rate=0.001, save_path="models/resnet11_cifar10_dropout.pth"
    )
    plot_results(train_loss_c10_drop, val_loss_c10_drop, val_acc_c10_drop, "ResNet-11 CIFAR-10 (Dropout)")
    
    # ----- CIFAR-100 -----
    print("\n=== ResNet-11 on CIFAR-100 (No Dropout) ===")
    model_c100 = get_resnet11(num_classes=100, dropout_rate=0.0)
    print("Number of parameters:", count_parameters(model_c100))
    train_loss_c100, val_loss_c100, val_acc_c100 = train_model(
        model_c100, trainloader_cifar100, testloader_cifar100,
        num_epochs=30, learning_rate=0.001, save_path="models/resnet11_cifar100_no_dropout.pth"
    )
    plot_results(train_loss_c100, val_loss_c100, val_acc_c100, "ResNet-11 CIFAR-100 (No Dropout)")
    
    print("\n=== ResNet-11 on CIFAR-100 (Dropout) ===")
    model_c100_drop = get_resnet11(num_classes=100, dropout_rate=0.3)
    train_loss_c100_drop, val_loss_c100_drop, val_acc_c100_drop = train_model(
        model_c100_drop, trainloader_cifar100, testloader_cifar100,
        num_epochs=30, learning_rate=0.001, save_path="models/resnet11_cifar100_dropout.pth"
    )
    plot_results(train_loss_c100_drop, val_loss_c100_drop, val_acc_c100_drop, "ResNet-11 CIFAR-100 (Dropout)")

if __name__ == "__main__":
    main()
