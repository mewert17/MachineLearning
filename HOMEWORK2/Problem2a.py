import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import sys
from contextlib import redirect_stdout

# Check device
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

# Redirect stdout to /dev/null during dataset initialization to suppress verbose download logs.
with open(os.devnull, 'w') as fnull:
    with redirect_stdout(fnull):
        trainset_cifar10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
        testset_cifar10 = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
        trainset_cifar100 = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=transform_train)
        testset_cifar100 = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)

trainloader_cifar10 = torch.utils.data.DataLoader(trainset_cifar10, batch_size=128, shuffle=True, num_workers=2)
testloader_cifar10 = torch.utils.data.DataLoader(testset_cifar10, batch_size=100, shuffle=False, num_workers=2)
trainloader_cifar100 = torch.utils.data.DataLoader(trainset_cifar100, batch_size=128, shuffle=True, num_workers=2)
testloader_cifar100 = torch.utils.data.DataLoader(testset_cifar100, batch_size=100, shuffle=False, num_workers=2)

# ----------------------------
# VGG Model Definition
# ----------------------------
def make_layers(cfg, batch_norm=False, dropout_rate=0.0):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            # Optionally add dropout after each conv layer
            if dropout_rate > 0:
                layers += [nn.Dropout(dropout_rate)]
            in_channels = v
    return nn.Sequential(*layers)

# VGG-16 configuration (we use a modified version for CIFAR)
cfg_vgg16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M']

class VGG(nn.Module):
    def __init__(self, features, num_classes=10, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        # For CIFAR images (32x32), after four pooling layers, feature maps are 2x2.
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        if init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# Helper function to instantiate a VGG-16 model.
def get_vgg16(dropout_rate=0.0, num_classes=10):
    features = make_layers(cfg_vgg16, batch_norm=False, dropout_rate=dropout_rate)
    model = VGG(features, num_classes=num_classes)
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ----------------------------
# Training, Evaluation, & Plotting Functions
# ----------------------------
def train_model(model, trainloader, testloader, num_epochs=50, learning_rate=0.001, save_path="vgg.pth"):
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
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        avg_val_loss = running_loss / len(testloader)
        accuracy = 100. * correct / total
        val_loss_list.append(avg_val_loss)
        val_accuracy_list.append(accuracy)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {accuracy:.2f}%")
    
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
# Main Function: Training on CIFAR-10 and CIFAR-100
# ----------------------------
def main():
    print("Training VGG16 for CIFAR-10 without dropout in conv layers...")
    model_vgg_cifar10 = get_vgg16(dropout_rate=0.0, num_classes=10)
    print("Number of parameters (VGG16 CIFAR-10, no dropout):", count_parameters(model_vgg_cifar10))
    train_loss_vgg10, val_loss_vgg10, val_acc_vgg10 = train_model(
        model_vgg_cifar10, trainloader_cifar10, testloader_cifar10,
        num_epochs=50, learning_rate=0.001, save_path="models/vgg16_cifar10_no_dropout.pth"
    )
    plot_results(train_loss_vgg10, val_loss_vgg10, val_acc_vgg10, "VGG16 CIFAR-10 (No Dropout)")
    
    print("\nTraining VGG16 for CIFAR-10 with dropout in conv layers...")
    model_vgg_dropout_cifar10 = get_vgg16(dropout_rate=0.3, num_classes=10)
    train_loss_vgg10_drop, val_loss_vgg10_drop, val_acc_vgg10_drop = train_model(
        model_vgg_dropout_cifar10, trainloader_cifar10, testloader_cifar10,
        num_epochs=50, learning_rate=0.001, save_path="models/vgg16_cifar10_dropout.pth"
    )
    plot_results(train_loss_vgg10_drop, val_loss_vgg10_drop, val_acc_vgg10_drop, "VGG16 CIFAR-10 (Dropout)")
    
    print("\nTraining VGG16 for CIFAR-100 without dropout in conv layers...")
    model_vgg_cifar100 = get_vgg16(dropout_rate=0.0, num_classes=100)
    print("Number of parameters (VGG16 CIFAR-100, no dropout):", count_parameters(model_vgg_cifar100))
    train_loss_vgg100, val_loss_vgg100, val_acc_vgg100 = train_model(
        model_vgg_cifar100, trainloader_cifar100, testloader_cifar100,
        num_epochs=50, learning_rate=0.001, save_path="models/vgg16_cifar100_no_dropout.pth"
    )
    plot_results(train_loss_vgg100, val_loss_vgg100, val_acc_vgg100, "VGG16 CIFAR-100 (No Dropout)")
    
    print("\nTraining VGG16 for CIFAR-100 with dropout in conv layers...")
    model_vgg_dropout_cifar100 = get_vgg16(dropout_rate=0.3, num_classes=100)
    train_loss_vgg100_drop, val_loss_vgg100_drop, val_acc_vgg100_drop = train_model(
        model_vgg_dropout_cifar100, trainloader_cifar100, testloader_cifar100,
        num_epochs=50, learning_rate=0.001, save_path="models/vgg16_cifar100_dropout.pth"
    )
    plot_results(train_loss_vgg100_drop, val_loss_vgg100_drop, val_acc_vgg100_drop, "VGG16 CIFAR-100 (Dropout)")

if __name__ == "__main__":
    main()
