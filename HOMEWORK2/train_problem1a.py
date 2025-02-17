import torch
import torch.optim as optim
import torch.nn as nn
from Problem1a import AlexNet_Modified
from dataloader import trainloader_cifar10, testloader_cifar10

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model
model = AlexNet_Modified(num_classes=10).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training and validation lists
train_loss_list = []
val_loss_list = []
val_accuracy_list = []

# Training loop
num_epochs = 10  # Train for more epochs if needed
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in trainloader_cifar10:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    train_loss_list.append(running_loss / len(trainloader_cifar10))

    # Validation
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader_cifar10:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss_list.append(running_loss / len(testloader_cifar10))
    val_accuracy = 100 * correct / total
    val_accuracy_list.append(val_accuracy)

    print(f'Epoch {epoch+1}, Training loss: {train_loss_list[-1]:.4f}, Validation loss: {val_loss_list[-1]:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

print(f'Final Validation Accuracy: {val_accuracy_list[-1]:.2f}%')

# Save the model
torch.save(model.state_dict(), "models/alexnet_cifar10.pth")
print("✅ Model saved successfully!")

# Report number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'Total number of parameters in modified AlexNet: {total_params}')
