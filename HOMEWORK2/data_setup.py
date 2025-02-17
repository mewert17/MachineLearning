import torchvision
import torchvision.transforms as transforms

# Define dataset transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize between -1 and 1
])

# Download CIFAR-10
torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Download CIFAR-100
torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

print("✅ CIFAR-10 and CIFAR-100 downloaded successfully!")
