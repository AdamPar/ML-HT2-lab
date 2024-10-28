import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from thop import profile
from ht2 import ht2  # Import your ht2 function
from ht2_decomposition import ht2_decomposition  # Import your ht2_decomposition function

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.MaxPool2d(2)(x)
        x = nn.ReLU()(self.conv2(x))
        x = nn.MaxPool2d(2)(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x

# Load CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Initialize and train the model
model = SimpleCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop (same as before)
for epoch in range(5):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Calculate FLOPS and Params for the baseline model
input_tensor = torch.randn(1, 3, 32, 32)
baseline_flops, baseline_params = profile(model, inputs=(input_tensor,))

# Apply HT2 decomposition
new_layers = []
energy_threshold = 0.8
for layer in model.children():
    if isinstance(layer, nn.Conv2d):
        decomposed_layers = ht2(layer, energy_threshold)
        new_layers.extend(decomposed_layers)  # Collect decomposed layers
    else:
        new_layers.append(layer)

# Build a new sequential model
compressed_model = nn.Sequential(*new_layers)

# Calculate FLOPS and Params for the compressed model
compressed_flops, compressed_params = profile(compressed_model, inputs=(input_tensor,))

# Compute compression ratios
flops_compression_ratio = baseline_flops / compressed_flops
params_compression_ratio = baseline_params / compressed_params

# Print results
print(f"Baseline FLOPS: {baseline_flops}, Compressed FLOPS: {compressed_flops}, FLOPS Compression Ratio: {flops_compression_ratio:.2f}")
print(f"Baseline Params: {baseline_params}, Compressed Params: {compressed_params}, Params Compression Ratio: {params_compression_ratio:.2f}")
