import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a more complex CNN architecture
class ComplexCNN(nn.Module):
    def __init__(self):
        super(ComplexCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)  # Increased output channels
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)  # Added another layer
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)  # Another layer
        self.fc1 = nn.Linear(256 * 3 * 3, 10)  # Adjusted based on output size after pooling

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# Function to apply HT2 Decomposition
def apply_ht2_decomposition(layer, energy_threshold=0.5):
    from ht2 import ht2  # Ensure to import your HT2 implementation
    decomposed_layer = ht2(layer, energy_threshold)
    return decomposed_layer

# Function to count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Initialize the complex CNN
model = ComplexCNN()

# Count original parameters
original_param_count = count_parameters(model)

# Apply HT2 to all convolutional layers
for name, layer in model.named_children():
    if isinstance(layer, nn.Conv2d):
        decomposed_layer = apply_ht2_decomposition(layer)
        setattr(model, name, decomposed_layer)

# Count parameters after decomposition
reduced_param_count = count_parameters(model)

# Calculate the ratio of reduced parameters
reduction_ratio = (original_param_count - reduced_param_count) / original_param_count

print(f"Original Parameters: {original_param_count}")
print(f"Reduced Parameters: {reduced_param_count}")
print(f"Reduction Ratio: {reduction_ratio:.2%}")
