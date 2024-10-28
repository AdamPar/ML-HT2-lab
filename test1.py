import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 28 * 28, 10)  # Assuming input image size is 28x28 (e.g., MNIST)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

def apply_ht2_decomposition(layer, energy_threshold=0.95):
    from ht2 import ht2  # Make sure to import your HT2 implementation

    # Apply HT2 to the convolutional layer
    decomposed_layer = ht2(layer, energy_threshold)
    return decomposed_layer

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Initialize the simple CNN
model = SimpleCNN()

# Count original parameters
original_param_count = count_parameters(model)

# Decompose the first layer
decomposed_conv_layer = apply_ht2_decomposition(model.conv1)

# Replace the original convolutional layer with the decomposed one
model.conv1 = decomposed_conv_layer

# Count parameters after decomposition
reduced_param_count = count_parameters(model)

# Calculate the ratio of reduced parameters
reduction_ratio = (original_param_count - reduced_param_count) / original_param_count

# Reconstruction Error: The reconstruction error measures how well the decomposed tensor approximates the original tensor.

print(f"Original Parameters: {original_param_count}")
print(f"Reduced Parameters: {reduced_param_count}")
print(f"Reduction Ratio: {reduction_ratio:.2%}")
