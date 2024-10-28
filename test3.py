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
def apply_ht2_decomposition(layer, energy_threshold=0.25):
    from ht2 import ht2  # Ensure to import your HT2 implementation
    decomposed_layer = ht2(layer, energy_threshold)
    return decomposed_layer

# Function to count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Function to calculate FLOPs for convolutional layer
def calculate_conv_flops(layer, input_size):
    """Calculate FLOPs for convolutional layer"""
    batch_size = 1
    out_channels = layer.out_channels
    in_channels = layer.in_channels
    kernel_size = layer.kernel_size[0] * layer.kernel_size[1]
    h, w = input_size

    h_out = (h + 2 * layer.padding[0] - layer.kernel_size[0]) // layer.stride[0] + 1
    w_out = (w + 2 * layer.padding[1] - layer.kernel_size[1]) // layer.stride[1] + 1

    return batch_size * out_channels * in_channels * kernel_size * h_out * w_out

# Function to calculate FLOPs for linear layer
def calculate_linear_flops(layer):
    """Calculate FLOPs for linear layer"""
    return layer.in_features * layer.out_features

# Function to calculate total FLOPs for the model
def calculate_total_flops(model, input_size=(32, 32)):
    """Calculate total FLOPs for the model"""
    total_flops = 0
    current_size = input_size

    for module in model.children():
        if isinstance(module, nn.Conv2d):
            flops = calculate_conv_flops(module, current_size)
            total_flops += flops

            # Update size after max pooling
            current_size = (current_size[0] // 2, current_size[1] // 2)

        elif isinstance(module, nn.Linear):
            flops = calculate_linear_flops(module)
            total_flops += flops

    return total_flops

# Initialize the complex CNN
model = ComplexCNN()

# Count original parameters
original_param_count = count_parameters(model)

# Calculate original FLOPS
original_flops = calculate_total_flops(model, input_size=(28, 28))

# Apply HT2 to all convolutional layers
for name, layer in model.named_children():
    if isinstance(layer, nn.Conv2d):
        decomposed_layer = apply_ht2_decomposition(layer)
        setattr(model, name, decomposed_layer)

# Count parameters after decomposition
reduced_param_count = count_parameters(model)

# Calculate reduced FLOPS
reduced_flops = calculate_total_flops(model, input_size=(28, 28))

# Calculate the ratio of reduced parameters
reduction_ratio = (original_param_count - reduced_param_count) / original_param_count

# Calculate FLOPS compression ratio
flops_reduction_ratio = original_flops / reduced_flops

# Print results
print(f"Original Parameters: {original_param_count}")
print(f"Reduced Parameters: {reduced_param_count}")
print(f"Reduction Ratio: {reduction_ratio:.2%}")
print(f"Original FLOPS: {original_flops}")
print(f"Reduced FLOPS: {reduced_flops}")
print(f"FLOPS Compression Ratio: {flops_reduction_ratio:.2f}")
