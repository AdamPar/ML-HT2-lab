import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a more complex CNN architecture
class ComplexCNN(nn.Module):
    def __init__(self):
        super(ComplexCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(256 * 3 * 3, 10)

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

# Function to calculate FLOPS for Conv2D layers
def calculate_flops_conv(layer, input_size):
    # Unpack input size
    in_channels, height, width = input_size
    
    # Calculate output size
    output_height = (height + 2 * layer.padding[0] - layer.kernel_size[0]) // layer.stride[0] + 1
    output_width = (width + 2 * layer.padding[1] - layer.kernel_size[1]) // layer.stride[1] + 1
    
    # Calculate FLOPS for the convolutional layer
    flops = 2 * output_height * output_width * layer.kernel_size[0] * layer.kernel_size[1] * in_channels * layer.out_channels
    return flops

# Function to calculate total FLOPS for the model
def calculate_total_flops(model, input_size):
    total_flops = 0
    for name, layer in model.named_children():
        if isinstance(layer, nn.Conv2d):
            flops = calculate_flops_conv(layer, input_size)
            if flops > 0:
                total_flops += flops
                # Update input size for the next layer
                input_size = (layer.out_channels,
                              (input_size[1] + 2 * layer.padding[0] - layer.kernel_size[0]) // layer.stride[0] + 1,
                              (input_size[2] + 2 * layer.padding[1] - layer.kernel_size[1]) // layer.stride[1] + 1)
            else:
                print(f"Warning: Layer {name} has zero FLOPS.")
    return total_flops

# Initialize the complex CNN
model = ComplexCNN()

# Assuming input size is (1, 28, 28) for MNIST
input_size = (1, 28, 28)

# Calculate FLOPS for the original model
original_flops = calculate_total_flops(model, input_size)

# Count original parameters
original_param_count = count_parameters(model)

# Apply HT2 to all convolutional layers
for name, layer in model.named_children():
    if isinstance(layer, nn.Conv2d):
        decomposed_layer = apply_ht2_decomposition(layer)
        print(f"Decomposed Layer {name}: {decomposed_layer}")  # Debugging line
        setattr(model, name, decomposed_layer)

# Count parameters after decomposition
reduced_param_count = count_parameters(model)

# Calculate FLOPS for the compressed model
compressed_flops = calculate_total_flops(model, input_size)
print(f"Compressed FLOPS after applying HT2: {compressed_flops}")  # Debugging line


# Calculate the ratio of reduced parameters
reduction_ratio = (original_param_count - reduced_param_count) / original_param_count

# Calculate FLOPS Compression Ratio (FCR)
if compressed_flops > 0:
    fcr = original_flops / compressed_flops
else:
    fcr = float('inf')  # or use another appropriate value

# Output results
print(f"Original Parameters: {original_param_count}")
print(f"Reduced Parameters: {reduced_param_count}")
print(f"Reduction Ratio: {reduction_ratio:.2%}")
print(f"Original FLOPS: {original_flops}")
print(f"Compressed FLOPS: {compressed_flops}")
print(f"FLOPS Compression Ratio: {fcr:.2f}")
