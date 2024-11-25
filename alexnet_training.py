import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from ht2_implementation import ht2

class DecomposedBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, original_block, energy_threshold=0.95):
        super(DecomposedBasicBlock, self).__init__()
        
        # Decompose the first conv layer
        self.conv1_decomposed = nn.Sequential(
            *ht2(original_block.conv1, energy_threshold)
        )
        self.bn1 = original_block.bn1
        
        # Decompose the second conv layer
        self.conv2_decomposed = nn.Sequential(
            *ht2(original_block.conv2, energy_threshold)
        )
        self.bn2 = original_block.bn2
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        
        if original_block.downsample is not None:
            downsample_conv = original_block.downsample[0]
            decomposed_downsample = ht2(downsample_conv, energy_threshold)
            self.downsample = nn.Sequential(
                *decomposed_downsample,
                original_block.downsample[1]  # Keep the original BatchNorm
            )

    def forward(self, x):
        identity = x

        out = self.conv1_decomposed(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2_decomposed(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class DecomposedAlexNet(nn.Module):
    def __init__(self, original_model, energy_threshold=0.95):
        super(DecomposedAlexNet, self).__init__()
        
        # Get the features from original AlexNet
        orig_features = original_model.features
        
        # Decompose convolutional layers and keep other layers as is
        self.features = nn.Sequential()
        for i, layer in enumerate(orig_features):
            if isinstance(layer, nn.Conv2d) and layer.kernel_size[0] > 1:
                # Apply HT2 decomposition to conv layers
                decomposed_layers = ht2(layer, energy_threshold)
                for j, decomp_layer in enumerate(decomposed_layers):
                    self.features.add_module(f'conv{i}_decomp_{j}', decomp_layer)
            else:
                # Keep non-conv layers as they are
                self.features.add_module(f'layer{i}', layer)
        
        # Keep classifier layers as they are
        self.avgpool = original_model.avgpool
        self.classifier = original_model.classifier

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def plot_training_curves(metrics, save_path, title_prefix=""):
    plt.figure(figsize=(8, 5))
    plt.plot(metrics['train_acc_top1'], label='Train')
    plt.plot(metrics['test_acc_top1'], label='Validation')
    plt.title(f'{title_prefix}Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_filename = f'{title_prefix.lower().replace(" ", "_")}training_curves.png'
    plt.savefig(os.path.join(save_path, plot_filename))
    plt.close()

def save_metrics(metrics, filename):
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=4)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_flops(model, input_size=(1, 3, 224, 224)):
    flops = 0
    
    def conv2d_flops(layer, input_shape):
        batch_size, in_channels, in_h, in_w = input_shape
        out_channels = layer.out_channels
        kernel_h, kernel_w = layer.kernel_size
        stride_h, stride_w = layer.stride if isinstance(layer.stride, tuple) else (layer.stride, layer.stride)
        padding_h, padding_w = layer.padding if isinstance(layer.padding, tuple) else (layer.padding, layer.padding)
        
        out_h = ((in_h + 2 * padding_h - kernel_h) // stride_h) + 1
        out_w = ((in_w + 2 * padding_w - kernel_w) // stride_w) + 1
        
        return 2 * kernel_h * kernel_w * in_channels * out_channels * out_h * out_w

    def linear_flops(layer, input_shape):
        return 2 * layer.in_features * layer.out_features * input_shape[0]

    def calculate_shape_after_conv(input_shape, conv_layer):
        batch_size, in_channels, in_h, in_w = input_shape
        padding_h, padding_w = conv_layer.padding if isinstance(conv_layer.padding, tuple) else (conv_layer.padding, conv_layer.padding)
        stride_h, stride_w = conv_layer.stride if isinstance(conv_layer.stride, tuple) else (conv_layer.stride, conv_layer.stride)
        kernel_h, kernel_w = conv_layer.kernel_size
        
        out_h = ((in_h + 2 * padding_h - kernel_h) // stride_h) + 1
        out_w = ((in_w + 2 * padding_w - kernel_w) // stride_w) + 1
        return (batch_size, conv_layer.out_channels, out_h, out_w)

    def calculate_shape_after_pool(input_shape, pool_layer):
        batch_size, channels, in_h, in_w = input_shape
        kernel_size = pool_layer.kernel_size if isinstance(pool_layer.kernel_size, tuple) else (pool_layer.kernel_size, pool_layer.kernel_size)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        stride = getattr(pool_layer, 'stride', kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        
        out_h = (in_h - kernel_size[0]) // stride[0] + 1
        out_w = (in_w - kernel_size[1]) // stride[1] + 1
        return (batch_size, channels, out_h, out_w)

    current_shape = input_size

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            flops += conv2d_flops(module, current_shape)
            current_shape = calculate_shape_after_conv(current_shape, module)
        elif isinstance(module, nn.Linear):
            if len(current_shape) > 2:
                current_shape = (current_shape[0], np.prod(current_shape[1:]))
            flops += linear_flops(module, current_shape)
            current_shape = (current_shape[0], module.out_features)
        elif isinstance(module, nn.MaxPool2d):
            current_shape = calculate_shape_after_pool(current_shape, module)

    return flops

def get_model_stats(model, input_size=(1, 3, 224, 224)):
    return {
        'parameters': count_parameters(model),
        'flops': count_flops(model, input_size)
    }

def evaluate_model(model, data_loader, device):
    model.eval()
    correct_1 = 0
    correct_5 = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            _, predicted = outputs.max(1)
            correct_1 += predicted.eq(labels).sum().item()
            
            _, top5_pred = outputs.topk(5, 1, True, True)
            top5_pred = top5_pred.t()
            correct_5 += top5_pred.eq(labels.view(1, -1).expand_as(top5_pred)).sum().item()
            
            total += labels.size(0)
    
    return correct_1/total * 100, correct_5/total * 100

def train_model(model, train_loader, test_loader, epochs, device, save_path, is_finetuning=False):    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    metrics = {
        'train_losses': [],
        'train_acc_top1': [],
        'train_acc_top5': [],
        'test_acc_top1': [],
        'test_acc_top5': []
    }
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_correct_1 = 0
        train_correct_5 = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, 
                         desc=f'{"Fine-tuning" if is_finetuning else "Training"} Epoch {epoch+1}/{epochs}')
        
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Calculate training accuracy
            _, predicted = outputs.max(1)
            train_correct_1 += predicted.eq(labels).sum().item()
            
            _, top5_pred = outputs.topk(5, 1, True, True)
            top5_pred = top5_pred.t()
            train_correct_5 += top5_pred.eq(labels.view(1, -1).expand_as(top5_pred)).sum().item()
            
            train_total += labels.size(0)
            running_loss += loss.item()
            
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct_1/train_total:.2f}%'
            })
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        train_acc_top1 = 100. * train_correct_1 / train_total
        train_acc_top5 = 100. * train_correct_5 / train_total
        test_acc_top1, test_acc_top5 = evaluate_model(model, test_loader, device)
        
        # Store metrics
        metrics['train_losses'].append(epoch_loss)
        metrics['train_acc_top1'].append(train_acc_top1)
        metrics['train_acc_top5'].append(train_acc_top5)
        metrics['test_acc_top1'].append(test_acc_top1)
        metrics['test_acc_top5'].append(test_acc_top5)
        
        # Plot current progress
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            plot_training_curves(
                metrics, 
                save_path, 
                "Fine-tuning " if is_finetuning else "Initial Training "
            )
    
    return model, metrics

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'alexnet_results_{timestamp}'
    os.makedirs(save_path, exist_ok=True)
    
    # Setup data loading with smaller image size for faster training
    transform = transforms.Compose([
        transforms.Resize(128),  # Smaller than standard AlexNet input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    print("Loading CIFAR-100 dataset...")
    train_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128,  # Larger batch size possible with AlexNet
                            shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, 
                           shuffle=False, num_workers=2)
    
    # 1. Load and adapt pretrained AlexNet
    print("Loading pretrained AlexNet...")
    model = models.alexnet(pretrained=True)
    # Modify the classifier for CIFAR-100
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 100)
    model = model.to(device)
    
    # Get initial model statistics
    initial_stats = get_model_stats(model, input_size=(1, 3, 128, 128))
    
    # 2. Train initial model
    print("Training initial model...")
    model, initial_metrics = train_model(
        model, train_loader, test_loader, 
        epochs=1,
        device=device, save_path=save_path
    )
    
    # Save initial model results
    plot_training_curves(initial_metrics, save_path, "Initial Model ")
    
    initial_results = {
        'accuracy_top1': initial_metrics['test_acc_top1'][-1],
        'accuracy_top5': initial_metrics['test_acc_top5'][-1],
        'parameters': initial_stats['parameters'],
        'flops': initial_stats['flops']
    }
    
    # 3. Create and train decomposed model
    print("Creating decomposed model...")
    decomposed_model = DecomposedAlexNet(model, energy_threshold=0.95)
    decomposed_model = decomposed_model.to(device)
    decomposed_stats = get_model_stats(decomposed_model, input_size=(1, 3, 128, 128))
    
    print("Training decomposed model...")
    decomposed_model, decomposed_metrics = train_model(
        decomposed_model, train_loader, test_loader,
        epochs=1,
        device=device, save_path=save_path
    )
    
    # 4. Fine-tune the decomposed model
    print("Fine-tuning decomposed model...")
    decomposed_model, finetuned_metrics = train_model(
        decomposed_model, train_loader, test_loader,
        epochs=1, device=device, save_path=save_path,
        is_finetuning=True
    )
    
    plot_training_curves(finetuned_metrics, save_path, "Fine-tuned Model ")
    
    # Calculate compression ratios
    flops_compression_ratio = initial_stats['flops'] / decomposed_stats['flops']
    params_compression_ratio = initial_stats['parameters'] / decomposed_stats['parameters']
    
    # Calculate accuracy changes
    delta_acc_top1 = finetuned_metrics['test_acc_top1'][-1] - initial_results['accuracy_top1']
    delta_acc_top5 = finetuned_metrics['test_acc_top5'][-1] - initial_results['accuracy_top5']
    
    # Save final results
    final_results = {
        'initial_model': initial_results,
        'final_accuracy_top1': finetuned_metrics['test_acc_top1'][-1],
        'final_accuracy_top5': finetuned_metrics['test_acc_top5'][-1],
        'parameters': decomposed_stats['parameters'],
        'flops': decomposed_stats['flops'],
        'flops_compression_ratio': float(flops_compression_ratio),
        'params_compression_ratio': float(params_compression_ratio),
        'delta_accuracy_top1': float(delta_acc_top1),
        'delta_accuracy_top5': float(delta_acc_top5)
    }
    
    save_metrics(final_results, os.path.join(save_path, 'alexnet_results.json'))

if __name__ == "__main__":
    main()
