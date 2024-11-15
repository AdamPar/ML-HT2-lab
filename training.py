import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import os
import numpy as np
from datetime import datetime
from ht2_implementation import ht2

class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 100)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class DecomposedConvNet(nn.Module):
    def __init__(self):
        super(DecomposedConvNet, self).__init__()
        self.features = nn.ModuleList([
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        ])
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 100)
        )

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        x = self.classifier(x)
        return x

def save_metrics(metrics, filename):
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=4)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_flops(model, input_size=(1, 3, 32, 32)):
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

def get_model_stats(model, input_size=(1, 3, 32, 32)):
    return {
        'parameters': count_parameters(model),
        'flops': count_flops(model, input_size)
    }

def evaluate_model(model, test_loader, device):
    model.eval()
    correct_1 = 0
    correct_5 = 0
    total = 0
    
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc='Testing')
        for images, labels in test_pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            _, predicted = outputs.max(1)
            correct_1 += predicted.eq(labels).sum().item()
            
            _, top5_pred = outputs.topk(5, 1, True, True)
            top5_pred = top5_pred.t()
            correct_5 += top5_pred.eq(labels.view(1, -1).expand_as(top5_pred)).sum().item()
            
            total += labels.size(0)
            
            test_pbar.set_postfix({
                'Top-1': f'{100.*correct_1/total:.2f}%',
                'Top-5': f'{100.*correct_5/total:.2f}%'
            })
    
    return correct_1/total * 100, correct_5/total * 100

def train_model(model, train_loader, test_loader, epochs, device, save_path, is_finetuning=False):    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    metrics = {
        'train_losses': [],
        'test_acc_top1': [],
        'test_acc_top5': []
    }
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        train_pbar = tqdm(train_loader, 
                         desc=f'{"Fine-tuning" if is_finetuning else "Training"} Epoch {epoch+1}/{epochs}')
        
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        epoch_loss = running_loss / len(train_loader)
        acc_top1, acc_top5 = evaluate_model(model, test_loader, device)
        
        metrics['train_losses'].append(epoch_loss)
        metrics['test_acc_top1'].append(acc_top1)
        metrics['test_acc_top5'].append(acc_top5)
    
    return model, metrics

def apply_ht2_to_model(model, energy_threshold=0.95):
    new_model = DecomposedConvNet()
    
    # First, copy weights from the original model
    original_state_dict = model.state_dict()
    new_state_dict = new_model.state_dict()
    
    # Copy non-conv weights (like classifier)
    for key in original_state_dict:
        if key in new_state_dict:
            new_state_dict[key] = original_state_dict[key]
    
    # Now apply HT2 decomposition to conv layers
    conv_idx = 0
    new_features = nn.ModuleList()
    
    for i, layer in enumerate(new_model.features):
        if isinstance(layer, nn.Conv2d) and layer.kernel_size[0] > 1:
            # Get original conv layer weights
            original_conv = model.features[i]
            # Apply HT2 decomposition
            decomposed_layers = ht2(original_conv, energy_threshold)
            # Add decomposed layers to new features
            new_features.extend([
                decomposed_layers[0],  # First conv
                decomposed_layers[1],  # Second conv
                decomposed_layers[2],  # Third conv
                decomposed_layers[3]   # Fourth conv
            ])
            conv_idx += 1
        else:
            # Keep non-conv layers as they are
            new_features.append(layer)
    
    # Replace features with decomposed version
    new_model.features = new_features
    
    return new_model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'results_{timestamp}'
    os.makedirs(save_path, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], 
                           std=[0.2675, 0.2565, 0.2761])
    ])
    
    train_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, 
                            shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, 
                           shuffle=False, num_workers=2)
    
    # 1. Train initial model
    model = SimpleConvNet().to(device)
    initial_stats = get_model_stats(model)
    model, initial_metrics = train_model(model, train_loader, test_loader, epochs=10, 
                                       device=device, save_path=save_path)
    
    # Save initial model results
    initial_results = {
        'accuracy_top1': initial_metrics['test_acc_top1'][-1],
        'accuracy_top5': initial_metrics['test_acc_top5'][-1],
        'parameters': initial_stats['parameters'],
        'flops': initial_stats['flops']
    }
    save_metrics(initial_results, f'{save_path}/initial_model_results.json')
    
    # 2. Apply HT2 decomposition and get new model
    decomposed_model = apply_ht2_to_model(model)
    decomposed_model = decomposed_model.to(device)
    decomposed_stats = get_model_stats(decomposed_model)
    
    # 3. Fine-tune the decomposed model
    decomposed_model, finetuned_metrics = train_model(
        decomposed_model, train_loader, test_loader, 
        epochs=5, device=device, save_path=save_path, 
        is_finetuning=True
    )
    
    # Save final finetuned results
    final_results = {
        'accuracy_top1': finetuned_metrics['test_acc_top1'][-1],
        'accuracy_top5': finetuned_metrics['test_acc_top5'][-1],
        'parameters': decomposed_stats['parameters'],
        'flops': decomposed_stats['flops']
    }
    save_metrics(final_results, f'{save_path}/finetuned_model_results.json')

if __name__ == "__main__":
    main()