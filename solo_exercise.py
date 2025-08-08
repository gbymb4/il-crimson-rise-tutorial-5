# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 14:14:48 2025

@author: taske
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("Solo Exercise: Custom MLP Architecture & Optimizer Comparison")
print("=" * 65)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data loading (provided - same as demo)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Data loaded: {len(train_dataset)} training, {len(test_dataset)} test samples")

# Training and validation functions (provided - same as demo)
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch"""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_samples += targets.size(0)
        correct_predictions += (predicted == targets).sum().item()
    
    return running_loss / len(train_loader), 100 * correct_predictions / total_samples

def validate_epoch(model, test_loader, criterion, device):
    """Evaluate the model on validation data"""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_samples += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()
    
    return running_loss / len(test_loader), 100 * correct_predictions / total_samples

# TODO 1: Design your own MLP architecture
class CustomMLP(nn.Module):
    """Your custom MLP architecture"""
    
    def __init__(self, input_size=784, num_classes=10):
        super(CustomMLP, self).__init__()
        
        # TODO: Define your architecture
        # Experiment with:
        # - Number of hidden layers (2, 3, or 4)
        # - Hidden layer sizes (32, 64, 128, 256, 512)
        # - Dropout probability (0.1, 0.2, 0.3, 0.5)
        # - Different activation functions (ReLU, LeakyReLU, Tanh)
        
        # Example: A deeper network
        # self.fc1 = nn.Linear(input_size, ???)  # First hidden layer
        # self.fc2 = nn.Linear(???, ???)         # Second hidden layer  
        # self.fc3 = nn.Linear(???, ???)         # Third hidden layer
        # self.fc4 = nn.Linear(???, num_classes) # Output layer
        # self.dropout = nn.Dropout(???)
        
        # TODO: Initialize your layers here
        pass  # Remove this when you add your layers
        
        # Store architecture info for display
        self.architecture = "Custom-Architecture"  # TODO: Update this description
    
    def forward(self, x):
        # TODO: Implement your forward pass
        # Remember to:
        # 1. Flatten input: x = x.view(x.size(0), -1)
        # 2. Apply layers with activations and dropout
        # 3. Return final output (no activation on output layer)
        
        pass  # TODO: Implement forward pass
    
    def get_num_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# TODO 2: Create different model configurations to test
print("\nTODO 2: Create and test different architectures")

# Configuration 1: Wide and shallow
model_config1 = {
    'name': 'Wide_Shallow',
    'model': None,  # TODO: Create CustomMLP with wide layers (e.g., 784->512->256->10)
    'description': 'Wide layers, few hidden layers'
}

# Configuration 2: Narrow and deep  
model_config2 = {
    'name': 'Narrow_Deep',
    'model': None,  # TODO: Create CustomMLP with narrow layers (e.g., 784->128->64->32->16->10)
    'description': 'Narrow layers, more hidden layers'
}

# Configuration 3: Your creative choice
model_config3 = {
    'name': 'Creative_Choice',
    'model': None,  # TODO: Create your own interesting architecture
    'description': 'Your creative architecture design'
}

# List of configurations to test
model_configs = [model_config1, model_config2, model_config3]

# TODO 3: Define different optimizers to compare
print("\nTODO 3: Define optimizers to test")

def get_optimizers(model_parameters):
    """Return a dictionary of different optimizers to test"""
    
    optimizers = {
        # TODO: Define different optimizers
        # Examples:
        # 'SGD': optim.SGD(model_parameters, lr=0.01, momentum=0.9),
        # 'Adam': optim.Adam(model_parameters, lr=0.001),
        # 'RMSprop': optim.RMSprop(model_parameters, lr=0.01),
        # 'AdaGrad': optim.Adagrad(model_parameters, lr=0.01),
        
        # TODO: Add at least 3 different optimizers here
    }
    
    return optimizers

# TODO 4: Experiment and compare results
print("\nStarting experiments...")
print("=" * 50)

# Storage for all results
all_results = {}
num_epochs = 8  # Shorter training for comparison

# TODO: Run experiments
"""
EXPERIMENT PSEUDOCODE:

for each model configuration:
    create model
    print model info (architecture, parameters)
    
    for each optimizer:
        reset model parameters (create new model instance)
        create optimizer for this model
        
        train for num_epochs:
            train_loss, train_acc = train_epoch(...)
            val_loss, val_acc = validate_epoch(...)
            store results
        
        print final results
        
store all results for plotting
"""

# TODO 4a: Implement the experiment loop
for config in model_configs:
    if config['model'] is None:
        print(f"Skipping {config['name']} - model not implemented")
        continue
        
    print(f"\nTesting {config['name']}: {config['description']}")
    print(f"Architecture: {config['model'].architecture}")
    print(f"Parameters: {config['model'].get_num_parameters():,}")
    
    model_results = {}
    optimizers = get_optimizers(config['model'].parameters())
    
    for opt_name, optimizer in optimizers.items():
        print(f"\n  Training with {opt_name}...")
        
        # TODO: Create fresh model instance (to reset parameters)
        model = None  # TODO: Create new instance of config['model']
        model = model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        
        # TODO: Create optimizer for this model
        optimizer = None  # TODO: Recreate optimizer for new model
        
        # Storage for this experiment
        train_losses = []
        val_accuracies = []
        
        # TODO: Training loop
        start_time = time.time()
        for epoch in range(num_epochs):
            # TODO: Train for one epoch
            train_loss, train_acc = None, None  # TODO: Call train_epoch
            val_loss, val_acc = None, None      # TODO: Call validate_epoch
            
            train_losses.append(train_loss)
            val_accuracies.append(val_acc)
            
            if epoch % 2 == 0 or epoch == num_epochs - 1:
                print(f"    Epoch {epoch+1}: Val Acc = {val_acc:.2f}%")
        
        training_time = time.time() - start_time
        final_acc = val_accuracies[-1]
        
        # Store results
        model_results[opt_name] = {
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'final_accuracy': final_acc,
            'training_time': training_time
        }
        
        print(f"    Final accuracy: {final_acc:.2f}% (trained in {training_time:.1f}s)")
    
    all_results[config['name']] = {
        'config': config,
        'results': model_results
    }

# 5. VISUALIZATION AND ANALYSIS (provided - plots your results)
print("\n" + "="*50)
print("EXPERIMENT RESULTS ANALYSIS")
print("="*50)

# Create comprehensive comparison plots
if all_results:
    n_models = len(all_results)
    n_optimizers = len(next(iter(all_results.values()))['results'])
    
    fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 10))
    if n_models == 1:
        axes = axes.reshape(2, 1)
    
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
    
    for i, (model_name, model_data) in enumerate(all_results.items()):
        config = model_data['config']
        results = model_data['results']
        
        # Plot 1: Validation accuracy over time
        ax1 = axes[0, i]
        for j, (opt_name, opt_results) in enumerate(results.items()):
            ax1.plot(range(1, num_epochs + 1), opt_results['val_accuracies'], 
                    color=colors[j % len(colors)], marker='o', linewidth=2, 
                    label=f"{opt_name} (Final: {opt_results['final_accuracy']:.1f}%)")
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Validation Accuracy (%)')
        ax1.set_title(f'{model_name}\n{config["description"]}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Final accuracy comparison
        ax2 = axes[1, i]
        opt_names = list(results.keys())
        final_accs = [results[opt]['final_accuracy'] for opt in opt_names]
        training_times = [results[opt]['training_time'] for opt in opt_names]
        
        bars = ax2.bar(range(len(opt_names)), final_accs, 
                      color=[colors[j % len(colors)] for j in range(len(opt_names))])
        ax2.set_xlabel('Optimizer')
        ax2.set_ylabel('Final Validation Accuracy (%)')
        ax2.set_title(f'{model_name} - Final Results')
        ax2.set_xticks(range(len(opt_names)))
        ax2.set_xticklabels(opt_names, rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add accuracy values on top of bars
        for j, (bar, acc, time) in enumerate(zip(bars, final_accs, training_times)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{acc:.1f}%\n({time:.1f}s)', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # Summary table
    print("\nSUMMARY TABLE")
    print("-" * 80)
    print(f"{'Model':<15} {'Optimizer':<10} {'Final Acc':<12} {'Time (s)':<10} {'Parameters':<12}")
    print("-" * 80)
    
    for model_name, model_data in all_results.items():
        config = model_data['config']
        results = model_data['results']
        param_count = config['model'].get_num_parameters() if config['model'] else 0
        
        for opt_name, opt_results in results.items():
            print(f"{model_name:<15} {opt_name:<10} {opt_results['final_accuracy']:<12.2f} "
                  f"{opt_results['training_time']:<10.1f} {param_count:<12,}")
    
    print("-" * 80)
    
    # Analysis questions
    print("\nANALYSIS QUESTIONS TO CONSIDER:")
    print("1. Which optimizer performed best overall?")
    print("2. Did deeper networks always perform better than wider ones?")
    print("3. Was there a trade-off between training time and accuracy?")
    print("4. Which architecture was most parameter-efficient?")
    print("5. Did any optimizer struggle with certain architectures?")

else:
    print("No experiments completed. Please implement the TODOs above.")