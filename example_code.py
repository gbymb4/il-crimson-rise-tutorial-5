# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 14:13:34 2025

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
from collections import defaultdict
import time

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("PyTorch Neural Networks - MNIST MLP Classification")
print("=" * 55)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. DATA LOADING AND PREPROCESSING
print("\n1. Loading MNIST dataset...")

# Define transformations for the training and test data
transform = transforms.Compose([
    transforms.ToTensor(),                          # Convert PIL Image to tensor
    transforms.Normalize((0.1307,), (0.3081,))     # Normalize with MNIST mean and std
])

# Download and load the training and test datasets
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create data loaders for batch processing
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Batch size: {batch_size}")
print(f"Training batches: {len(train_loader)}")
print(f"Test batches: {len(test_loader)}")

# Visualize sample data
def show_sample_images(loader, num_samples=8):
    """Display a few sample images from the dataset"""
    data_iter = iter(loader)
    images, labels = next(data_iter)
    
    plt.figure(figsize=(12, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        # Convert tensor to numpy and remove channel dimension
        img = images[i].numpy().squeeze()
        plt.imshow(img, cmap='gray')
        plt.title(f'Label: {labels[i].item()}')
        plt.axis('off')
    
    plt.suptitle('Sample MNIST Images')
    plt.tight_layout()
    plt.show()

show_sample_images(train_loader)

# 2. MODEL DEFINITION
print("\n2. Defining the MLP model...")

class MNISTmlp(nn.Module):
    """Multi-Layer Perceptron for MNIST classification"""
    
    def __init__(self, input_size=784, hidden1_size=128, hidden2_size=64, num_classes=10, dropout_prob=0.2):
        super(MNISTmlp, self).__init__()
        
        # Define the layers
        self.fc1 = nn.Linear(input_size, hidden1_size)       # Input to first hidden layer
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)     # First to second hidden layer
        self.fc3 = nn.Linear(hidden2_size, num_classes)      # Second hidden to output layer
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_prob)
        
        # Store architecture info
        self.input_size = input_size
        self.architecture = f"{input_size}-{hidden1_size}-{hidden2_size}-{num_classes}"
    
    def forward(self, x):
        # Flatten the input (batch_size, 1, 28, 28) -> (batch_size, 784)
        x = x.view(x.size(0), -1)
        
        # First hidden layer with ReLU activation
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Second hidden layer with ReLU activation  
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Output layer (no activation - CrossEntropyLoss handles this)
        x = self.fc3(x)
        
        return x
    
    def get_num_parameters(self):
        """Count total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Create model instance
model = MNISTmlp().to(device)
print(f"Model architecture: {model.architecture}")
print(f"Total trainable parameters: {model.get_num_parameters():,}")

# Print model structure
print("\nModel structure:")
print(model)

# 3. TRAINING SETUP
print("\n3. Setting up training components...")

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Loss function: {criterion}")
print(f"Optimizer: {optimizer}")

# Training parameters
num_epochs = 10
print(f"Training epochs: {num_epochs}")

# 4. TRAINING AND VALIDATION FUNCTIONS
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch"""
    model.train()  # Set model to training mode
    
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Move data to device
        data, targets = data.to(device), targets.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_samples += targets.size(0)
        correct_predictions += (predicted == targets).sum().item()
    
    # Calculate averages
    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct_predictions / total_samples
    
    return avg_loss, accuracy

def validate_epoch(model, test_loader, criterion, device):
    """Evaluate the model on validation data"""
    model.eval()  # Set model to evaluation mode
    
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():  # Disable gradient computation for efficiency
        for data, targets in test_loader:
            # Move data to device
            data, targets = data.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_samples += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()
    
    # Calculate averages
    avg_loss = running_loss / len(test_loader)
    accuracy = 100 * correct_predictions / total_samples
    
    return avg_loss, accuracy

# 5. TRAINING LOOP
print("\n4. Starting training...")
print("=" * 55)

# Storage for training history
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

# Training loop
start_time = time.time()

for epoch in range(num_epochs):
    # Train for one epoch
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    
    # Validate
    val_loss, val_acc = validate_epoch(model, test_loader, criterion, device)
    
    # Store history
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    
    # Print progress
    print(f"Epoch {epoch+1:2d}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:5.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:5.2f}%")

training_time = time.time() - start_time
print(f"\nTraining completed in {training_time:.2f} seconds")
print(f"Final validation accuracy: {val_accuracies[-1]:.2f}%")

# 6. VISUALIZATION AND ANALYSIS
print("\n5. Analyzing results...")

# Create comprehensive plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Training and validation loss
ax1.plot(range(1, num_epochs + 1), train_losses, 'b-', label='Training Loss', linewidth=2)
ax1.plot(range(1, num_epochs + 1), val_losses, 'r-', label='Validation Loss', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Training and validation accuracy
ax2.plot(range(1, num_epochs + 1), train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
ax2.plot(range(1, num_epochs + 1), val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training and Validation Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Sample predictions
def show_predictions(model, test_loader, num_samples=8):
    """Show model predictions on test samples"""
    model.eval()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # Move back to CPU for plotting
    images = images.cpu()
    labels = labels.cpu()
    predicted = predicted.cpu()
    
    for i in range(num_samples):
        plt.subplot(2, 4, i + 1)
        img = images[i].numpy().squeeze()
        plt.imshow(img, cmap='gray')
        
        actual = labels[i].item()
        pred = predicted[i].item()
        color = 'green' if actual == pred else 'red'
        plt.title(f'True: {actual}, Pred: {pred}', color=color)
        plt.axis('off')

# Plot 3: Model predictions
ax3.axis('off')
ax3.text(0.5, 0.5, 'Sample Predictions\n(See separate plot)', 
         ha='center', va='center', fontsize=14, transform=ax3.transAxes)

# Plot 4: Training summary
ax4.axis('off')
summary_text = f"""Training Summary

Architecture: {model.architecture}
Parameters: {model.get_num_parameters():,}
Training Time: {training_time:.2f}s
Epochs: {num_epochs}

Final Results:
Train Accuracy: {train_accuracies[-1]:.2f}%
Val Accuracy: {val_accuracies[-1]:.2f}%
Train Loss: {train_losses[-1]:.4f}
Val Loss: {val_losses[-1]:.4f}

Optimizer: Adam (lr=0.001)
Batch Size: {batch_size}"""

ax4.text(0.1, 0.9, summary_text, fontsize=10, verticalalignment='top', 
         transform=ax4.transAxes, fontfamily='monospace')

plt.tight_layout()
plt.show()

# Show sample predictions in a separate plot
plt.figure(figsize=(12, 6))
show_predictions(model, test_loader)
plt.suptitle('Model Predictions on Test Data (Green=Correct, Red=Incorrect)')
plt.tight_layout()
plt.show()

print("\nKey Observations:")
print("1. The model learns quickly - MNIST is a relatively simple dataset")
print("2. Training and validation curves should be close (no major overfitting)")
print("3. Accuracy should reach >95% easily with this architecture")
print("4. PyTorch handles all the complex gradient computations automatically")
print("5. The program flow is clean and modular - easy to modify and experiment")