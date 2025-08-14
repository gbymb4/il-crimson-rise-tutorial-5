# -*- coding: utf-8 -*-
"""
MNIST MLP Classification - Homework Assignment
Complete the TODOs to build a working MLP classifier for MNIST digits

Learning objectives:
- Load and preprocess MNIST dataset
- Design a custom MLP architecture
- Implement training and evaluation loops
- Analyze model performance

@author: [Your Name]
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

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("MNIST MLP Classification Homework")
print("=" * 40)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

#%% TODO 1: Data Loading and Preprocessing
print("\nTODO 1: Data Loading and Preprocessing")

# TODO 1a: Define data transformation
# Hint: MNIST images just need to be converted to tensors
# Use transforms.ToTensor() - this converts PIL images to tensors and scales to [0,1]
transform = None  # TODO: Set transform to ToTensor()

# TODO 1b: Load the MNIST dataset
# Hint: Use torchvision.datasets.MNIST
# Remember to set train=True/False and transform=transform
train_dataset = None  # TODO: Load training data
test_dataset = None   # TODO: Load test data

# TODO 1c: Create data loaders
batch_size = 64  # You can experiment with different batch sizes
train_loader = None  # TODO: Create DataLoader for training data
test_loader = None   # TODO: Create DataLoader for test data

# TODO 1d: Print dataset information
print(f"Training samples: ???")  # TODO: Print number of training samples
print(f"Test samples: ???")     # TODO: Print number of test samples
print(f"Number of classes: ???")  # TODO: Print number of classes (should be 10)
print(f"Input image shape: ???")  # TODO: Print shape of one image

# TODO 1e: Visualize some sample images (BONUS)
# Hint: Use matplotlib to display a few images with their labels
# This helps verify your data loading is working correctly

#%% TODO 2: Model Architecture
print("\nTODO 2: Model Architecture")

class MLP(nn.Module):
    """Multi-Layer Perceptron for MNIST classification"""
    
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(MLP, self).__init__()
        
        # TODO 2a: Define the layers
        # Hint: MNIST images are 28x28 = 784 pixels when flattened
        # Suggested architecture: Input -> Hidden1 -> Hidden2 -> Output
        # Use nn.Linear for fully connected layers
        self.fc1 = None    # TODO: First hidden layer (input_size -> hidden_size)
        self.fc2 = None    # TODO: Second hidden layer (hidden_size -> hidden_size)
        self.fc3 = None    # TODO: Output layer (hidden_size -> num_classes)
        
        # TODO 2b: Add dropout for regularization (optional but recommended)
        self.dropout = None  # TODO: Add dropout layer (try dropout probability = 0.2)
        
    def forward(self, x):
        # TODO 2c: Implement the forward pass
        # Remember to:
        # 1. Flatten the input: x.view(x.size(0), -1)
        # 2. Apply layers with ReLU activation
        # 3. Apply dropout between layers
        # 4. No activation on final output layer
        
        # TODO: Flatten input (batch_size, 28, 28) -> (batch_size, 784)
        x = None
        
        # TODO: First layer + activation
        x = None
        
        # TODO: Apply dropout (if implemented)
        
        # TODO: Second layer + activation
        x = None
        
        # TODO: Apply dropout (if implemented)
        
        # TODO: Output layer (no activation - CrossEntropyLoss handles this)
        x = None
        
        return x

# TODO 2d: Create model instance and move to device
model = None  # TODO: Create MLP instance
# TODO: Move model to device

# TODO 2e: Print model information
print(f"Model architecture:\n{model}")
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params:,}")

#%% TODO 3: Training Setup
print("\nTODO 3: Training Setup")

# TODO 3a: Define loss function
# Hint: Use CrossEntropyLoss for multi-class classification
criterion = None  # TODO: Define loss function

# TODO 3b: Define optimizer
# Hint: Try Adam optimizer with learning rate 0.001
optimizer = None  # TODO: Define optimizer

# Training parameters
num_epochs = 10

# TODO 3c: Create lists to store training history
train_losses = []     # TODO: List to store training losses
train_accuracies = [] # TODO: List to store training accuracies
test_accuracies = []  # TODO: List to store test accuracies

#%% TODO 4: Training Loop
print("\nTODO 4: Training Loop")

for epoch in range(num_epochs):
    # TODO 4a: Training phase
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        # TODO: Move data and targets to device
        data, targets = None, None
        
        # TODO 4b: Forward pass
        # Clear gradients, compute outputs, compute loss
        optimizer.zero_grad()
        outputs = None    # TODO: Forward pass through model
        loss = None       # TODO: Compute loss
        
        # TODO 4c: Backward pass and optimization
        # TODO: Compute gradients (backward pass)
        # TODO: Update parameters (optimizer step)
        
        # TODO 4d: Track training statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += targets.size(0)
        correct_train += (predicted == targets).sum().item()
    
    # Calculate training accuracy and loss for this epoch
    train_acc = 100 * correct_train / total_train
    avg_train_loss = running_loss / len(train_loader)
    
    # TODO 4e: Evaluation phase
    model.eval()  # Set model to evaluation mode
    correct_test = 0
    total_test = 0
    
    with torch.no_grad():  # Disable gradient computation for evaluation
        for data, targets in test_loader:
            # TODO: Move data and targets to device
            data, targets = None, None
            
            # TODO: Forward pass (no gradients needed)
            outputs = None    # TODO: Forward pass
            _, predicted = torch.max(outputs, 1)
            
            total_test += targets.size(0)
            correct_test += (predicted == targets).sum().item()
    
    test_acc = 100 * correct_test / total_test
    
    # TODO 4f: Store training history
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    
    # Print progress
    print(f'Epoch [{epoch+1}/{num_epochs}] - '
          f'Train Loss: {avg_train_loss:.4f}, '
          f'Train Acc: {train_acc:.2f}%, '
          f'Test Acc: {test_acc:.2f}%')

#%% TODO 5: Results Analysis and Visualization
print("\nTODO 5: Results Analysis")

# TODO 5a: Print final results
final_train_acc = train_accuracies[-1]
final_test_acc = test_accuracies[-1]
print(f"\nFinal Training Accuracy: {final_train_acc:.2f}%")
print(f"Final Test Accuracy: {final_test_acc:.2f}%")

# TODO 5b: Plot training curves
# Hint: Create subplots for loss and accuracy over epochs
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# TODO: Plot training loss
ax1.plot(None, None)  # TODO: Plot epochs vs train_losses
ax1.set_title('Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.grid(True)

# TODO: Plot accuracies
ax2.plot(None, None, label='Train Accuracy')    # TODO: Plot training accuracy
ax2.plot(None, None, label='Test Accuracy')     # TODO: Plot test accuracy
ax2.set_title('Model Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

#%% TODO 6: Model Evaluation (BONUS)
print("\nTODO 6: Model Evaluation (BONUS)")

# TODO 6a: Test on individual samples
# Pick a few test samples and show predictions vs actual labels

# TODO 6b: Confusion matrix
# Hint: Use sklearn.metrics.confusion_matrix if available
# This shows which digits are most commonly confused

# TODO 6c: Calculate per-class accuracy
# Show accuracy for each digit (0-9) separately

#%% TODO 7: Experiments (BONUS)
print("\nTODO 7: Experiments (BONUS)")

# Try these experiments and compare results:
# 1. Different architectures (more/fewer layers, different sizes)
# 2. Different optimizers (SGD vs Adam vs RMSprop)
# 3. Different learning rates
# 4. Different batch sizes
# 5. With/without dropout

print("\nHomework completed! 🎉")
print("\nReflection Questions:")
print("1. How did your model perform? Was the accuracy acceptable?")
print("2. Did you observe overfitting? How could you tell?")
print("3. What would you try to improve the model performance?")
print("4. How long did training take? How could you speed it up?")