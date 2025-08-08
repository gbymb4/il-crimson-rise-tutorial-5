# Machine Learning Session 5: PyTorch Neural Networks
## Building MLPs for MNIST Classification

### Session Overview
**Duration**: 1 hour  
**Prerequisites**: Completed Session 4 (PyTorch Direct Parameter Optimization)  
**Goal**: Build neural networks using PyTorch's high-level APIs on real data  
**Focus**: Program flow, network architecture, and optimizer comparison

### Session Timeline
| Time      | Activity                                    |
| --------- | ------------------------------------------- |
| 0:00 - 0:05 | 1. Touching Base & Session Overview    |
| 0:05 - 0:15 | 2. From Manual to Automatic - PyTorch's High-Level APIs |
| 0:15 - 0:25 | 3. MNIST MLP Example - Complete Program Flow |
| 0:25 - 0:35 | 4. Running the Demo & Understanding Each Component |
| 0:35 - 0:55 | 5. Solo Exercise: Custom MLP & Optimizer Comparison |
| 0:55 - 1:00 | 6. Wrap-up & Next Steps |

---

## 1. Touching Base & Session Overview (5 minutes)

### Quick Check-in
- Review Session 4's manual gradient descent with PyTorch
- Ensure additional dependencies are installed (`pip install torchvision`)
- Preview today's shift from manual to automatic neural network training

### Today's Learning Objectives
By the end of this session, you will be able to:
- Build neural networks using `torch.nn.Module` and `torch.nn` layers
- Load and preprocess real datasets using `torchvision`
- Implement complete training and validation loops
- Compare different optimizers (SGD, Adam, RMSprop)
- Understand the standard PyTorch program flow for deep learning
- Interpret training metrics and visualizations

---

## 2. From Manual to Automatic - PyTorch's High-Level APIs (10 minutes)

### Key Progression from Session 4

**Session 4 Approach (Manual)**
- Defined parameters explicitly: `torch.tensor([0.5], requires_grad=True)`
- Implemented forward pass manually: `predictions = slope * x + intercept`
- Wrote gradient descent by hand: `param -= lr * param.grad`
- Managed gradient zeroing manually: `param.grad.zero_()`

**Session 5 Approach (Neural Networks)**
- Define layers using `torch.nn`: `nn.Linear(784, 128)`
- Forward pass through network: `output = self.network(x)`
- Use built-in optimizers: `torch.optim.Adam(model.parameters())`
- Automatic gradient management: `optimizer.zero_grad()`, `optimizer.step()`

### Core PyTorch Components for Neural Networks

**1. `torch.nn.Module` - The Foundation**
```python
class MyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Define layers here
    
    def forward(self, x):
        # Define forward pass here
        return output
```

**2. `torch.nn` Layers**
- `nn.Linear(in_features, out_features)` - Fully connected layer
- `nn.ReLU()` - Activation function
- `nn.Dropout(p=0.2)` - Regularization
- `nn.CrossEntropyLoss()` - Loss function for classification

**3. `torch.optim` Optimizers**
- `optim.SGD(model.parameters(), lr=0.01)` - Stochastic Gradient Descent
- `optim.Adam(model.parameters(), lr=0.001)` - Adaptive learning rates
- `optim.RMSprop(model.parameters(), lr=0.01)` - Root Mean Square propagation

**4. `torchvision` for Data**
- `torchvision.datasets` - Common datasets (MNIST, CIFAR-10, etc.)
- `torchvision.transforms` - Data preprocessing
- `torch.utils.data.DataLoader` - Batch processing

### The Standard PyTorch Program Flow
1. **Define the model** (architecture)
2. **Load the data** (datasets and dataloaders)
3. **Set up training** (loss function, optimizer)
4. **Training loop** (forward pass, loss calculation, backprop, update)
5. **Validation loop** (evaluate performance)
6. **Visualization** (plot metrics and results)

---

## 3. MNIST MLP Example - Complete Program Flow (10 minutes)

### Understanding MNIST
- **Dataset**: 70,000 handwritten digit images (0-9)
- **Task**: Classify which digit (0-9) each image shows
- **Input**: 28x28 pixel grayscale images (784 features when flattened)
- **Output**: 10 classes (one for each digit)
- **Why MNIST**: Simple, fast to train, well-understood benchmark

### Architecture Overview
We'll build a Multi-Layer Perceptron (MLP) with:
- **Input layer**: 784 neurons (28×28 flattened pixels)
- **Hidden layer 1**: 128 neurons + ReLU activation
- **Hidden layer 2**: 64 neurons + ReLU activation  
- **Output layer**: 10 neurons (one per digit class)
- **Dropout**: For regularization between layers

---

## 4. Running the Demo & Understanding Each Component (10 minutes)

*This complete example will be demonstrated live, explaining each section*

### Complete MNIST MLP Example

```python
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
```

### Key Points to Explain During Demo
- **Class inheritance**: `nn.Module` provides the foundation for all neural networks
- **Layer definitions**: How `nn.Linear` creates fully connected layers
- **Forward pass**: The `forward()` method defines how data flows through the network
- **Training vs evaluation modes**: `model.train()` vs `model.eval()`
- **Gradient management**: `optimizer.zero_grad()` and `optimizer.step()`
- **Device handling**: Moving tensors to GPU/CPU as needed
- **Data loading**: How PyTorch datasets and dataloaders work

---

## 5. Solo Exercise: Custom MLP & Optimizer Comparison (20 minutes)

### Exercise Instructions
Your task is to design your own MLP architecture and compare different optimizers. You'll experiment with network depth, width, and optimization strategies to see how they affect training performance.

### Exercise Script with TODOs

```python
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

# TODO 5: Bonus challenges (optional)
print("\n" + "="*50)
print("BONUS CHALLENGES (Optional)")
print("="*50)

bonus_challenges = """
If you've completed the main exercise, try these additional experiments:

1. LEARNING RATE SENSITIVITY
   - Test the same optimizer with different learning rates (0.1, 0.01, 0.001, 0.0001)
   - Plot how learning rate affects convergence speed and final accuracy

2. BATCH SIZE EFFECTS  
   - Try different batch sizes (32, 64, 128, 256, 512)
   - Observe how batch size affects training stability and speed

3. ACTIVATION FUNCTION COMPARISON
   - Replace ReLU with other activation functions (LeakyReLU, Tanh, Sigmoid)
   - Compare convergence behavior and final performance

4. REGULARIZATION TECHNIQUES
   - Add L2 weight decay to optimizers
   - Experiment with different dropout probabilities
   - Try batch normalization between layers

5. EARLY STOPPING
   - Implement early stopping based on validation loss
   - Compare training efficiency across different stopping criteria

6. LEARNING RATE SCHEDULING
   - Implement learning rate decay (step, exponential, cosine)
   - Compare with constant learning rates

Code template for bonus challenges:
```python
# Example: Learning rate sensitivity
learning_rates = [0.1, 0.01, 0.001, 0.0001]
for lr in learning_rates:
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Train and compare results...
```
"""

print(bonus_challenges)
```

### Expected Learning Outcomes
By completing this exercise, students will:
- Understand how to design custom neural network architectures
- Gain hands-on experience with different optimizers
- Learn to compare and analyze experimental results
- Develop intuition for architecture and optimizer selection
- Practice the complete PyTorch workflow from data to results

### Hints for Common Issues
- **Model not learning**: Check learning rate (too high/low), architecture (too simple/complex)
- **Memory errors**: Reduce batch size or model size
- **Slow training**: Use GPU if available, reduce model complexity
- **Poor convergence**: Try different optimizers or learning rates
- **Implementation errors**: Check tensor shapes, device placement, gradient flow

---

## 6. Wrap-up & Next Steps (5 minutes)

### Key Takeaways
- PyTorch provides high-level abstractions while maintaining flexibility
- Neural networks follow a standard pattern: define → load data → train → evaluate
- Different optimizers have different strengths and convergence behaviors
- Architecture design involves trade-offs between complexity, performance, and training time
- Visualization and systematic comparison are crucial for understanding model behavior

### What We've Accomplished
- Built complete neural networks using PyTorch's `nn.Module`
- Loaded and processed real-world data (MNIST)
- Implemented full training and validation pipelines
- Compared different architectures and optimizers systematically
- Learned to interpret training metrics and visualizations

### Progression Through Sessions
- **Session 3**: High-level ML with scikit-learn (black box approach)
- **Session 4**: Low-level parameter optimization (complete manual control)
- **Session 5**: Neural networks with PyTorch (balanced abstraction and control)
- **Next**: Advanced architectures and specialized techniques

### Next Session Preview
In Session 6, we'll explore specialized architectures:
- Convolutional Neural Networks (CNNs) for image classification
- Working with more complex datasets (CIFAR-10)
- Understanding feature learning and representation
- Transfer learning and pre-trained models

### Key Insights for Deep Learning Success
1. **Start simple**: Begin with basic architectures before adding complexity
2. **Compare systematically**: Always test multiple configurations
3. **Visualize everything**: Plots reveal insights that numbers alone cannot
4. **Understand your data**: Preprocessing and data loading are crucial
5. **Experiment iteratively**: Build intuition through hands-on experience

### Questions & Discussion
- How did the neural network approach compare to Session 4's manual optimization?
- Which optimizer surprised you the most in terms of performance?
- What architecture design principles did you discover?
- How might you apply these techniques to other domains beyond image classification?
- What aspects of the PyTorch workflow do you want to explore further?

### Homework/Practice (Optional)
1. Try the exercise with a different dataset (Fashion-MNIST, CIFAR-10)
2. Implement the bonus challenges (learning rate scheduling, regularization)
3. Experiment with very deep networks (5+ hidden layers)
4. Research and implement advanced optimizers (AdamW, RAdam)
5. Create visualizations of learned features and network weights