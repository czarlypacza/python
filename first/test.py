import torch
import torch.nn as nn
import torch.optim as optim

# Check for GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Define a simple model
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.layer1 = nn.Linear(10, 50)  # input size of 10, hidden layer of 50
        self.layer2 = nn.Linear(50, 2)   # output size of 2 (for example, binary classification)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# Instantiate the model, define a loss function and optimizer
model = SimpleMLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create some dummy data
inputs = torch.randn(16, 10).to(device)  # Batch size of 16, input size of 10
targets = torch.randint(0, 2, (16,)).to(device)  # Random integer targets for testing

# Forward pass through the network
outputs = model(inputs)
loss = criterion(outputs, targets)
print(f"Initial loss: {loss.item()}")

# Backward pass and optimization
loss.backward()
optimizer.step()

print("Simple model training step completed successfully.")
