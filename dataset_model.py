import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

# Example of a simple neural network in PyTorch
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(10, 128)  # 10 input features (for age, gender, etc.)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)  # Output: binary classification (0 or 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x
        
df = pd.read_csv("cardio_train.csv", sep=";")  # Reads CSV into a DataFrame
X = df[['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco']].values
y = df['cardio'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # Make it a column vector
X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

model = SimpleNN()

# Define loss function and optimizer
loss_function = nn.BCELoss()  # Binary Cross Entropy loss
optimizer = optim.Adam(model.parameters(), lr=0.03)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)  # Send model to GPU if available

# Training loop
epochs = 10000
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    optimizer.zero_grad()  # Zero out gradients
    
    # Forward pass
    outputs = model(X_train.to(device))
    loss = loss_function(outputs, y_train.to(device))
    # Backward pass
    loss.backward()
    
    # Optimize
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")


# Set the model to evaluation mode
model.eval()

# Forward pass on the test data
with torch.no_grad():
    predictions = model(X_test.to(device))
    predicted_classes = (predictions > 0.5).float()  # Convert to binary predictions (0 or 1)
    
    # Calculate accuracy
    correct = (predicted_classes == y_test.to(device)).sum().item()
    accuracy = correct / y_test.size(0)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
