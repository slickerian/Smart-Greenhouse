import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import joblib

# Define the neural network architecture
class CropNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CropNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=0.5)  # Adjust dropout rate
        self.activation = nn.ReLU()  # Changed activation function
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.activation(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return self.softmax(x)

# Load and preprocess the data
data = pd.read_csv('Greenhouse.csv')
features = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'soil_moisture']]
target = data['label']

# Encode target labels
label_encoder = LabelEncoder()
encoded_target = label_encoder.fit_transform(target)

# Save the label encoder
joblib.dump(label_encoder, 'label_encoder.pkl')

# Standardize features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, encoded_target, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader for training and testing
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Changed batch size
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # Changed batch size

# Initialize the model, loss function, optimizer, and learning rate scheduler
input_dim = X_train.shape[1]
num_classes = len(np.unique(encoded_target))

model = CropNN(input_dim, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adjusted learning rate
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

# Early stopping parameters
patience = 5
best_loss = float('inf')
early_stop_counter = 0

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_features, batch_labels in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_features.size(0)

    epoch_loss = running_loss / len(train_dataset)
    scheduler.step()

    # Print loss for current epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    # Early stopping check
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        early_stop_counter = 0
        # Save the model
        torch.save(model.state_dict(), 'models/crop_nn_model.pth')
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("Early stopping triggered")
            break

# Load the best model for evaluation
model.load_state_dict(torch.load('models/crop_nn_model.pth', weights_only=True))
model.eval()

# Evaluate the model
with torch.no_grad():
    correct = 0
    total = 0
    for batch_features, batch_labels in test_loader:
        outputs = model(batch_features)
        _, predicted = torch.max(outputs, 1)
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
