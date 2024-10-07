import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Define the neural network architecture
class CropNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CropNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=0.5)
        self.activation = nn.ReLU()
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

# Load the model
input_dim = 7  # Number of features
num_classes = 22  # Number of classes
model = CropNN(input_dim, num_classes)
model.load_state_dict(torch.load('models/crop_nn_model.pth'))
model.eval()

# Load the scaler and label encoder
scaler = joblib.load('models/scaler.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

# Define the class-to-name mapping (from label encoder or manually)
class_names = label_encoder.classes_

# Sample input for prediction (replace with your feature values)
input_features = np.array([[71, 54, 16, 23.5, 65, 6.0, 300]])  # Replace with your feature values
input_df = pd.DataFrame(input_features, columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'soil_moisture'])

# Preprocess the input data
input_scaled = scaler.transform(input_df)
input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

# Make prediction
with torch.no_grad():
    outputs = model(input_tensor)
    _, predicted_class = torch.max(outputs, 1)
    predicted_class_index = predicted_class.item()

# Map class index to crop name
predicted_crop_name = class_names[predicted_class_index]
print(f'Predicted Crop Name: {predicted_crop_name}')
