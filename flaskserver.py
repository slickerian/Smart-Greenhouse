from flask import Flask, request, jsonify
import pandas as pd
import time
import torch
import torch.nn as nn
import joblib
import numpy as np
import serial  # For serial communication with Arduino

app = Flask(__name__)

# Serial port configuration
SERIAL_PORT = 'COM3'
BAUD_RATE = 9600

def get_serial_connection():
    """ Create and return a serial connection, with debug messages """
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        if ser.is_open:
            print(f"Successfully connected to Arduino on {SERIAL_PORT}")
        return ser
    except serial.SerialException as e:
        print(f"Failed to connect to Arduino on {SERIAL_PORT}: {str(e)}")
        return None


# Load the dataset globally
data = pd.read_csv('Greenhouse.csv')

def get_summary_for_crop(crop_type):
    # Filter the dataset for the given crop type
    filtered_data = data[data['label'] == crop_type]
    
    # Check if there are entries for the given crop type
    if filtered_data.empty:
        return f"No data available for crop type '{crop_type}'"
    
    # Extract all relevant feature values
    values = filtered_data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'soil_moisture']]
    
    # Calculate average for each feature
    averages = {
        'N': values['N'].mean(),
        'P': values['P'].mean(),
        'K': values['K'].mean(),
        'temperature': values['temperature'].mean(),
        'humidity': values['humidity'].mean(),
        'ph': values['ph'].mean(),
        'soil_moisture': values['soil_moisture'].mean()
    }
    
    return averages

# Endpoint to handle parameter auto-detection
@app.route('/auto_detect', methods=['POST'])
def auto_detect():
    try:
        print("Attempting to connect to Arduino...")
        ser = get_serial_connection()
        print("Connection established.")
        
        ser.write(b'SEND\n')  # Send a command to Arduino
        print("Command sent.")
        
        time.sleep(2)  # Wait for data to be available
        if ser.in_waiting > 0:
            sensor_data = ser.readline().decode('utf-8').strip()
            print(f"Data received: {sensor_data}")
            return jsonify({'sensor_data': sensor_data})
        else:
            print("No data received from Arduino.")
            return jsonify({'error': 'No data received from Arduino'}), 500
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return jsonify({'error': str(e)}), 500


# Endpoint to receive sensor data from Arduino
@app.route('/update_sensor_data', methods=['GET'])
def update_sensor_data():
    temperature = request.args.get('TEMP')
    humidity = request.args.get('HUM')
    soil1 = request.args.get('SOIL1')
    
    # Process the received data as needed (e.g., save to a database or perform analysis)
    
    return jsonify({'status': 'success', 'received': {'temperature': temperature, 'humidity': humidity, 'soil1': soil1}})


@app.route('/send_parameters', methods=['POST'])
def send_parameters():
    """ Receive parameters from the request and send them to the Arduino """
    parameters = request.json
    if not parameters:
        return jsonify({'error': 'No parameters provided'}), 400
    
    try:
        ser = get_serial_connection()
        if ser is None:
            return jsonify({'error': 'Failed to connect to Arduino'}), 500
        
        # Extract only the relevant parameters
        relevant_parameters = {
            'temperature': parameters.get('temperature', ''),
            'humidity': parameters.get('humidity', ''),
            'soil_moisture': parameters.get('soil_moisture', '')
        }
        
        # Construct command string
        command_parts = [f"{key}:{value}" for key, value in relevant_parameters.items() if value]
        command = f"PARAMS:{','.join(command_parts)}\n"
        print(f"Sending command to Arduino: {command}")
        ser.write(command.encode())  # Send command to Arduino

        # Read response from Arduino
        time.sleep(1)  # Wait for response
        if ser.in_waiting > 0:
            arduino_response = ser.readline().decode('utf-8').strip()
            print(f"Arduino response: {arduino_response}")
            return jsonify({'status': 'success', 'message': arduino_response})
        else:
            return jsonify({'status': 'success', 'message': 'Parameters sent, but no response from Arduino'}), 500

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500




    

# Endpoint to handle crop prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
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

    # Prepare input features
    input_features = np.array([[data.get('N', 0), data.get('P', 0), data.get('K', 0),
                                data.get('temperature', 0), data.get('humidity', 0),
                                data.get('ph', 0), data.get('soil_moisture', 0)]])
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

    result = {
        'crop': predicted_crop_name,
        'parameters': {
            'N': data.get('N', 'N/A'),
            'P': data.get('P', 'N/A'),
            'K': data.get('K', 'N/A'),
            'temperature': data.get('temperature', 'N/A'),
            'humidity': data.get('humidity', 'N/A'),
            'ph': data.get('ph', 'N/A'),
            'soil_moisture': data.get('soil_moisture', 'N/A'),
        }
    }

    return jsonify(result)

# Endpoint to handle parameter fetching
@app.route('/get_parameters', methods=['GET'])
def get_parameters():
    crop_type = request.args.get('crop_type')
    parameters = get_summary_for_crop(crop_type)
    return jsonify(parameters)

# Endpoint to handle sending commands to Arduino
@app.route('/send_command', methods=['POST'])
def send_command():
    command = request.json.get('command')
    if not command:
        return jsonify({'error': 'No command provided'}), 400
    
    try:
        with get_serial_connection() as ser:
            ser.write(command.encode() + b'\n')  # Send command to Arduino
            return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug = True)
