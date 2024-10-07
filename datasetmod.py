import pandas as pd

# Load the dataset
df = pd.read_csv('Crop_recommendation.csv')  # Replace with your actual file name

# Define the sensor value range
min_sensor_value = 0
max_sensor_value = 1073

# Calculate the min and max rainfall values in the dataset
min_rainfall = df['rainfall'].min()
max_rainfall = df['rainfall'].max()

# Function to map rainfall to soil moisture sensor value
def map_rainfall_to_soil_moisture(rainfall):
    return ((rainfall - min_rainfall) / (max_rainfall - min_rainfall)) * (max_sensor_value - min_sensor_value) + min_sensor_value

# Apply the mapping function to the rainfall column and replace it
df['rainfall'] = df['rainfall'].apply(map_rainfall_to_soil_moisture)

# Rename the 'rainfall' column to 'soil_moisture'
df.rename(columns={'rainfall': 'soil_moisture'}, inplace=True)

# Save the updated dataset to a new CSV file
df.to_csv('Greenhouse.csv', index=False)

# Display the first few rows of the updated dataset
print(df.head())
