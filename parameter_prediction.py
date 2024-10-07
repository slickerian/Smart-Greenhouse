import pandas as pd

# Load the dataset
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

# Example usage
crop_type = 'rice'  # Replace with the crop type you want to check
crop_summary = get_summary_for_crop(crop_type)

if isinstance(crop_summary, str):
    print(crop_summary)
else:
    print(f"Average values for '{crop_type}':")
    for feature, avg in crop_summary.items():
        print(f"{feature} - Average: {avg:.2f}")
