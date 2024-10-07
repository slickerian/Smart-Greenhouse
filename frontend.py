import streamlit as st
import requests

FLASK_SERVER_URL = 'http://localhost:5000'

st.set_page_config(page_title="Smart Greenhouse", page_icon=":sunrise:", layout="wide")

# Function to auto-detect parameters
def auto_detect_params():
    response = requests.post(f'{FLASK_SERVER_URL}/auto_detect')
    if response.status_code == 200:
        return response.json()
    else:
        st.error('Failed to get sensor data')
        return {}

# Function for crop prediction
def crop_prediction():
    st.title('Crop Prediction')
    st.write('Enter the following parameters:')

    # Initialize session state for auto-detected parameters
    if 'auto_detected' not in st.session_state:
        st.session_state['auto_detected'] = {
            'temperature': '',
            'humidity': '',
            'soil_moisture': ''
        }

    # Update text input fields with values from session state
    temperature = st.text_input('Temperature', value=st.session_state['auto_detected'].get('temperature', ''))
    humidity = st.text_input('Humidity', value=st.session_state['auto_detected'].get('humidity', ''))
    soil_moisture = st.text_input('Soil Moisture', value=st.session_state['auto_detected'].get('soil_moisture', ''))

    N = st.text_input('Nitrogen (N)', value='')
    P = st.text_input('Phosphorus (P)', value='')
    K = st.text_input('Potassium (K)', value='')
    ph = st.text_input('pH', value='')

    # Auto Detect button
    if st.button('Auto Detect'):
        params = auto_detect_params()
        st.session_state['auto_detected'] = params
        #st.experimental_rerun()  # Rerun to update the state and inputs

    # Predict button
    if st.button('Predict'):
        payload = {
            'N': N,
            'P': P,
            'K': K,
            'temperature': temperature,
            'humidity': humidity,
            'ph': ph,
            'soil_moisture': soil_moisture
        }
        response = requests.post(f"{FLASK_SERVER_URL}/predict", json=payload)
        if response.status_code == 200:
            result = response.json()
            st.write('Predicted Crop:', result.get('crop'))
            st.write('Parameters:')
            for key, value in result['parameters'].items():
                st.write(f"{key}: {value}")
        else:
            st.error("Prediction failed.")

# Function for parameter prediction
def parameter_prediction():
    st.title('Parameter Prediction')
    st.write('Select a crop to view the optimal parameters:')

    crop_options = ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate', 'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple', 'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee']
    selected_crop = st.selectbox('Crop', crop_options)

    if st.button('Get Parameters'):
        response = requests.get(f'{FLASK_SERVER_URL}/get_parameters', params={'crop_type': selected_crop})
        if response.status_code == 200:
            result = response.json()
            st.write('Optimal Parameters:')
            for key, value in result.items():
                st.write(f"{key}: {value}")
            
            # Store the fetched parameters in the session state
            st.session_state['fetched_parameters'] = result
            
            # Display the "Send Parameters to Arduino" button
            st.session_state['display_send_button'] = True
            
            #st.experimental_rerun()

    # Send Parameters to Arduino Button
    if st.session_state.get('display_send_button', False) and st.button('Send Parameters to Arduino'):
        payload = {
            'temperature': st.session_state['fetched_parameters'].get('temperature', ''),
            'humidity': st.session_state['fetched_parameters'].get('humidity', ''),
            'soil_moisture': st.session_state['fetched_parameters'].get('soil_moisture', '')
        }
        response = requests.post(f"{FLASK_SERVER_URL}/send_parameters", json=payload)
        if response.status_code == 200:
            st.success('Parameters sent to Arduino successfully!')  
        else:
            st.error('Failed to send parameters to Arduino')

# Function for manual control
def manual_control():
    st.title('Manual Control')
    st.write('Enter the following parameters manually:')

    temperature = st.text_input('Temperature', value='')
    humidity = st.text_input('Humidity', value='')
    soil_moisture = st.text_input('Soil Moisture', value='')

    N = st.text_input('Nitrogen (N)', value='')
    P = st.text_input('Phosphorus (P)', value='')
    K = st.text_input('Potassium (K)', value='')
    ph = st.text_input('pH', value='')

    # Send Parameters to Arduino Button
    if st.button('Send Parameters to Arduino'):
        payload = {
            'N': N,
            'P': P,
            'K': K,
            'temperature': temperature,
            'humidity': humidity,
            'ph': ph,
            'soil_moisture': soil_moisture
        }
        response = requests.post(f"{FLASK_SERVER_URL}/send_parameters", json=payload)
        if response.status_code == 200:
            st.success('Parameters sent to Arduino successfully!')
        else:
            st.error('Failed to send parameters to Arduino')

# Main function to control the sidebar and options
def main():
    st.sidebar.title("Select Option")
    option = st.sidebar.selectbox('Choose a mode', ['Crop Prediction', 'Parameter Prediction', 'Manual Control'])

    if option == 'Crop Prediction':
        crop_prediction()
    elif option == 'Parameter Prediction':
        parameter_prediction()
    elif option == 'Manual Control':
        manual_control()

if __name__ == "__main__":
    main()
