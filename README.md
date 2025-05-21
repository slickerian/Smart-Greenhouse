The Smart Greenhouse System is an innovative IoT and AI-driven agricultural solution designed to enhance greenhouse farming by automating environmental control and predicting the most suitable crops based on real-time sensor data. This system integrates machine learning-based crop recommendations, automated temperature, humidity, and irrigation control, and a user-friendly web interface that enables remote monitoring and manual adjustments. By leveraging real-time data collection from DHT11 and soil moisture sensors, the system ensures optimal growing conditions, reducing manual intervention while maximizing crop yield and resource efficiency.

One of the key features of this system is its AI-powered crop prediction model, which analyzes environmental parameters and suggests the best crops suited for the current conditions, helping farmers and greenhouse operators make informed decisions. Additionally, the system includes automated control mechanisms that adjust temperature, humidity, and soil moisture levels in real-time, ensuring a stable and optimized environment for plant growth. For added flexibility, it provides manual controls, allowing users to override automation and set parameters based on specific requirements.

To further enhance reliability, the system incorporates an emergency shutdown mechanism, which detects anomalies such as extreme temperature fluctuations, excessive humidity, or system failures and takes preventive actions to safeguard the greenhouse environment. The Streamlit-based web interface makes it easy for users to interact with the system, view real-time data, receive AI-driven recommendations, and control the greenhouse settings from anywhere.

By integrating machine learning, IoT, and automation, this system not only improves agricultural efficiency but also promotes sustainable farming practices by minimizing water and energy waste. With its ability to dynamically adapt to changing environmental conditions and optimize plant growth, the Smart Greenhouse System represents a significant advancement in precision agriculture, making greenhouse farming more efficient, reliable, and accessible.

Just a smart greenhouse that uses ML models to predict the type of crop that can be grown in the soil as well as predict crop parameters based on the input crop. Uses simple streamlit for the frontend. Was planning to integrate it with hardware that can control the soil parameters, but that future work. For now, the website can use the models to predict.
Please place the models folder in the same directory as the other files.
Run flaskserver.py
Run frontend.py
Open the website and enjoy!

![image](https://github.com/user-attachments/assets/0d78a2d8-245c-4982-a988-6348c4eaa2a3)


ps. please feel free to change, modify and give me ideas on how i can implement it or how i can improve it.
You are free to use this is your own hackathon/ideathon/projects/college work :)



Commands:

python -m venv greenhouse

pip install requirements.txt

python flaskserver.py

streamlit run frontend.py


