from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_socketio import SocketIO
import paho.mqtt.client as mqtt
import torch
from safetensors.torch import load_file
import numpy as np

app = Flask(__name__)
socketio = SocketIO(app)

# MQTT configurations
mqtt_server = "broker.emqx.io"
mqtt_port = 1883
mqtt_topic_mq1 = "sensors/mq1"
mqtt_topic_mq7 = "sensors/mq7"
mqtt_topic_mq135 = "sensors/mq135"

# Global sensor data variables
sensor_data = {
    "mq1": "Methane: Waiting for data...",
    "mq7": "Sulphur Dioxide: Waiting for data...",
    "mq135": "Air quality: Waiting for data..."
}

# Load the saved model
MODEL_PATH = "pollution_model.safetensors"
model_state = load_file(MODEL_PATH)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PollutionModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(PollutionModel, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.fc(x)

input_size = 341  # Adjust to match the saved model's input size
output_size = 4  # SO2, NO2, PM10, PM2.5
model = PollutionModel(input_size, output_size)
model.load_state_dict(model_state)  # Load the state dictionary
model.to(device)
model.eval()

# Flask routes
@app.route('/')
def location_page():
    # Renders location.html as the initial page
    return render_template('location.html')

@app.route('/index')
def index():
    # Renders index.html for the sensor dashboard page
    return render_template('index.html')

@app.route('/forecast')
def forecast_page():
    # Renders forecast_page.html for pollution forecasting
    return render_template('forecast_page.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Handle prediction requests
    data = request.json
    city = data.get('city', 'Unknown')
    days = int(data.get('days', 1))

    # Example preprocessing (mock categorical and numerical data)
    # Replace this with actual preprocessing logic
    city_encoding = [1 if city == 'ExampleCity' else 0] * (input_size - 5)  # Mock encoding
    numerical_features = [days, 10, 20, 30, 40]  # Mock numerical features

    input_tensor = torch.tensor(city_encoding + numerical_features, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(input_tensor).cpu().numpy()[0]

    result = {
        "aqi": np.round(prediction[0], 2),
        "forecast": [
            {"SO2": round(prediction[0] + i, 2), "NO2": round(prediction[1] + i, 2), 
             "PM10": round(prediction[2] + i, 2), "PM25": round(prediction[3] + i, 2)}
            for i in range(1, days + 1)
        ]
    }
    return jsonify(result)

# MQTT callbacks
def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe(mqtt_topic_mq1)
    client.subscribe(mqtt_topic_mq7)
    client.subscribe(mqtt_topic_mq135)

def on_message(client, userdata, msg):
    global sensor_data
    print(msg.topic + " " + str(msg.payload.decode()))
    # Update the sensor data based on the topic
    if msg.topic == mqtt_topic_mq1:
        sensor_data['mq1'] = f"Methane: {msg.payload.decode()} ppm"
    elif msg.topic == mqtt_topic_mq7:
        sensor_data['mq7'] = f"Sulphur Dioxide: {msg.payload.decode()} ppb"
    elif msg.topic == mqtt_topic_mq135:
        sensor_data['mq135'] = f"Air quality: {msg.payload.decode()} AQI"

    # Emit the updated sensor data to all connected clients
    socketio.emit('update_sensor_data', sensor_data)

# MQTT setup with updated API
client = mqtt.Client(protocol=mqtt.MQTTv311)  # Use MQTTv311 or MQTTv5

# Attach callbacks
client.on_connect = on_connect
client.on_message = on_message

# Connect to the MQTT broker
client.connect(mqtt_server, mqtt_port, 60)
client.loop_start()

if __name__ == '__main__':
    # Use allow_unsafe_werkzeug for local development
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)
