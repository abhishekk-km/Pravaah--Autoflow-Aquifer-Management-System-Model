from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('data/b1.csv')

# Load the pre-trained model
with open('model/water_level_predictor.pkl', 'rb') as f:
    model = pickle.load(f)

# Extract unique station numbers for the dropdown
station_numbers = df['station_no'].unique()

# Helper function to get station info
def get_station_info(station_no):
    station = df[df['station_no'] == station_no]
    if station.empty:
        return None, None
    tds_level = station['TDS(mg/L)'].values[0]
    station_name = station['station_name'].values[0]
    return tds_level, station_name

# Helper function to predict water level and flow state using user-provided flow rate
def predict_flow(flow_rate, tds_level):
    features = np.array([[flow_rate, tds_level]])  # Use user-provided flow rate and TDS level
    predicted_depth = model.predict(features)[0]

    # Handling negative predicted depth values
    if predicted_depth < 0:
        # Logarithmic transformation for negative depth values
        predicted_depth_log = np.log10(abs(predicted_depth)) / np.log10(100)
        flow_state = "Water above the ground level"
        return predicted_depth_log, flow_state
    
    if predicted_depth < 2:
        flow_state = "Overflow"
    elif 2 <= predicted_depth <= 25:
        flow_state = "Moderate Flow"
    else:
        flow_state = "Underflow"
    
    return predicted_depth, flow_state

# Home route
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Get the station number input
            station_no = int(request.form.get('station_no'))
        except ValueError:
            return render_template('index.html', station_numbers=station_numbers, error="Please select a valid station number.")
        
        try:
            # Get the flow rate input from the user
            flow_rate = float(request.form.get('flow_rate'))
        except ValueError:
            return render_template('index.html', station_numbers=station_numbers, error="Please enter a valid flow rate.")
        
        # Get the TDS level and station name based on the selected station number
        tds_level, station_name = get_station_info(station_no)
        if station_name is None:
            return render_template('index.html', station_numbers=station_numbers, error="Invalid station number")
        
        # Predict water depth and flow state using user input for flow rate and the station's TDS level
        water_depth, flow_state = predict_flow(flow_rate, tds_level)

        # Render the result with the predicted values
        return render_template('index.html', station_numbers=station_numbers, station_name=station_name, tds_level=tds_level, water_depth=water_depth, flow_state=flow_state)
    
    # On a GET request, render the form without any results
    return render_template('index.html', station_numbers=station_numbers)

if __name__ == '__main__':
    app.run(debug=True)

