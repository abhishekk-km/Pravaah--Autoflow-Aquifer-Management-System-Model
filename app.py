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

# Helper function to get station info
def get_station_info(station_no):
    station = df[df['station_no'] == station_no]
    if station.empty:
        return None, None
    tds_level = station['TDS(mg/L)'].values[0]
    station_name = station['station_name'].values[0]
    return tds_level, station_name

# Helper function to predict water level and flow state
def predict_flow(flow_rate, tds_level):
    features = np.array([[flow_rate, tds_level]])
    predicted_depth = model.predict(features)[0]
    
    if predicted_depth < 2:
        flow_state = "Overflow"
    elif 2 <= predicted_depth <= 30:
        flow_state = "Moderate Flow"
    else:
        flow_state = "Underflow"
    
    return predicted_depth, flow_state

# Home route
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            station_no = int(request.form.get('station_no'))
        except ValueError:
            return render_template('index.html', error="Please enter a valid station number.")
        
        try:
            flow_rate = float(request.form.get('flow_rate'))
        except ValueError:
            return render_template('index.html', error="Please enter a valid flow rate.")
        
        # Get station info
        tds_level, station_name = get_station_info(station_no)
        if station_name is None:
            return render_template('index.html', error="Invalid station number")
        
        # Predict water depth and flow state
        water_depth, flow_state = predict_flow(flow_rate, tds_level)

        return render_template('index.html', station_name=station_name, tds_level=tds_level, water_depth=water_depth, flow_state=flow_state)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
