from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Station data with names and TDS levels for stations 1001 to 1012
stations = {
    "1001": {"name": "Alipur", "tds": 250.5},
    "1002": {"name": "Civil Lines", "tds": 751.0},
    "1003": {"name": "Defence Colony", "tds": 451.5},
    "1004": {"name": "Dwarka", "tds": 952.0},
    "1005": {"name": "Karol Bagh", "tds": 352.5},
    "1006": {"name": "Mehrauli", "tds": 753.0},
    "1007": {"name": "Najafgarh", "tds": 593.5},
    "1008": {"name": "Narela", "tds": 254.0},
    "1009": {"name": "New Delhi", "tds": 254.5},
    "1010": {"name": "Paharganj", "tds": 585.0},
    "1011": {"name": "Rohini", "tds": 255.5},
    "1012": {"name": "Shahdara", "tds": 856.0},
}

# Function to predict water depth based on rainfall (in mm)
def predict_water_depth(rainfall):
    if rainfall > 10:
        return 0, "Overflow"
    depth = 25 - (2 * rainfall)
    if depth <= 5:
        state = "Underflow"
    else:
        state = "Moderate"
    return depth, state

@app.route("/")
def home():
    return render_template("index.html", stations=stations)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract rainfall and station number from form data
        station_no = request.form["station_no"]
        rainfall = float(request.form["rainfall"])
        station = stations.get(station_no, {})
        depth, state = predict_water_depth(rainfall)
        response = {
            "station_name": station.get("name", "Unknown Station"),
            "tds_level": station.get("tds", "N/A"),
            "predicted_depth": depth,
            "flow_state": state
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
