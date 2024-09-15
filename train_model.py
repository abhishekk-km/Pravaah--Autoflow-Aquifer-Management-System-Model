import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pickle
import numpy as np

# Load the dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Custom prediction logic for depth
def custom_depth_prediction(flow_rate, tds_level):
    if flow_rate < 5:
        # As flow rate decreases, depth increases (up to 30 meters)
        return 30 - (flow_rate * 5)
    else:
        # As flow rate increases beyond 5 lpm, depth becomes negative
        return -(flow_rate - 5)

# Prepare the ML model pipeline (Polynomial Regression)
def build_model():
    pipeline = Pipeline([
        ('polynomial_features', PolynomialFeatures(degree=2)),  # Adds polynomial features to model non-linearity
        ('scaler', StandardScaler()),  # Scaling features
        ('model', LinearRegression())  # Linear Regression
    ])
    return pipeline

# Train the model with custom depth logic
def train_model(file_path):
    data = load_data(file_path)

    # Using 'Flow Rate (lpm)' and 'TDS(mg/L)' as features to predict 'Depth(m)'
    X = data[['Flow Rate (lpm)', 'TDS(mg/L)']]
    y = data['Depth(m)']

    # Apply the custom depth prediction logic based on flow rate and TDS
    y_custom = X.apply(lambda row: custom_depth_prediction(row['Flow Rate (lpm)'], row['TDS(mg/L)']), axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y_custom, test_size=0.2, random_state=42)

    model = build_model()
    model.fit(X_train, y_train)

    # Save the trained model using pickle
    with open('model/water_level_predictor.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Testing accuracy on the test set
    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy}")

    return accuracy

# Train the model and save it
train_model('data/b1.csv')

