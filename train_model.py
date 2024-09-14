import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load the dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Prepare the ML model pipeline
def build_model():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Scaling features
        ('model', RandomForestRegressor())  # Random Forest Regressor
    ])

    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [None, 10, 20, 30],
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    return grid_search

# Train the model
def train_model(file_path):
    data = load_data(file_path)

    # Using 'Flow Rate (lpm)' and 'TDS(mg/L)' as features to predict 'Depth(m)'
    X = data[['Flow Rate (lpm)', 'TDS(mg/L)']]
    y = data['Depth(m)']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_model()
    model.fit(X_train, y_train)

    # Save the trained model using pickle
    with open('model/water_level_predictor.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Testing accuracy on test set
    accuracy = model.score(X_test, y_test)

    return accuracy

# Train the model and save it
train_model('data/b1.csv')
