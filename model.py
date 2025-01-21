import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle

class WaterLevelPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def load_data(self, file_path):
        return pd.read_csv(file_path)

    def prepare_features(self, data):
        data['station_no'] = data['station_no'].astype(int)
        X = data[['station_no', 'Rainfall(mm)', 'TDS(mg/L)']]
        y = data['Depth(m)']
        return X, y

    def train_model(self, file_path):
        data = self.load_data(file_path)
        X, y = self.prepare_features(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        self.model.fit(X_train_scaled, y_train)

        print(f"Training Score: {self.model.score(X_train_scaled, y_train):.4f}")
        print(f"Testing Score: {self.model.score(X_test_scaled, y_test):.4f}")

        with open('model/water_level_predictor.pkl', 'wb') as file:
            pickle.dump({'model': self.model, 'scaler': self.scaler}, file)

if __name__ == '__main__':
    predictor = WaterLevelPredictor()
    predictor.train_model('data/b1.csv')
