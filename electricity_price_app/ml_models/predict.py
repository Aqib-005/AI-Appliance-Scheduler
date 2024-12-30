import os
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import json

# Use absolute paths for models
ml_models_dir = os.path.dirname(os.path.abspath(__file__))
rf_model = joblib.load(os.path.join(ml_models_dir, 'random_forest_model.pkl'))
lstm_model = load_model(os.path.join(ml_models_dir, 'lstm_model.h5'))
scaler = joblib.load(os.path.join(ml_models_dir, 'scaler.pkl'))

def predict_prices():
    feature_names = [
    'temperature_2m (°C)', 
    'wind_speed_100m (km/h)', 
    'Total (grid consumption) [MWh]', 
    'Day', 
    'Hour', 
    'Lag_Price'
]
    predicted_prices = {}
    for hour in range(24):
        input_data = pd.DataFrame([[20, 10, 500, 1, hour, 50]], columns=feature_names)
        input_scaled = scaler.transform(input_data)
        rf_pred = rf_model.predict(input_scaled)
        input_reshaped = np.concatenate(
            [input_scaled.reshape(1, 1, -1), np.array(rf_pred).reshape(1, 1, -1)],
            axis=2
        )
        lstm_residual = lstm_model.predict(input_reshaped).flatten()
        final_prediction = rf_pred[0] + lstm_residual[0]
        predicted_prices[f"{hour}:00"] = final_prediction

    return predicted_prices

if __name__ == "__main__":
    print(json.dumps(predict_prices()))
