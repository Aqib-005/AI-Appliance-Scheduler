import os
import json
import joblib
import numpy as np
from tensorflow.keras.models import load_model

# Get the full path to the ml_models folder
ml_models_dir = os.path.dirname(os.path.abspath(__file__))

# Load the models and scaler using the full path
rf_model = joblib.load(os.path.join(ml_models_dir, 'random_forest_model.pkl'))
lstm_model = load_model(os.path.join(ml_models_dir, 'lstm_model.h5'))
scaler = joblib.load(os.path.join(ml_models_dir, 'scaler.pkl'))

# Feature names used during training
feature_names = [
    'temperature_2m (°C)', 
    'wind_speed_100m (km/h)', 
    'Total (grid consumption) [MWh]', 
    'Day', 
    'Hour', 
    'Lag_Price'
]

# Generate predicted prices for the next 24 hours
def predict_prices():
    # Example: Generate 24 hours of input data
    predicted_prices = {}
    for hour in range(24):
        # Create input data with the correct number of features
        input_data = [20, 10, 500, 1, hour, 50]  # 6 features (same as during training)
        
        # Scale the input data
        input_scaled = scaler.transform([input_data])
        
        # Predict with Random Forest
        rf_pred = rf_model.predict(input_scaled)
        
        # Prepare input for LSTM
        # Reshape the input to match the LSTM's expected shape: (1, 1, 7)
        # The LSTM expects 7 features because it was trained with the residuals appended
        input_reshaped = np.concatenate([
            input_scaled.reshape(1, 1, -1),  # Original features (6)
            np.array(rf_pred).reshape(1, 1, -1)  # RF predictions (1)
        ], axis=2)
        
        # Predict residuals with LSTM
        lstm_residual = lstm_model.predict(input_reshaped).flatten()
        
        # Combine predictions
        final_prediction = rf_pred[0] + lstm_residual[0]
        
        # Store the predicted price
        predicted_prices[f"{hour}:00"] = final_prediction

    return predicted_prices

# Send the predicted prices back to Laravel
print(json.dumps(predict_prices()))