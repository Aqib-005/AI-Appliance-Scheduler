import json
import joblib
import numpy as np
from tensorflow.keras.models import load_model

# Load the models and scaler
rf_model = joblib.load('random_forest_model.pkl')
lstm_model = load_model('lstm_model.h5')
scaler = joblib.load('scaler.pkl')

# Generate predicted prices for the next 24 hours
def predict_prices():
    # Example: Generate 24 hours of input data
    predicted_prices = {}
    for hour in range(24):
        # Create dummy input data (replace with actual features)
        input_data = [20, 10, 500, 1, hour, 50]  # Example: [temperature, wind_speed, grid_consumption, day, hour, lag_price]
        input_scaled = scaler.transform([input_data])
        input_reshaped = input_scaled.reshape(1, 1, -1)

        # Make predictions
        rf_pred = rf_model.predict(input_scaled)
        lstm_residual = lstm_model.predict(input_reshaped).flatten()
        final_prediction = rf_pred[0] + lstm_residual[0]

        # Store the predicted price
        predicted_prices[f"{hour}:00"] = final_prediction

    return predicted_prices

# Send the predicted prices back to Laravel
print(json.dumps(predict_prices()))