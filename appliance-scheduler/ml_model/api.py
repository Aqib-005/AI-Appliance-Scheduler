from fastapi import FastAPI
import pandas as pd
import numpy as np
from prophet import Prophet
import joblib
from pydantic import BaseModel

# Load the pre-trained model and scaler
model = joblib.load('xgb_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define the input data model for the API
class PredictionRequest(BaseModel):
    start_date: str  # Start date for predictions (e.g., "2024-01-01")

# Define the list of features used during training
features = [
    'temperature_2m (°C)', 
    'wind_speed_100m (km/h)', 
    'Total (grid consumption) [MWh]', 
    'Day', 
    'Hour', 
    'DayOfWeek',
    'Rolling_Temp_24h',
    'Rolling_Wind_24h',
    'Rolling_Load_24h',
    'Lag_Price'
]

# Initialize FastAPI app
app = FastAPI()

# Function to predict future prices
def predict_future_prices(start_date):
    # Load historical data
    data = pd.read_csv("data/merged-data.csv")

    # Data cleaning: Remove commas from numeric columns
    data['Price Germany/Luxembourg [Euro/MWh]'] = data['Price Germany/Luxembourg [Euro/MWh]'].replace({',': ''}, regex=True).astype(float)
    data['Total (grid consumption) [MWh]'] = data['Total (grid consumption) [MWh]'].replace({',': ''}, regex=True).astype(float)

    # Convert Start date/time to datetime
    data['Start date/time'] = pd.to_datetime(data['Start date/time'], dayfirst=True)

    # Generate future dates
    last_date = data['Start date/time'].max()
    future_dates = pd.date_range(start=start_date, periods=7*24, freq='H')

    # Create future DataFrame
    future_data = pd.DataFrame(index=future_dates)
    future_data['Year'] = future_data.index.year
    future_data['Month'] = future_data.index.month
    future_data['Day'] = future_data.index.day
    future_data['Hour'] = future_data.index.hour
    future_data['DayOfWeek'] = future_data.index.dayofweek
    future_data['Lag_Price'] = data['Price Germany/Luxembourg [Euro/MWh]'].iloc[-1]

    # Forecast weather and load data using Prophet
    def forecast_feature(data, feature, periods=7*24):
        feature_data = data[['Start date/time', feature]].rename(columns={'Start date/time': 'ds', feature: 'y'})
        model = Prophet()
        model.fit(feature_data)
        future = model.make_future_dataframe(periods=periods, freq='H')
        forecast = model.predict(future)
        return forecast['yhat'].tail(periods).values

    future_data['temperature_2m (°C)'] = forecast_feature(data, 'temperature_2m (°C)')
    future_data['wind_speed_100m (km/h)'] = forecast_feature(data, 'wind_speed_100m (km/h)')
    future_data['Total (grid consumption) [MWh]'] = forecast_feature(data, 'Total (grid consumption) [MWh]')

    # Add rolling averages for future data
    future_data['Rolling_Temp_24h'] = future_data['temperature_2m (°C)'].rolling(window=24).mean()
    future_data['Rolling_Wind_24h'] = future_data['wind_speed_100m (km/h)'].rolling(window=24).mean()
    future_data['Rolling_Load_24h'] = future_data['Total (grid consumption) [MWh]'].rolling(window=24).mean()
    future_data.fillna(method='bfill', inplace=True)

    # Predict future prices iteratively
    xgb_future_pred = []
    for i in range(len(future_data)):
        # Scale the features for the current row
        future_data_scaled = scaler.transform(future_data.iloc[i:i+1][features])
        pred = model.predict(future_data_scaled)
        xgb_future_pred.append(pred[0])
        if i < len(future_data) - 1:
            future_data.at[future_data.index[i+1], 'Lag_Price'] = pred[0]

    # Save predictions
    future_predictions_df = pd.DataFrame({
        'Start date/time': future_dates,
        'Predicted Price [Euro/MWh]': xgb_future_pred
    })
    return future_predictions_df.to_dict(orient='records')

# Define the API endpoint
@app.post("/predict")
def predict(request: PredictionRequest):
    predictions = predict_future_prices(request.start_date)
    return {"predictions": predictions}

# Run the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)