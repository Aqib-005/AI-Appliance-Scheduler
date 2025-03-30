from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from aiocache import cached
import xgboost as xgb
import logging

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the pre-trained model and scaler
try:
    model = joblib.load('xgb_model.pkl')
    scaler = joblib.load('scaler_xgb.pkl')
except FileNotFoundError as e:
    logger.error(f"Model or scaler file not found: {e}")
    raise HTTPException(status_code=500, detail="Model or scaler file not found.")

# Define the input data model for the API
class PredictionRequest(BaseModel):
    start_date: str  # Expected format: 'YYYY-MM-DD HH:MM:SS' or 'YYYY-MM-DD'

# Define the list of features used during training
features = [
    'temperature', 'precipitation', 'relative_humidity', 'total_consumption',
    'total_generation', 'Hour', 'Hour_sin', 'Hour_cos', 'Lag_Price_1h',
    'Lag_Price_24h', 'Rolling_Temp_24h', 'Rolling_Load_24h', 'Price_RollingStd24'
]

# Cache predictions for 1 hour
@app.post("/predict")
@cached(ttl=3600)
async def predict(request: PredictionRequest):
    try:
        predictions = predict_future_prices(request.start_date)
        return {"predictions": predictions}
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Also define /schedule for compatibility with your Laravel app
@app.post("/schedule")
@cached(ttl=3600)
async def schedule(request: PredictionRequest):
    try:
        predictions = predict_future_prices(request.start_date)
        return {"predictions": predictions}
    except Exception as e:
        logger.error(f"Schedule prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def predict_future_prices(start_date):
    """
    Predict future electricity prices for a given start date.
    The forecast window is 7 days (hourly). If the requested window extends beyond
    the available future data, missing hours are filled with fallback values computed
    as the historical average over the last 48 hours.
    """
    try:
        # Load CSVs
        historical_data = pd.read_csv("data/merged-data.csv")
        future_data = pd.read_csv("data/future-data.csv")
        
        if historical_data.empty or future_data.empty:
            raise HTTPException(status_code=500, detail="Historical or future data is empty.")
        
        historical_data = load_and_preprocess_data(historical_data)
        future_data = load_and_preprocess_data(future_data)
        
        logger.info(f"historical_data 'StartDateTime' dtype after preprocessing: {historical_data['StartDateTime'].dtype}")
        logger.info(f"future_data 'StartDateTime' dtype after preprocessing: {future_data['StartDateTime'].dtype}")
        
        # Parse the requested start date
        try:
            user_start_dt = pd.to_datetime(start_date, errors='raise')
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid start_date format. Use 'YYYY-MM-DD HH:MM:SS'.")
        
        # Get available future date range
        min_future_dt = future_data['StartDateTime'].min()
        max_future_dt = future_data['StartDateTime'].max()
        
        # Define the forecast window: 7 days starting from user_start_dt
        forecast_start_dt = user_start_dt
        forecast_end_dt = user_start_dt + pd.DateOffset(days=7)
        
        # Create complete hourly index for forecast window
        full_forecast_index = pd.date_range(start=forecast_start_dt, end=forecast_end_dt, freq='h')
        
        # Filter future_data to rows within available dates
        available_future = future_data[
            (future_data['StartDateTime'] >= forecast_start_dt) &
            (future_data['StartDateTime'] <= max_future_dt)
        ]
        
        # If no rows are available, create an empty DataFrame with correct dtypes
        if available_future.empty:
            available_future = pd.DataFrame(columns=future_data.columns).astype(future_data.dtypes.to_dict())
        
        logger.info(f"available_future 'StartDateTime' dtype: {available_future['StartDateTime'].dtype}")
        
        # Identify missing timestamps
        available_index = pd.Index(available_future['StartDateTime'])
        missing_index = full_forecast_index.difference(available_index)
        
        # Compute fallback values from historical data
        last_hist = historical_data.tail(48)
        if len(last_hist) < 48:
            raise HTTPException(status_code=500, detail="Insufficient historical data for fallback values.")
        engineered_hist = engineer_features(last_hist.copy())
        fallback_values = {feat: engineered_hist[feat].mean() for feat in features}
        
        # Create DataFrame for missing hours with fallback values
        if not missing_index.empty:
            missing_df = pd.DataFrame(index=missing_index, columns=['StartDateTime'] + features)
            missing_df['StartDateTime'] = missing_index
            for feat in features:
                missing_df[feat] = fallback_values[feat]
            missing_df = missing_df.reset_index(drop=True)
        else:
            missing_df = pd.DataFrame(columns=['StartDateTime'] + features).astype(
                {'StartDateTime': 'datetime64[ns]'} | {feat: 'float64' for feat in features}
            )
        
        logger.info(f"missing_df 'StartDateTime' dtype: {missing_df['StartDateTime'].dtype}")
        
        # Combine available_future and missing_df
        full_future = pd.concat([available_future, missing_df], ignore_index=True)
        full_future['StartDateTime'] = pd.to_datetime(full_future['StartDateTime'], errors='coerce')
        
        logger.info(f"full_future 'StartDateTime' dtype after concat: {full_future['StartDateTime'].dtype}")
        
        # Reindex to full forecast index and forward-fill
        full_future.set_index('StartDateTime', inplace=True)
        full_future = full_future.reindex(full_forecast_index).ffill().reset_index()
        full_future.rename(columns={'index': 'StartDateTime'}, inplace=True)
        full_future['StartDateTime'] = pd.to_datetime(full_future['StartDateTime'], errors='coerce')
        
        if full_future['StartDateTime'].isnull().any():
            raise ValueError("Some 'StartDateTime' values became NaT after reindexing.")
        
        logger.info(f"full_future 'StartDateTime' dtype before engineer_features: {full_future['StartDateTime'].dtype}")
        
        # Engineer features
        full_future = engineer_features(full_future)
        
        # Prepare for iterative prediction
        price_history = list(historical_data.tail(48)['Price'])
        predictions = []
        
        for idx, row in full_future.iterrows():
            feature_vector = []
            for feat in features:
                if feat in ['Lag_Price_1h', 'Lag_Price_24h']:
                    lag = int(feat.split('_')[-1].replace('h', ''))
                    value = predictions[-lag] if len(predictions) >= lag else price_history[-lag]
                    feature_vector.append(value)
                elif 'Rolling' in feat or 'Std' in feat:
                    value = row[feat] if pd.notna(row[feat]) else fallback_values.get(feat, 0)
                    feature_vector.append(value)
                else:
                    value = row[feat] if pd.notna(row[feat]) else fallback_values.get(feat, 0)
                    feature_vector.append(value)
            
            # Scale and predict
            feature_vector_scaled = scaler.transform([feature_vector])
            pred_price = model.predict(feature_vector_scaled)[0]
            predictions.append(pred_price)
            price_history.append(pred_price)
        
        full_future['Predicted_Price'] = predictions
        return full_future[['StartDateTime', 'Predicted_Price']].to_dict(orient='records')
    
    except Exception as e:
        logger.error(f"Error in predict_future_prices: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def load_and_preprocess_data(data):
    """Preprocess data consistently across models."""
    try:
        data.columns = data.columns.str.strip().str.lower()
        mapping = {
            'start date/time': 'StartDateTime',
            'temperature': 'temperature',
            'relative humidity': 'relative_humidity',
            'precipitation': 'precipitation',
            'rain': 'rain',
            'snowfall': 'snowfall',
            'weather code': 'weather_code',
            'wind speed': 'wind_speed',
            'price': 'Price',
            'grid load': 'total_consumption',
            'total generation': 'total_generation',
            'day of week': 'DayOfWeek'
        }
        for col in data.columns:
            if col in mapping:
                data.rename(columns={col: mapping[col]}, inplace=True)
        
        data['StartDateTime'] = pd.to_datetime(data['StartDateTime'], format='%d/%m/%Y %H:%M', errors='coerce')
        if data['StartDateTime'].isnull().any():
            raise ValueError("Some 'StartDateTime' values could not be parsed.")
        
        if not pd.api.types.is_datetime64_any_dtype(data['StartDateTime']):
            raise TypeError("'StartDateTime' is not datetime type after conversion.")
        
        potential_numeric_cols = [
            'Price', 'total_consumption', 'temperature', 'precipitation', 'rain',
            'snowfall', 'wind_speed', 'relative_humidity', 'weather_code', 'total_generation'
        ]
        numeric_cols = [col for col in potential_numeric_cols if col in data.columns]
        for col in numeric_cols:
            data[col] = pd.to_numeric(
                data[col].astype(str).str.replace(',', '').str.replace(' ', '').str.replace('–', '-'),
                errors='coerce'
            )
        
        data = data.sort_values('StartDateTime').dropna(subset=['StartDateTime']).reset_index(drop=True)
        
        day_map = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
                   'friday': 4, 'saturday': 5, 'sunday': 6}
        if 'DayOfWeek' in data.columns:
            data['DayOfWeek'] = data['DayOfWeek'].str.strip().str.lower().map(day_map)
            data = data.dropna(subset=['DayOfWeek'])
        
        for col in data.columns:
            if data[col].dtype in [np.float64, np.int64]:
                data[col] = data[col].interpolate(method='linear', limit_direction='both').ffill().bfill()
        
        return data
    except Exception as e:
        logger.error(f"Error in load_and_preprocess_data: {e}")
        raise

def engineer_features(data):
    """Engineer features consistently across models."""
    try:
        df = data.copy()
        
        if 'StartDateTime' not in df.columns:
            raise ValueError("'StartDateTime' column is missing.")
        
        df['StartDateTime'] = pd.to_datetime(df['StartDateTime'], errors='coerce')
        if df['StartDateTime'].isnull().any():
            raise ValueError("Some 'StartDateTime' values could not be converted to datetime.")
        
        if not pd.api.types.is_datetime64_any_dtype(df['StartDateTime']):
            raise TypeError("'StartDateTime' is not of datetime type.")
        
        logger.info(f"engineer_features 'StartDateTime' dtype: {df['StartDateTime'].dtype}")
        
        df['Hour'] = df['StartDateTime'].dt.hour
        df['Day'] = df['StartDateTime'].dt.day
        df['DayOfWeek'] = pd.to_numeric(df['DayOfWeek'], errors='coerce')
        df['Hour'] = pd.to_numeric(df['Hour'], errors='coerce')
        df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        
        if 'Price' in df.columns:
            df['Lag_Price_1h'] = df['Price'].shift(1)
            df['Lag_Price_24h'] = df['Price'].shift(24)
            df['Price_RollingStd24'] = df['Price'].rolling(window=24, min_periods=1).std()
            df['Lag_Price_1h'] = df['Lag_Price_1h'].interpolate(method='linear', limit_direction='both').ffill().bfill()
            df['Lag_Price_24h'] = df['Lag_Price_24h'].interpolate(method='linear', limit_direction='both').ffill().bfill()
            df['Price_RollingStd24'] = df['Price_RollingStd24'].interpolate(method='linear', limit_direction='both').ffill().bfill()
        else:
            df['Lag_Price_1h'] = np.nan
            df['Lag_Price_24h'] = np.nan
            df['Price_RollingStd24'] = np.nan
        
        df['Rolling_Temp_24h'] = df['temperature'].rolling(window=24, min_periods=1).mean()
        df['Rolling_Load_24h'] = df['total_consumption'].rolling(window=24, min_periods=1).mean()
        
        return df
    except Exception as e:
        logger.error(f"Error in engineer_features: {e}")
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)