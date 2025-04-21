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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    model = joblib.load('xgb_model.pkl')
    scaler = joblib.load('scaler_xgb.pkl')
except FileNotFoundError as e:
    logger.error(f"Model or scaler file not found: {e}")
    raise HTTPException(status_code=500, detail="Model or scaler file not found.")

class PredictionRequest(BaseModel):
    start_date: str

features = [
    'temperature', 'precipitation', 'relative_humidity', 'total_consumption',
    'total_generation', 'Hour', 'Hour_sin', 'Hour_cos', 'Lag_Price_1h',
    'Lag_Price_24h', 'Rolling_Temp_24h', 'Rolling_Load_24h', 'Price_RollingStd24'
]

@app.post("/predict")
@cached(ttl=3600)
async def predict(request: PredictionRequest):
    try:
        predictions = predict_future_prices(request.start_date)
        return {"predictions": predictions}
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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

    historical_data = pd.read_csv("data/merged-data.csv")
    future_data = pd.read_csv("data/future-data.csv")
    
    if historical_data.empty:
        raise HTTPException(status_code=500, detail="Historical data is empty.")
    
    historical_data = load_and_preprocess_data(historical_data)
    future_data = load_and_preprocess_data(future_data) if not future_data.empty else pd.DataFrame()

    user_start_dt = pd.to_datetime(start_date)
    forecast_end_dt = user_start_dt + pd.DateOffset(days=7)
    forecast_index = pd.date_range(user_start_dt, forecast_end_dt, freq='H')

    if not future_data.empty:
        max_future_dt = future_data['StartDateTime'].max()
        in_range = future_data[
            (future_data['StartDateTime'] >= user_start_dt) &
            (future_data['StartDateTime'] <= max_future_dt)
        ]
    else:
        in_range = pd.DataFrame()

    have_index = in_range['StartDateTime'] if not in_range.empty else pd.Index([])
    missing_index = forecast_index.difference(have_index)

    last_year_filled = []
    if not missing_index.empty:
        for ts in missing_index:
            last_year_ts = ts - pd.DateOffset(years=1)
            # We'll try to find a row in historical_data for that exact last_year_ts
            row = historical_data[historical_data['StartDateTime'] == last_year_ts]
            if not row.empty:
                new_row = row.copy()
                new_row['StartDateTime'] = ts  # Shift the timestamp to the future year
                last_year_filled.append(new_row)
    
    # Combine all the last-year-filled rows into a DataFrame
    if last_year_filled:
        last_year_df = pd.concat(last_year_filled, ignore_index=True)
    else:
        last_year_df = pd.DataFrame(columns=historical_data.columns)
        
    still_missing_index = missing_index.difference(last_year_df['StartDateTime']) if not last_year_df.empty else missing_index
    random_fallback_rows = []
    if not still_missing_index.empty:
        # We'll do a minimal random approach around the means of the last 48 hours
        last_hist_48 = historical_data.tail(48)
        means_48 = last_hist_48[features].mean(numeric_only=True)
        stds_48 = last_hist_48[features].std(numeric_only=True)

        for ts in still_missing_index:
            # create a row dict
            row_dict = {'StartDateTime': ts}
            for feat in features:
                # random value ~ N(mean, std)
                # or 0 if std is NaN
                mean_val = means_48.get(feat, 0)
                std_val = stds_48.get(feat, 0)
                rand_val = np.random.normal(mean_val, std_val if pd.notna(std_val) else 1e-3)
                row_dict[feat] = rand_val
            random_fallback_rows.append(row_dict)
    
    random_fallback_df = pd.DataFrame(random_fallback_rows)
    
    combined_df = pd.concat([in_range, last_year_df, random_fallback_df], ignore_index=True)
    
    # 9. Reindex to the full forecast window, forward-fill if needed
    combined_df.set_index('StartDateTime', inplace=True)
    combined_df = combined_df.sort_index().reindex(forecast_index).ffill().reset_index()
    combined_df.rename(columns={'index': 'StartDateTime'}, inplace=True)

    # 10. Engineer features on the combined data
    combined_df = engineer_features(combined_df)

    # 11. Iterative Price Forecast
    price_history = list(historical_data.tail(48)['Price'])
    predictions = []
    for idx, row in combined_df.iterrows():
        feat_vec = []
        for feat in features:
            if feat in ['Lag_Price_1h', 'Lag_Price_24h']:
                lag = int(feat.split('_')[-1].replace('h', ''))
                if len(predictions) >= lag:
                    val = predictions[-lag]
                else:
                    val = price_history[-lag]
                feat_vec.append(val)
            else:
                feat_vec.append(row.get(feat, 0))

        scaled = scaler.transform([feat_vec])
        pred_price = model.predict(scaled)[0]
        predictions.append(pred_price)
        price_history.append(pred_price)

    combined_df['Predicted_Price'] = predictions
    return combined_df[['StartDateTime','Predicted_Price']].to_dict(orient='records')

def load_and_preprocess_data(df):
    df.columns = df.columns.str.strip().str.lower()
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
    for c in df.columns:
        if c in mapping:
            df.rename(columns={c: mapping[c]}, inplace=True)

    df['StartDateTime'] = pd.to_datetime(df['StartDateTime'], format='%d/%m/%Y %H:%M', errors='coerce')
    df.dropna(subset=['StartDateTime'], inplace=True)
    df.sort_values('StartDateTime', inplace=True)
    df.reset_index(drop=True, inplace=True)

    numeric_cols = ['Price','total_consumption','temperature','precipitation','rain','snowfall',
                    'wind_speed','relative_humidity','weather_code','total_generation']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = (df[c].astype(str)
                        .str.replace(',', '')
                        .str.replace(' ', '')
                        .str.replace('–', '-')
                     )
            df[c] = pd.to_numeric(df[c], errors='coerce')
            df[c] = df[c].interpolate(method='linear', limit_direction='both').ffill().bfill()

    return df

def engineer_features(df):
    data = df.copy()
    data['Hour'] = data['StartDateTime'].dt.hour
    data['Day'] = data['StartDateTime'].dt.day
    data['Hour_sin'] = np.sin(2 * np.pi * data['Hour'] / 24)
    data['Hour_cos'] = np.cos(2 * np.pi * data['Hour'] / 24)

    # If 'Price' in columns, create lag
    if 'Price' in data.columns:
        data['Lag_Price_1h'] = data['Price'].shift(1)
        data['Lag_Price_24h'] = data['Price'].shift(24)
        data['Price_RollingStd24'] = data['Price'].rolling(24, min_periods=1).std()
        for c in ['Lag_Price_1h','Lag_Price_24h','Price_RollingStd24']:
            data[c] = data[c].interpolate(method='linear', limit_direction='both').ffill().bfill()
    else:
        data['Lag_Price_1h'] = np.nan
        data['Lag_Price_24h'] = np.nan
        data['Price_RollingStd24'] = np.nan

    if 'temperature' in data.columns:
        data['Rolling_Temp_24h'] = data['temperature'].rolling(24, min_periods=1).mean()
    else:
        data['Rolling_Temp_24h'] = np.nan

    if 'total_consumption' in data.columns:
        data['Rolling_Load_24h'] = data['total_consumption'].rolling(24, min_periods=1).mean()
    else:
        data['Rolling_Load_24h'] = np.nan

    return data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
