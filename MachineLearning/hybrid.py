import pandas as pd
import numpy as np
import pywt  # For wavelet transforms
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import xgboost as xgb
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler
import joblib

# -------------------------------
# Utility Functions
# -------------------------------

def clean_numeric_column(series):
    """Replace commas with dots, remove spaces, and convert to numeric."""
    return pd.to_numeric(
        series.astype(str)
              .str.replace(',', '.')  # Replace comma with dot
              .str.replace(' ', '')   # Remove spaces
              .str.replace('–', '-')  # Replace en dash with minus
              .str.replace(',', ''),  # Remove any remaining commas
        errors='coerce'
    )

def wavelet_denoise(series, wavelet='sym4', level=3):
    """
    Denoise a signal using discrete wavelet transform with partial soft thresholding.
    This is applied only on historical data.
    """
    coeffs = pywt.wavedec(series, wavelet, level=level)
    detail_coeffs = coeffs[1:]
    sigma = np.median(np.abs(detail_coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(series)))
    denoised_coeffs = [coeffs[0]]
    for i, detail in enumerate(detail_coeffs, start=1):
        # Threshold only the finest level(s)
        if i >= level:
            denoised_detail = pywt.threshold(detail, uthresh, mode='soft')
        else:
            denoised_detail = detail
        denoised_coeffs.append(denoised_detail)
    denoised = pywt.waverec(denoised_coeffs, wavelet)
    return denoised[:len(series)]

def smape(actual, predicted):
    """Compute Symmetric Mean Absolute Percentage Error."""
    denominator = np.abs(actual) + np.abs(predicted)
    denominator = np.where(denominator == 0, 1e-8, denominator)  # avoid div by zero
    return np.mean(200.0 * np.abs(predicted - actual) / denominator)

# -------------------------------
# Load and Preprocess Historical Data
# -------------------------------

try:
    hist_data = pd.read_csv("data/merged-data.csv")
    print(f"Loaded historical data with {hist_data.shape[0]} rows and {hist_data.shape[1]} columns.")
except FileNotFoundError:
    raise FileNotFoundError("Could not find 'data/merged-data.csv'. Please check the file path.")

# Rename columns for consistency
hist_data.rename(columns={
    'start date/time': 'StartDateTime',
    'day_price': 'Price',
    'grid_load': 'total-consumption',
    'day of the week': 'DayOfWeek'
}, inplace=True)

required_cols = ['StartDateTime', 'Price', 'total-consumption', 'DayOfWeek']
missing_cols = [col for col in required_cols if col not in hist_data.columns]
if missing_cols:
    raise KeyError(f"Missing required columns: {missing_cols}")

numeric_cols = [
    'Price', 'total-consumption', 'temperature_2m',
    'precipitation (mm)', 'rain (mm)', 'snowfall (cm)',
    'wind_speed_100m (km/h)', 'relative_humidity_2m (%)',
    'weather_code (wmo code)'
]
for col in numeric_cols:
    if col in hist_data.columns:
        hist_data[col] = clean_numeric_column(hist_data[col])

# Convert datetime
hist_data['StartDateTime'] = pd.to_datetime(hist_data['StartDateTime'], format='%d/%m/%Y %H:%M', errors='coerce')
hist_data = hist_data.sort_values('StartDateTime').reset_index(drop=True)
hist_data = hist_data.dropna(subset=['StartDateTime'])

# Map DayOfWeek to numeric
day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
hist_data['DayOfWeek'] = hist_data['DayOfWeek'].str.title().map(day_map)
hist_data = hist_data.dropna(subset=['DayOfWeek'])

# Create basic time features
hist_data['Year'] = hist_data['StartDateTime'].dt.year
hist_data['Month'] = hist_data['StartDateTime'].dt.month
hist_data['Day'] = hist_data['StartDateTime'].dt.day
hist_data['Hour'] = hist_data['StartDateTime'].dt.hour

# -------------------------------
# Feature Engineering (Historical)
# -------------------------------

# Create lag features for raw Price
hist_data['Lag_Price_1h'] = hist_data['Price'].shift(1)
hist_data['Lag_Price_6h'] = hist_data['Price'].shift(6)
hist_data['Lag_Price_24h'] = hist_data['Price'].shift(24)

# Rolling features for external variables
hist_data['Rolling_Temp_6h'] = hist_data['temperature_2m'].rolling(window=6, min_periods=1).mean()
hist_data['Rolling_Temp_24h'] = hist_data['temperature_2m'].rolling(window=24, min_periods=1).mean()
hist_data['Rolling_Wind_6h'] = hist_data['wind_speed_100m (km/h)'].rolling(window=6, min_periods=1).mean()
hist_data['Rolling_Wind_24h'] = hist_data['wind_speed_100m (km/h)'].rolling(window=24, min_periods=1).mean()
hist_data['Rolling_Load_6h'] = hist_data['total-consumption'].rolling(window=6, min_periods=1).mean()
hist_data['Rolling_Load_24h'] = hist_data['total-consumption'].rolling(window=24, min_periods=1).mean()

# Price volatility features
hist_data['Price_StdDev_6h'] = hist_data['Price'].rolling(window=6, min_periods=1).std()
hist_data['Price_StdDev_24h'] = hist_data['Price'].rolling(window=24, min_periods=1).std()

# -------------------------------
# Wavelet Transform on Historical Price
# -------------------------------
# Compute a denoised price signal from historical Price.
hist_data['Price_Denoised'] = wavelet_denoise(hist_data['Price'])
# Create lag features from the denoised signal.
hist_data['Lag_Price_Denoised_1h'] = hist_data['Price_Denoised'].shift(1)
hist_data['Lag_Price_Denoised_6h'] = hist_data['Price_Denoised'].shift(6)
hist_data['Lag_Price_Denoised_24h'] = hist_data['Price_Denoised'].shift(24)
# Rolling denoised features
hist_data['Rolling_Price_Denoised_6h'] = hist_data['Price_Denoised'].rolling(window=6, min_periods=1).mean()
hist_data['Rolling_Price_Denoised_24h'] = hist_data['Price_Denoised'].rolling(window=24, min_periods=1).mean()

# -------------------------------
# (Optional) Cyclical Encoding – Uncomment if desired
# -------------------------------
# hist_data['Hour_sin'] = np.sin(2 * np.pi * hist_data['Hour'] / 24)
# hist_data['Hour_cos'] = np.cos(2 * np.pi * hist_data['Hour'] / 24)
# hist_data['DayOfWeek_sin'] = np.sin(2 * np.pi * hist_data['DayOfWeek'] / 7)
# hist_data['DayOfWeek_cos'] = np.cos(2 * np.pi * hist_data['DayOfWeek'] / 7)

# -------------------------------
# Impute any missing values
# -------------------------------
for col in hist_data.columns:
    if hist_data[col].dtype in [np.float64, np.int64]:
        hist_data[col] = hist_data[col].interpolate(method='linear', limit_direction='both')
        hist_data[col] = hist_data[col].fillna(hist_data[col].mean())

# Define features and target.
# You can choose to include cyclical features if you uncomment them.
features = [
    'temperature_2m', 'precipitation (mm)', 'rain (mm)', 'snowfall (cm)', 'weather_code (wmo code)',
    'wind_speed_100m (km/h)', 'relative_humidity_2m (%)', 'total-consumption',
    'Lag_Price_1h', 'Lag_Price_6h', 'Lag_Price_24h',
    'Rolling_Temp_6h', 'Rolling_Temp_24h', 'Rolling_Wind_6h', 'Rolling_Wind_24h',
    'Rolling_Load_6h', 'Rolling_Load_24h',
    'Price_StdDev_6h', 'Price_StdDev_24h',
    'Lag_Price_Denoised_1h', 'Lag_Price_Denoised_6h', 'Lag_Price_Denoised_24h',
    'Rolling_Price_Denoised_6h', 'Rolling_Price_Denoised_24h'
]
target = 'Price'
hist_data.dropna(subset=features + [target], inplace=True)

# Prepare modeling data.
X = hist_data[features]
y = hist_data[target]
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
print(f"Scaled features shape: {X_scaled.shape}")

# -------------------------------
# Model Training with XGBoost & Optuna Tuning
# -------------------------------
tscv = TimeSeriesSplit(n_splits=10)

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 20),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 20),
        'random_state': 42,
        'eval_metric': 'rmse',
        'early_stopping_rounds': 50
    }
    scores = []
    for train_index, val_index in tscv.split(X_scaled):
        X_train, X_val = X_scaled[train_index], X_scaled[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = model.predict(X_val)
        scores.append(mean_squared_error(y_val, y_pred))
    return np.mean(scores)

study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=100)
best_params = study.best_params
print(f"Best Parameters: {best_params}")

best_xgb = xgb.XGBRegressor(**best_params, random_state=42)
best_xgb.fit(X_scaled, y)

# Evaluate on the last fold of the historical data.
train_index, test_index = list(tscv.split(X_scaled))[-1]
X_train, X_test = X_scaled[train_index], X_scaled[test_index]
y_train, y_test = y.iloc[train_index], y.iloc[test_index]
y_pred_hist = best_xgb.predict(X_test)

rmse_hist = np.sqrt(mean_squared_error(y_test, y_pred_hist))
mae_hist = mean_absolute_error(y_test, y_pred_hist)
r2_hist = r2_score(y_test, y_pred_hist)
mape_hist = mean_absolute_percentage_error(y_test, y_pred_hist) * 100

print("\n=== Historical Evaluation ===")
print(f"RMSE: {rmse_hist}")
print(f"MAE: {mae_hist}")
print(f"R²: {r2_hist}")
print(f"MAPE: {mape_hist}%")

# -------------------------------
# Future Forecasting (Recursive Prediction)
# -------------------------------
# Load future data.
future_df = pd.read_csv('data/future-data.csv')
future_df.rename(columns={
    'start date/time': 'StartDateTime',
    'grid_load': 'total-consumption',
    'day of the week': 'DayOfWeek'
}, inplace=True)
for col in numeric_cols:
    if col in future_df.columns and col != 'Price':
        future_df[col] = clean_numeric_column(future_df[col])
future_df['StartDateTime'] = pd.to_datetime(future_df['StartDateTime'], format='%d/%m/%Y %H:%M', errors='coerce')
future_df['DayOfWeek'] = future_df['DayOfWeek'].str.title().map(day_map)
future_df.dropna(subset=['StartDateTime', 'DayOfWeek'], inplace=True)
future_df = future_df.sort_values('StartDateTime').reset_index(drop=True)

# Focus on forecast window.
forecast_start_dt = pd.to_datetime('06/01/2025 00:00:00', format='%d/%m/%Y %H:%M:%S')
forecast_end_dt = pd.to_datetime('12/01/2025 23:00:00', format='%d/%m/%Y %H:%M:%S')
future_df = future_df[(future_df['StartDateTime'] >= forecast_start_dt) &
                      (future_df['StartDateTime'] <= forecast_end_dt)].copy()

# Impute missing values in future_df.
for col in future_df.columns:
    if future_df[col].dtype in [np.float64, np.int64]:
        future_df[col] = future_df[col].interpolate(method='linear', limit_direction='both')
        future_df[col] = future_df[col].fillna(future_df[col].mean())
future_df.reset_index(drop=True, inplace=True)

# --- Recursive Forecasting Setup ---
# We'll initialize raw history with last 48 historical raw prices and similarly for wavelet (denoised) history.
last_hist = hist_data.tail(48)
raw_history = last_hist['Price'].tolist()
# For the denoised history, we use the historical Price_Denoised.
denoised_history = last_hist['Price_Denoised'].tolist()

predictions = []  # To store future predicted raw prices

# For each future time step, build feature vector using:
# - Raw lag features from raw_history (updated recursively)
# - For wavelet features, we approximate the denoised value by a simple 6-hour moving average of raw_history.
for idx, row in future_df.iterrows():
    dt = row['StartDateTime']
    temperature = float(row['temperature_2m'])
    precipitation = float(row['precipitation (mm)'])
    rain = float(row['rain (mm)'])
    snowfall = float(row['snowfall (cm)'])
    weather_code = float(row['weather_code (wmo code)'])
    wind_speed = float(row['wind_speed_100m (km/h)'])
    rel_humidity = float(row['relative_humidity_2m (%)'])
    total_consumption = float(row['total-consumption'])
    
    # Time features (from future data)
    Day = dt.day
    Hour = dt.hour
    DayOfWeek = dt.dayofweek
    
    # Lag features for raw Price.
    Lag_Price_1h = predictions[-1] if predictions else raw_history[-1]
    Lag_Price_6h = predictions[-6] if len(predictions) >= 6 else raw_history[-6]
    Lag_Price_24h = predictions[-24] if len(predictions) >= 24 else raw_history[-24]
    
    # For wavelet-denoised lags, approximate current denoised value as 6-hr moving average of raw_history.
    if len(raw_history) >= 6:
        current_denoised = np.mean(raw_history[-6:])
    else:
        current_denoised = raw_history[-1]
    # We mimic lags for denoised features similarly.
    Lag_Price_Denoised_1h = denoised_history[-1]
    Lag_Price_Denoised_6h = denoised_history[-6] if len(denoised_history) >= 6 else denoised_history[-1]
    Lag_Price_Denoised_24h = denoised_history[-24] if len(denoised_history) >= 24 else denoised_history[-1]
    
    # Rolling features for denoised: use moving average over the denoised_history.
    Rolling_Price_Denoised_6h = np.mean(denoised_history[-6:]) if len(denoised_history) >= 6 else np.mean(denoised_history)
    Rolling_Price_Denoised_24h = np.mean(denoised_history[-24:]) if len(denoised_history) >= 24 else np.mean(denoised_history)
    
    # Use same rolling features for external variables as in historical data.
    Rolling_Temp_6h = np.mean([temperature]*6)
    Rolling_Temp_24h = np.mean([temperature]*24)
    Rolling_Wind_6h = np.mean([wind_speed]*6)
    Rolling_Wind_24h = np.mean([wind_speed]*24)
    Rolling_Load_6h = np.mean([total_consumption]*6)
    Rolling_Load_24h = np.mean([total_consumption]*24)
    
    # Price volatility from raw history.
    Price_StdDev_6h = np.std(raw_history[-6:]) if len(raw_history) >= 6 else np.std(raw_history)
    Price_StdDev_24h = np.std(raw_history[-24:]) if len(raw_history) >= 24 else np.std(raw_history)
    
    # Assemble feature vector (same order as training features).
    feature_vector = [
        temperature, precipitation, rain, snowfall, weather_code,
        wind_speed, rel_humidity, total_consumption,
        Lag_Price_1h, Lag_Price_6h, Lag_Price_24h,
        Rolling_Temp_6h, Rolling_Temp_24h, Rolling_Wind_6h, Rolling_Wind_24h,
        Rolling_Load_6h, Rolling_Load_24h, Price_StdDev_6h, Price_StdDev_24h,
        Lag_Price_Denoised_1h, Lag_Price_Denoised_6h, Lag_Price_Denoised_24h,
        Rolling_Price_Denoised_6h, Rolling_Price_Denoised_24h
    ]
    
    # Scale and predict.
    feature_vector_scaled = scaler.transform([feature_vector])
    pred_price = best_xgb.predict(feature_vector_scaled)[0]
    predictions.append(pred_price)
    
    # Update raw history.
    raw_history.append(pred_price)
    # Update denoised history with the approximated current denoised value.
    # (Here we use the 6-hr moving average of raw_history as an approximation.)
    denoised_history.append(np.mean(raw_history[-6:]))
    
# Attach predictions to future_df.
future_df['Predicted Price [Euro/MWh]'] = predictions

# -------------------------------
# Future Metrics & Plotting
# -------------------------------
# Suppose you have actual future prices available.
actual_data = [
    {"start date/time": "2025-01-06 00:00:00", "actual_price": 27.52},
    {"start date/time": "2025-01-06 01:00:00", "actual_price": 19.26},
    {"start date/time": "2025-01-06 02:00:00", "actual_price": 11.35},
    {"start date/time": "2025-01-06 03:00:00", "actual_price": 9.20},
    {"start date/time": "2025-01-06 04:00:00", "actual_price": 10.00},
    {"start date/time": "2025-01-06 05:00:00", "actual_price": 14.81},
    {"start date/time": "2025-01-06 06:00:00", "actual_price": 23.82},
    {"start date/time": "2025-01-06 07:00:00", "actual_price": 31.72},
    {"start date/time": "2025-01-06 08:00:00", "actual_price": 41.37},
    {"start date/time": "2025-01-06 09:00:00", "actual_price": 36.36},
    {"start date/time": "2025-01-06 10:00:00", "actual_price": 34.69},
    {"start date/time": "2025-01-06 11:00:00", "actual_price": 32.83},
    {"start date/time": "2025-01-06 12:00:00", "actual_price": 31.28},
    {"start date/time": "2025-01-06 13:00:00", "actual_price": 28.60},
    {"start date/time": "2025-01-06 14:00:00", "actual_price": 25.61},
    {"start date/time": "2025-01-06 15:00:00", "actual_price": 26.32},
    {"start date/time": "2025-01-06 16:00:00", "actual_price": 26.87},
    {"start date/time": "2025-01-06 17:00:00", "actual_price": 31.66},
    {"start date/time": "2025-01-06 18:00:00", "actual_price": 31.58},
    {"start date/time": "2025-01-06 19:00:00", "actual_price": 28.74},
    {"start date/time": "2025-01-06 20:00:00", "actual_price": 26.59},
    {"start date/time": "2025-01-06 21:00:00", "actual_price": 13.82},
    {"start date/time": "2025-01-06 22:00:00", "actual_price": 26.94},
    {"start date/time": "2025-01-06 23:00:00", "actual_price": 12.99},
    {"start date/time": "2025-01-07 00:00:00", "actual_price": 19.07},
    {"start date/time": "2025-01-07 01:00:00", "actual_price": 8.71},
    {"start date/time": "2025-01-07 02:00:00", "actual_price": 8.90},
    {"start date/time": "2025-01-07 03:00:00", "actual_price": 5.01},
    {"start date/time": "2025-01-07 04:00:00", "actual_price": 5.13},
    {"start date/time": "2025-01-07 05:00:00", "actual_price": 5.80},
    {"start date/time": "2025-01-07 06:00:00", "actual_price": 48.86},
    {"start date/time": "2025-01-07 07:00:00", "actual_price": 76.83},
    {"start date/time": "2025-01-07 08:00:00", "actual_price": 85.08},
    {"start date/time": "2025-01-07 09:00:00", "actual_price": 84.24},
    {"start date/time": "2025-01-07 10:00:00", "actual_price": 75.25},
    {"start date/time": "2025-01-07 11:00:00", "actual_price": 62.80},
    {"start date/time": "2025-01-07 12:00:00", "actual_price": 62.44},
    {"start date/time": "2025-01-07 13:00:00", "actual_price": 63.90},
    {"start date/time": "2025-01-07 14:00:00", "actual_price": 72.56},
    {"start date/time": "2025-01-07 15:00:00", "actual_price": 78.11},
    {"start date/time": "2025-01-07 16:00:00", "actual_price": 79.98},
    {"start date/time": "2025-01-07 17:00:00", "actual_price": 96.03},
    {"start date/time": "2025-01-07 18:00:00", "actual_price": 101.11},
    {"start date/time": "2025-01-07 19:00:00", "actual_price": 86.21},
    {"start date/time": "2025-01-07 20:00:00", "actual_price": 78.01},
    {"start date/time": "2025-01-07 21:00:00", "actual_price": 72.45},
    {"start date/time": "2025-01-07 22:00:00", "actual_price": 72.45},
    {"start date/time": "2025-01-07 23:00:00", "actual_price": 50.04},
    {"start date/time": "2025-01-08 00:00:00", "actual_price": 71.05},
    {"start date/time": "2025-01-08 01:00:00", "actual_price": 68.01},
    {"start date/time": "2025-01-08 02:00:00", "actual_price": 63.34},
    {"start date/time": "2025-01-08 03:00:00", "actual_price": 57.01},
    {"start date/time": "2025-01-08 04:00:00", "actual_price": 66.29},
    {"start date/time": "2025-01-08 05:00:00", "actual_price": 72.07},
    {"start date/time": "2025-01-08 06:00:00", "actual_price": 82.70},
    {"start date/time": "2025-01-08 07:00:00", "actual_price": 100.73},
    {"start date/time": "2025-01-08 08:00:00", "actual_price": 128.22},
    {"start date/time": "2025-01-08 09:00:00", "actual_price": 108.18},
    {"start date/time": "2025-01-08 10:00:00", "actual_price": 94.65},
    {"start date/time": "2025-01-08 11:00:00", "actual_price": 100.01},
    {"start date/time": "2025-01-08 12:00:00", "actual_price": 89.99},
    {"start date/time": "2025-01-08 13:00:00", "actual_price": 97.20},
    {"start date/time": "2025-01-08 14:00:00", "actual_price": 110.19},
    {"start date/time": "2025-01-08 15:00:00", "actual_price": 127.80},
    {"start date/time": "2025-01-08 16:00:00", "actual_price": 135.82},
    {"start date/time": "2025-01-08 17:00:00", "actual_price": 155.46},
    {"start date/time": "2025-01-08 18:00:00", "actual_price": 149.23},
    {"start date/time": "2025-01-08 19:00:00", "actual_price": 146.48},
    {"start date/time": "2025-01-08 20:00:00", "actual_price": 136.86},
    {"start date/time": "2025-01-08 21:00:00", "actual_price": 127.86},
    {"start date/time": "2025-01-08 22:00:00", "actual_price": 115.92},
    {"start date/time": "2025-01-08 23:00:00", "actual_price": 103.59},
    {"start date/time": "2025-01-09 00:00:00", "actual_price": 101.44},
    {"start date/time": "2025-01-09 01:00:00", "actual_price": 100.00},
    {"start date/time": "2025-01-09 02:00:00", "actual_price": 98.77},
    {"start date/time": "2025-01-09 03:00:00", "actual_price": 95.22},
    {"start date/time": "2025-01-09 04:00:00", "actual_price": 98.28},
    {"start date/time": "2025-01-09 05:00:00", "actual_price": 102.65},
    {"start date/time": "2025-01-09 06:00:00", "actual_price": 133.54},
    {"start date/time": "2025-01-09 07:00:00", "actual_price": 148.80},
    {"start date/time": "2025-01-09 08:00:00", "actual_price": 164.88},
    {"start date/time": "2025-01-09 09:00:00", "actual_price": 156.15},
    {"start date/time": "2025-01-09 10:00:00", "actual_price": 147.64},
    {"start date/time": "2025-01-09 11:00:00", "actual_price": 140.21},
    {"start date/time": "2025-01-09 12:00:00", "actual_price": 128.18},
    {"start date/time": "2025-01-09 13:00:00", "actual_price": 121.15},
    {"start date/time": "2025-01-09 14:00:00", "actual_price": 123.95},
    {"start date/time": "2025-01-09 15:00:00", "actual_price": 127.53},
    {"start date/time": "2025-01-09 16:00:00", "actual_price": 123.75},
    {"start date/time": "2025-01-09 17:00:00", "actual_price": 130.91},
    {"start date/time": "2025-01-09 18:00:00", "actual_price": 134.75},
    {"start date/time": "2025-01-09 19:00:00", "actual_price": 125.44},
    {"start date/time": "2025-01-09 20:00:00", "actual_price": 119.23},
    {"start date/time": "2025-01-09 21:00:00", "actual_price": 104.99},
    {"start date/time": "2025-01-09 22:00:00", "actual_price": 101.10},
    {"start date/time": "2025-01-09 23:00:00", "actual_price": 88.19},
    {"start date/time": "2025-01-10 00:00:00", "actual_price": 84.79},
    {"start date/time": "2025-01-10 01:00:00", "actual_price": 80.23},
    {"start date/time": "2025-01-10 02:00:00", "actual_price": 71.29},
    {"start date/time": "2025-01-10 03:00:00", "actual_price": 69.05},
    {"start date/time": "2025-01-10 04:00:00", "actual_price": 69.89},
    {"start date/time": "2025-01-10 05:00:00", "actual_price": 83.90},
    {"start date/time": "2025-01-10 06:00:00", "actual_price": 99.09},
    {"start date/time": "2025-01-10 07:00:00", "actual_price": 123.92},
    {"start date/time": "2025-01-10 08:00:00", "actual_price": 139.47},
    {"start date/time": "2025-01-10 09:00:00", "actual_price": 136.92},
    {"start date/time": "2025-01-10 10:00:00", "actual_price": 123.57},
    {"start date/time": "2025-01-10 11:00:00", "actual_price": 113.59},
    {"start date/time": "2025-01-10 12:00:00", "actual_price": 107.43},
    {"start date/time": "2025-01-10 13:00:00", "actual_price": 105.01},
    {"start date/time": "2025-01-10 14:00:00", "actual_price": 110.01},
    {"start date/time": "2025-01-10 15:00:00", "actual_price": 128.69},
    {"start date/time": "2025-01-10 16:00:00", "actual_price": 134.87},
    {"start date/time": "2025-01-10 17:00:00", "actual_price": 142.56},
    {"start date/time": "2025-01-10 18:00:00", "actual_price": 144.12},
    {"start date/time": "2025-01-10 19:00:00", "actual_price": 141.05},
    {"start date/time": "2025-01-10 20:00:00", "actual_price": 134.82},
    {"start date/time": "2025-01-10 21:00:00", "actual_price": 122.51},
    {"start date/time": "2025-01-10 22:00:00", "actual_price": 119.77},
    {"start date/time": "2025-01-10 23:00:00", "actual_price": 111.71},
    {"start date/time": "2025-01-11 00:00:00", "actual_price": 106.64},
    {"start date/time": "2025-01-11 01:00:00", "actual_price": 98.99},
    {"start date/time": "2025-01-11 02:00:00", "actual_price": 95.71},
    {"start date/time": "2025-01-11 03:00:00", "actual_price": 89.45},
    {"start date/time": "2025-01-11 04:00:00", "actual_price": 88.35},
    {"start date/time": "2025-01-11 05:00:00", "actual_price": 88.40},
    {"start date/time": "2025-01-11 06:00:00", "actual_price": 88.13},
    {"start date/time": "2025-01-11 07:00:00", "actual_price": 94.68},
    {"start date/time": "2025-01-11 08:00:00", "actual_price": 107.65},
    {"start date/time": "2025-01-11 09:00:00", "actual_price": 103.14},
    {"start date/time": "2025-01-11 10:00:00", "actual_price": 101.13},
    {"start date/time": "2025-01-11 11:00:00", "actual_price": 99.56},
    {"start date/time": "2025-01-11 12:00:00", "actual_price": 96.74},
    {"start date/time": "2025-01-11 13:00:00", "actual_price": 93.15},
    {"start date/time": "2025-01-11 14:00:00", "actual_price": 94.90},
    {"start date/time": "2025-01-11 15:00:00", "actual_price": 104.89},
    {"start date/time": "2025-01-11 16:00:00", "actual_price": 112.12},
    {"start date/time": "2025-01-11 17:00:00", "actual_price": 119.80},
    {"start date/time": "2025-01-11 18:00:00", "actual_price": 122.34},
    {"start date/time": "2025-01-11 19:00:00", "actual_price": 114.85},
    {"start date/time": "2025-01-11 20:00:00", "actual_price": 111.19},
    {"start date/time": "2025-01-11 21:00:00", "actual_price": 104.80},
    {"start date/time": "2025-01-11 22:00:00", "actual_price": 103.07},
    {"start date/time": "2025-01-11 23:00:00", "actual_price": 105.69},
    {"start date/time": "2025-01-12 00:00:00", "actual_price": 83.58},
]

actual_df = pd.DataFrame(actual_data)
actual_df['start date/time'] = pd.to_datetime(actual_df['start date/time'])

merged_df = pd.merge(
    future_df[['StartDateTime', 'Predicted Price [Euro/MWh]']],
    actual_df, left_on='StartDateTime', right_on='start date/time', how='inner'
)

plt.figure(figsize=(14,7))
plt.plot(merged_df['StartDateTime'], merged_df['Predicted Price [Euro/MWh]'], label='Predicted Price', color='blue', marker='o')
plt.plot(merged_df['StartDateTime'], merged_df['actual_price'], label='Actual Price', color='red', marker='x')
plt.xlabel('Date and Time')
plt.ylabel('Price [Euro/MWh]')
plt.title('Predicted vs Actual Electricity Prices (06-Jan-2025 to 12-Jan-2025)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

if not merged_df.empty:
    y_true_future = merged_df['actual_price'].values
    y_pred_future = merged_df['Predicted Price [Euro/MWh]'].values
    future_rmse = np.sqrt(mean_squared_error(y_true_future, y_pred_future))
    future_mae = mean_absolute_error(y_true_future, y_pred_future)
    future_r2 = r2_score(y_true_future, y_pred_future)
    future_mape = mean_absolute_percentage_error(y_true_future, y_pred_future) * 100
    print("\n=== Future Prediction Metrics ===")
    print(f"RMSE: {future_rmse}")
    print(f"MAE: {future_mae}")
    print(f"R²: {future_r2}")
    print(f"MAPE: {future_mape}%")
else:
    print("\nNo overlapping timestamps in future and actual data to evaluate future metrics.")