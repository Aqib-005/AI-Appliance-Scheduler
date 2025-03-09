import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
from xgboost import XGBRegressor

# Define helper functions
def clean_numeric_column(series):
    """Convert a series to numeric, handling commas, spaces, and dashes."""
    return pd.to_numeric(
        series.astype(str).str.replace(',', '.').str.replace(' ', '').str.replace('–', '-'),
        errors='coerce'
    )

def safe_mape(actual, predicted):
    """Calculate MAPE, avoiding division by zero and handling small values."""
    mask = np.abs(actual) > 1e-8  # Avoid division by very small values
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

# Load and Preprocess Historical Data
hist_data = pd.read_csv("data/merged-data.csv")
hist_data.columns = hist_data.columns.str.strip()
hist_data.rename(columns={
    'start date/time': 'StartDateTime',
    'day_price': 'Price',
    'grid_load': 'total-consumption',
    'day of the week': 'DayOfWeek'
}, inplace=True)

# Convert numeric columns
numeric_cols = [
    'Price', 'total-consumption', 'temperature_2m', 'precipitation (mm)', 'rain (mm)',
    'snowfall (cm)', 'wind_speed_100m (km/h)', 'relative_humidity_2m (%)', 'weather_code (wmo code)'
]
for col in numeric_cols:
    if col in hist_data.columns:
        hist_data[col] = clean_numeric_column(hist_data[col])

# Convert datetime and sort
hist_data['StartDateTime'] = pd.to_datetime(hist_data['StartDateTime'], format='%d/%m/%Y %H:%M', errors='coerce')
hist_data = hist_data.sort_values('StartDateTime').dropna(subset=['StartDateTime']).reset_index(drop=True)

# Map day of week
day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
hist_data['DayOfWeek'] = hist_data['DayOfWeek'].str.title().map(day_map)
hist_data = hist_data.dropna(subset=['DayOfWeek'])

# Feature Engineering
hist_data['Hour'] = hist_data['StartDateTime'].dt.hour
hist_data['Day'] = hist_data['StartDateTime'].dt.day
hist_data['Month'] = hist_data['StartDateTime'].dt.month
hist_data['IsWeekend'] = hist_data['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
hist_data['Price_Lag1'] = hist_data['Price'].shift(1)
hist_data['Price_Lag24'] = hist_data['Price'].shift(24)
hist_data['Price_RollingMean24'] = hist_data['Price'].rolling(window=24, min_periods=1).mean()
hist_data['Price_RollingStd24'] = hist_data['Price'].rolling(window=24, min_periods=1).std()
hist_data['Temp_Wind_Interaction'] = hist_data['temperature_2m'] * hist_data['wind_speed_100m (km/h)']
hist_data['Hourly_Volatility'] = hist_data['Price'].rolling(window=24, min_periods=1).std().shift(1)

# Handle missing values and outliers
for col in hist_data.columns:
    if hist_data[col].dtype in [np.float64, np.int64]:
        hist_data[col] = hist_data[col].fillna(method='ffill').fillna(method='bfill')
        Q1 = hist_data[col].quantile(0.25)
        Q3 = hist_data[col].quantile(0.75)
        IQR = Q3 - Q1
        hist_data[col] = hist_data[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)

# Define features and target
features = [
    'total-consumption', 'temperature_2m', 'wind_speed_100m (km/h)', 'relative_humidity_2m (%)',
    'Price_Lag1', 'Price_Lag24', 'Price_RollingMean24', 'Price_RollingStd24', 'Temp_Wind_Interaction',
    'Hourly_Volatility', 'Hour', 'Day', 'Month', 'DayOfWeek', 'IsWeekend'
]
target = 'Price'
valid_features = [col for col in features if col in hist_data.columns and not hist_data[col].isna().all()]
print("Valid features:", valid_features)

# Prepare data
hist_data = hist_data.dropna(subset=valid_features + [target])
X = hist_data[valid_features]
y = hist_data[target]

# Scale features
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=valid_features)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X_scaled_df, y, test_size=0.2, shuffle=False)

# Train LightGBM with hyperparameter tuning
param_grid = {
    'num_leaves': [31, 50, 100],
    'min_data_in_leaf': [10, 20, 50],
    'max_depth': [-1, 10, 20],
    'learning_rate': [0.01, 0.05, 0.1],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [0, 0.1, 1]
}
lgbm = lgb.LGBMRegressor()
grid_search = GridSearchCV(lgbm, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)

# Validation prediction and calibration
y_pred_val = best_model.predict(X_val)
calibration_model = LinearRegression()
calibration_model.fit(y_pred_val.reshape(-1, 1), y_val)
calibrated_y_pred_val = calibration_model.predict(y_pred_val.reshape(-1, 1))

# Validation metrics
print("\n=== Calibrated Historical Evaluation (LightGBM) ===")
print(f"RMSE: {np.sqrt(mean_squared_error(y_val, calibrated_y_pred_val)):.2f}")
print(f"MAE: {mean_absolute_error(y_val, calibrated_y_pred_val):.2f}")
print(f"R²: {r2_score(y_val, calibrated_y_pred_val):.3f}")
print(f"MAPE: {safe_mape(y_val, calibrated_y_pred_val):.2f}%")

# Load and Preprocess Future Data
future_df = pd.read_csv('data/future-data.csv')
future_df.columns = future_df.columns.str.strip()
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
future_df = future_df.dropna(subset=['StartDateTime', 'DayOfWeek']).sort_values('StartDateTime').reset_index(drop=True)

# Filter future data
forecast_start_dt = pd.to_datetime('06/01/2025 00:00:00', format='%d/%m/%Y %H:%M:%S')
forecast_end_dt = pd.to_datetime('12/01/2025 00:00:00', format='%d/%m/%Y %H:%M:%S')
future_df = future_df[(future_df['StartDateTime'] >= forecast_start_dt) & (future_df['StartDateTime'] <= forecast_end_dt)]

# Add time features
future_df['Hour'] = future_df['StartDateTime'].dt.hour
future_df['Day'] = future_df['StartDateTime'].dt.day
future_df['Month'] = future_df['StartDateTime'].dt.month
future_df['IsWeekend'] = future_df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

# Warm-up period for better first-day predictions
time_steps = 24
warm_up_X = X_scaled_df.iloc[-time_steps:]
warm_up_predictions = best_model.predict(warm_up_X)
calibrated_warm_up_predictions = calibration_model.predict(warm_up_predictions.reshape(-1, 1))
past_prices = list(calibrated_warm_up_predictions)

# Prediction loop
predictions = []
for i in range(len(future_df)):
    new_row = pd.Series(index=valid_features, dtype=float)
    for feature in ['total-consumption', 'temperature_2m', 'wind_speed_100m (km/h)', 'relative_humidity_2m (%)', 'Hour', 'Day', 'Month', 'DayOfWeek', 'IsWeekend']:
        if feature in valid_features and feature in future_df.columns:
            new_row[feature] = future_df.iloc[i][feature]
    if 'Temp_Wind_Interaction' in valid_features:
        new_row['Temp_Wind_Interaction'] = future_df.iloc[i]['temperature_2m'] * future_df.iloc[i]['wind_speed_100m (km/h)']
    if 'Hourly_Volatility' in valid_features:
        new_row['Hourly_Volatility'] = np.std(past_prices[-24:]) if len(past_prices) >= 24 else 0
    if 'Price_Lag1' in valid_features:
        new_row['Price_Lag1'] = past_prices[-1]
    if 'Price_Lag24' in valid_features:
        new_row['Price_Lag24'] = past_prices[-24] if len(past_prices) >= 24 else past_prices[0]
    if 'Price_RollingMean24' in valid_features:
        new_row['Price_RollingMean24'] = np.mean(past_prices[-24:]) if len(past_prices) >= 24 else np.mean(past_prices)
    if 'Price_RollingStd24' in valid_features:
        new_row['Price_RollingStd24'] = np.std(past_prices[-24:]) if len(past_prices) >= 24 else 0

    new_row = new_row.fillna(0)
    new_row_scaled = scaler.transform(new_row.to_frame().T)
    pred_price = best_model.predict(new_row_scaled)[0]
    calibrated_pred_price = calibration_model.predict([[pred_price]])[0]
    calibrated_pred_price = np.clip(calibrated_pred_price, -150, 200)
    past_prices.append(calibrated_pred_price)
    predictions.append(calibrated_pred_price)

future_df['Predicted Price [Euro/MWh]'] = predictions

# Actual Data (Replace with your full dataset)
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

# Check actual data coverage
print("Actual data range:", actual_df['start date/time'].min(), "to", actual_df['start date/time'].max())
if actual_df['start date/time'].max() < forecast_end_dt:
    print("Warning: Actual data incomplete. Update 'actual_data' to cover up to", forecast_end_dt)

# Merge data (use 'left' to include all predictions)
merged_df = pd.merge(
    future_df[['StartDateTime', 'Predicted Price [Euro/MWh]']],
    actual_df.rename(columns={'start date/time': 'StartDateTime', 'actual_price': 'Actual Price'}),
    on='StartDateTime',
    how='left'
)

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(merged_df['StartDateTime'], merged_df['Predicted Price [Euro/MWh]'], label='Predicted Price', color='blue', marker='o')
plt.plot(merged_df['StartDateTime'], merged_df['Actual Price'], label='Actual Price', color='red', marker='x')
plt.xlabel('Date and Time')
plt.ylabel('Price [Euro/MWh]')
plt.title('Predicted vs Actual Electricity Prices (06-Jan-2025 to 12-Jan-2025)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Future Prediction Metrics
if merged_df['Actual Price'].notna().sum() > 0:
    mask = merged_df['Actual Price'].notna()
    y_true = merged_df.loc[mask, 'Actual Price'].values
    y_pred = merged_df.loc[mask, 'Predicted Price [Euro/MWh]'].values
    print("\n=== Future Prediction Metrics ===")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"R²: {r2_score(y_true, y_pred):.3f}")
    print(f"MAPE: {safe_mape(y_true, y_pred):.2f}%")
else:
    print("No actual data available for metric calculation.")

# Alternative Models (XGBoost)
xgb_model = XGBRegressor(objective='reg:squarederror')
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_val)
print("\n=== XGBoost Evaluation ===")
print(f"RMSE: {np.sqrt(mean_squared_error(y_val, y_pred_xgb)):.2f}")
print(f"MAE: {mean_absolute_error(y_val, y_pred_xgb):.2f}")
print(f"R²: {r2_score(y_val, y_pred_xgb):.3f}")
print(f"MAPE: {safe_mape(y_val, y_pred_xgb):.2f}%")

# Alternative Models (LSTM)
X_train_lstm = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_val_lstm = X_val.values.reshape((X_val.shape[0], 1, X_val.shape[1]))
lstm_model = Sequential([
    LSTM(50, activation='relu', input_shape=(1, X_train.shape[1])),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, verbose=0)
y_pred_lstm = lstm_model.predict(X_val_lstm).flatten()
print("\n=== LSTM Evaluation ===")
print(f"RMSE: {np.sqrt(mean_squared_error(y_val, y_pred_lstm)):.2f}")
print(f"MAE: {mean_absolute_error(y_val, y_pred_lstm):.2f}")
print(f"R²: {r2_score(y_val, y_pred_lstm):.3f}")
print(f"MAPE: {safe_mape(y_val, y_pred_lstm):.2f}%")