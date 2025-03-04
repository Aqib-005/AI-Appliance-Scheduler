import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Function to clean numeric columns
def clean_numeric_column(series):
    """Clean numeric columns by replacing commas with dots, removing spaces, and converting to numeric."""
    return pd.to_numeric(
        series.astype(str)
            .str.replace(',', '.')  # Replace comma with dot
            .str.replace(' ', '')   # Remove spaces
            .str.replace('–', '-')  # Replace en dash with minus
            .str.replace(',', ''),  # Remove any remaining commas
        errors='coerce'
    )

# Load historical data
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

# Verify required columns exist
required_cols = ['StartDateTime', 'Price', 'total-consumption', 'DayOfWeek']
missing_cols = [col for col in required_cols if col not in hist_data.columns]
if missing_cols:
    raise KeyError(f"Missing required columns in historical data: {missing_cols}")

# Columns to clean and convert to numeric
numeric_cols = [
    'Price', 'total-consumption', 'temperature_2m',
    'precipitation (mm)', 'rain (mm)', 'snowfall (cm)',
    'wind_speed_100m (km/h)', 'relative_humidity_2m (%)',
    'weather_code (wmo code)'
]

# Clean numeric columns
for col in numeric_cols:
    if col in hist_data.columns:
        hist_data[col] = clean_numeric_column(hist_data[col])

# Convert datetime with specific format
hist_data['StartDateTime'] = pd.to_datetime(
    hist_data['StartDateTime'], 
    format='%d/%m/%Y %H:%M', 
    errors='coerce'
)
hist_data = hist_data.sort_values('StartDateTime').reset_index(drop=True)

# Check for NaT in 'StartDateTime' and drop those rows
print(f"Number of NaT in 'StartDateTime': {hist_data['StartDateTime'].isna().sum()}")
hist_data = hist_data.dropna(subset=['StartDateTime'])
print(f"Rows after dropping NaT 'StartDateTime': {hist_data.shape[0]}")

# Map DayOfWeek to numeric, standardizing case
day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
hist_data['DayOfWeek'] = hist_data['DayOfWeek'].str.title().map(day_map)

# Check for NaN in 'DayOfWeek' after mapping and drop those rows
print(f"Number of NaN in 'DayOfWeek' after mapping: {hist_data['DayOfWeek'].isna().sum()}")
hist_data = hist_data.dropna(subset=['DayOfWeek'])
print(f"Rows after dropping unmapped 'DayOfWeek': {hist_data.shape[0]}")

# Create time features
hist_data['Year'] = hist_data['StartDateTime'].dt.year
hist_data['Month'] = hist_data['StartDateTime'].dt.month
hist_data['Day'] = hist_data['StartDateTime'].dt.day
hist_data['Hour'] = hist_data['StartDateTime'].dt.hour

# Feature engineering
hist_data['Lag_Price_1h'] = hist_data['Price'].shift(1)
hist_data['Lag_Price_6h'] = hist_data['Price'].shift(6)
hist_data['Lag_Price_24h'] = hist_data['Price'].shift(24)

hist_data['Rolling_Temp_6h'] = hist_data['temperature_2m'].rolling(window=6, min_periods=1).mean()
hist_data['Rolling_Temp_24h'] = hist_data['temperature_2m'].rolling(window=24, min_periods=1).mean()
hist_data['Rolling_Wind_6h'] = hist_data['wind_speed_100m (km/h)'].rolling(window=6, min_periods=1).mean()
hist_data['Rolling_Wind_24h'] = hist_data['wind_speed_100m (km/h)'].rolling(window=24, min_periods=1).mean()
hist_data['Rolling_Load_6h'] = hist_data['total-consumption'].rolling(window=6, min_periods=1).mean()
hist_data['Rolling_Load_24h'] = hist_data['total-consumption'].rolling(window=24, min_periods=1).mean()

hist_data['Price_StdDev_6h'] = hist_data['Price'].rolling(window=6, min_periods=1).std()
hist_data['Price_StdDev_24h'] = hist_data['Price'].rolling(window=24, min_periods=1).std()

# Handle missing values with interpolation and mean fallback for numeric columns
for col in hist_data.columns:
    if hist_data[col].dtype in [np.float64, np.int64]:
        hist_data[col] = hist_data[col].interpolate(method='linear', limit_direction='both')
        mean_val = hist_data[col].mean()
        hist_data[col] = hist_data[col].fillna(mean_val if not pd.isna(mean_val) else 0)

print(f"Rows after feature engineering and imputation: {hist_data.shape[0]}")

# Define features and target
features = [
    'temperature_2m', 'precipitation (mm)', 'rain (mm)', 
    'snowfall (cm)', 'weather_code (wmo code)', 
    'wind_speed_100m (km/h)', 'relative_humidity_2m (%)', 
    'total-consumption', 'Day', 'Hour', 'DayOfWeek',
    'Lag_Price_1h', 'Lag_Price_6h', 'Lag_Price_24h',
    'Rolling_Temp_6h', 'Rolling_Temp_24h',
    'Rolling_Wind_6h', 'Rolling_Wind_24h',
    'Rolling_Load_6h', 'Rolling_Load_24h',
    'Price_StdDev_6h', 'Price_StdDev_24h'
]
target = 'Price'

# Drop rows with NaNs only in the features and target columns
hist_data.dropna(subset=features + [target], inplace=True)
print(f"Rows after dropping NaNs in features and target: {hist_data.shape[0]}")
if hist_data.shape[0] == 0:
    raise ValueError("DataFrame is empty after dropping NaNs in features and target.")

# Prepare data for modeling
X = hist_data[features]
y = hist_data[target]

# Robust scaling
scaler_X = RobustScaler()
scaler_y = RobustScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

# Sequence preparation for LSTM
def create_sequences(X, y, time_steps=24):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i+time_steps])
        y_seq.append(y[i+time_steps])
    return np.array(X_seq), np.array(y_seq)

# Create sequences
time_steps = 24
X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)

# Split data for training and testing
def split_sequences(X_seq, y_seq):
    # Use last 20% for testing
    split_idx = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_sequences(X_seq, y_seq)

# Build LSTM Model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Input shape: (time_steps, features)
input_shape = (X_train.shape[1], X_train.shape[2])
model = build_lstm_model(input_shape)

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=5, 
    min_lr=0.00001
)

# Train the model
history = model.fit(
    X_train, y_train, 
    epochs=100, 
    batch_size=32, 
    validation_split=0.2, 
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Predict on test set
y_pred_scaled = model.predict(X_test).flatten()
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Evaluation metrics
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred) * 100

print("Evaluation Metrics on Historical Data:")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R²: {r2}")
print(f"MAPE: {mape}%")

# Prepare future forecasting
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
future_df.sort_values('StartDateTime', inplace=True)

forecast_start_dt = pd.to_datetime('06/01/2025 00:00:00', format='%d/%m/%Y %H:%M:%S')
forecast_end_dt = pd.to_datetime('12/01/2025 23:00:00', format='%d/%m/%Y %H:%M:%S')
future_df = future_df[(future_df['StartDateTime'] >= forecast_start_dt) & 
                      (future_df['StartDateTime'] <= forecast_end_dt)].copy()

# Imputation for future data
for col in future_df.columns:
    if future_df[col].dtype in [np.float64, np.int64]:
        future_df[col] = future_df[col].interpolate(method='linear', limit_direction='both')
        mean_val = future_df[col].mean()
        future_df[col] = future_df[col].fillna(mean_val if not pd.isna(mean_val) else 0)

future_df.reset_index(drop=True, inplace=True)

# Recursive prediction
predictions = []
last_historical_data = hist_data.tail(time_steps)

# Initialize histories with historical data
temp_history_6h = last_historical_data['temperature_2m'].tolist()
temp_history_24h = last_historical_data['temperature_2m'].tolist()
wind_history_6h = last_historical_data['wind_speed_100m (km/h)'].tolist()
wind_history_24h = last_historical_data['wind_speed_100m (km/h)'].tolist()
load_history_6h = last_historical_data['total-consumption'].tolist()
load_history_24h = last_historical_data['total-consumption'].tolist()
price_history_1h = last_historical_data['Price'].tolist()
price_history_6h = last_historical_data['Price'].tolist()
price_history_24h = last_historical_data['Price'].tolist()

# Sequences to track for LSTM prediction
recent_sequence = last_historical_data[features].values

# Scale the recent sequence
recent_sequence_scaled = scaler_X.transform(recent_sequence)

for idx, row in future_df.iterrows():
    # Extract current row features
    dt = row['StartDateTime']
    temperature = float(row['temperature_2m'])
    precipitation = float(row['precipitation (mm)'])
    rain = float(row['rain (mm)'])
    snowfall = float(row['snowfall (cm)'])
    weather_code = float(row['weather_code (wmo code)'])
    wind_speed = float(row['wind_speed_100m (km/h)'])
    rel_humidity = float(row['relative_humidity_2m (%)'])
    total_consumption = float(row['total-consumption'])
    
    # Time features
    Day = dt.day
    Hour = dt.hour
    DayOfWeek = dt.dayofweek

    # Update recent sequence with new data
    new_features = np.array([
        temperature, precipitation, rain, snowfall, weather_code,
        wind_speed, rel_humidity, total_consumption, 
        Day, Hour, DayOfWeek,
        price_history_1h[-1], price_history_6h[-1], price_history_24h[-1],
        np.mean(temp_history_6h),
        np.mean(temp_history_6h),
        np.mean(temp_history_24h),
        np.mean(wind_history_6h),
        np.mean(wind_history_24h),
        np.mean(load_history_6h),
        np.mean(load_history_24h),
        np.std(price_history_6h),
        np.std(price_history_24h)
    ])

    # Scale new features
    new_features_scaled = scaler_X.transform(new_features.reshape(1, -1))

    # Prepare input sequence for LSTM prediction
    input_sequence_scaled = np.roll(recent_sequence_scaled, -1, axis=0)
    input_sequence_scaled[-1] = new_features_scaled

    # Predict next price
    pred_scaled = model.predict(input_sequence_scaled.reshape(1, time_steps, input_sequence_scaled.shape[1])).flatten()
    pred_price = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
    
    # Store prediction
    predictions.append(pred_price)

    # Update histories
    temp_history_6h.append(temperature)
    temp_history_24h.append(temperature)
    wind_history_6h.append(wind_speed)
    wind_history_24h.append(wind_speed)
    load_history_6h.append(total_consumption)
    load_history_24h.append(total_consumption)
    
    # Truncate histories
    temp_history_6h = temp_history_6h[-6:]
    temp_history_24h = temp_history_24h[-24:]
    wind_history_6h = wind_history_6h[-6:]
    wind_history_24h = wind_history_24h[-24:]
    load_history_6h = load_history_6h[-6:]
    load_history_24h = load_history_24h[-24:]
    
    # Update price histories
    price_history_1h.append(pred_price)
    price_history_6h.append(pred_price)
    price_history_24h.append(pred_price)
    
    price_history_1h = price_history_1h[-6:]
    price_history_6h = price_history_6h[-24:]
    price_history_24h = price_history_24h[-24:]

# Append predictions to future_df
future_df['Predicted Price [Euro/MWh]'] = predictions

# Actual data for validation (same as previous script)
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

# Convert actual data to DataFrame
actual_df = pd.DataFrame(actual_data)
actual_df['start date/time'] = pd.to_datetime(actual_df['start date/time'])

# Merge predicted and actual data
merged_df = pd.merge(future_df[['StartDateTime', 'Predicted Price [Euro/MWh]']], 
                     actual_df, 
                     left_on='StartDateTime', 
                     right_on='start date/time', 
                     how='inner')

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(merged_df['StartDateTime'], merged_df['Predicted Price [Euro/MWh]'], 
         label='Predicted Price', color='blue', marker='o')
plt.plot(merged_df['StartDateTime'], merged_df['actual_price'], 
         label='Actual Price', color='red', marker='x')
plt.xlabel('Date and Time')
plt.ylabel('Price [Euro/MWh]')
plt.title('LSTM: Predicted vs Actual Electricity Prices (06-Jan-2025 to 12-Jan-2025)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Compute comparison metrics
pred_prices = merged_df['Predicted Price [Euro/MWh]']
actual_prices = merged_df['actual_price']

lstm_rmse = np.sqrt(mean_squared_error(actual_prices, pred_prices))
lstm_mae = mean_absolute_error(actual_prices, pred_prices)
lstm_r2 = r2_score(actual_prices, pred_prices)
lstm_mape = mean_absolute_percentage_error(actual_prices, pred_prices) * 100

print("\nLSTM Prediction Performance:")
print(f"RMSE: {lstm_rmse:.2f}")
print(f"MAE: {lstm_mae:.2f}")
print(f"R²: {lstm_r2:.4f}")
print(f"MAPE: {lstm_mape:.2f}%")

# Display predictions
print("\nPredictions for 06-Jan-2025 to 12-Jan-2025:")
print(future_df[['StartDateTime', 'Predicted Price [Euro/MWh]']].to_string(index=False))
