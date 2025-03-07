import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras_tuner
from keras_tuner import RandomSearch

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# For reproducibility
np.random.seed(42)
tf.random.set_seed(42)

#####################################
# 1. Utility Functions
#####################################

def clean_numeric_column(series):
    """
    Clean numeric columns by replacing commas with dots, removing spaces, 
    and converting to numeric.
    """
    return pd.to_numeric(
        series.astype(str)
              .str.replace(',', '.')
              .str.replace(' ', '')
              .str.replace('–', '-')
              .str.replace(',', ''),  # Repeated replace just to be safe
        errors='coerce'
    )

def safe_mape(y_true, y_pred, eps=1.0):
    """
    Compute a MAPE that clips actual values to avoid dividing by zero or near-zero.
    """
    y_true_clipped = np.clip(y_true, eps, None)
    return np.mean(np.abs((y_true - y_pred) / y_true_clipped)) * 100

#####################################
# 2. Load and Prepare Historical Data
#####################################

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

# Check required columns
required_cols = ['StartDateTime', 'Price', 'total-consumption', 'DayOfWeek']
missing_cols = [col for col in required_cols if col not in hist_data.columns]
if missing_cols:
    raise KeyError(f"Missing required columns in historical data: {missing_cols}")

# Clean numeric columns
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
hist_data['StartDateTime'] = pd.to_datetime(
    hist_data['StartDateTime'],
    format='%d/%m/%Y %H:%M',
    errors='coerce'
)
hist_data.sort_values('StartDateTime', inplace=True, ignore_index=True)

# Map DayOfWeek to numeric
day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
           'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
hist_data['DayOfWeek'] = hist_data['DayOfWeek'].str.title().map(day_map)

# Create additional cyclical time features for Hour and DayOfWeek
hist_data['Hour'] = hist_data['StartDateTime'].dt.hour
hist_data['Hour_sin'] = np.sin(2 * np.pi * hist_data['Hour'] / 24)
hist_data['Hour_cos'] = np.cos(2 * np.pi * hist_data['Hour'] / 24)
hist_data['DOW_sin'] = np.sin(2 * np.pi * hist_data['DayOfWeek'] / 7)
hist_data['DOW_cos'] = np.cos(2 * np.pi * hist_data['DayOfWeek'] / 7)

# (Optional) Keep original Year, Month, Day
hist_data['Year'] = hist_data['StartDateTime'].dt.year
hist_data['Month'] = hist_data['StartDateTime'].dt.month
hist_data['Day'] = hist_data['StartDateTime'].dt.day

# Feature engineering: lag and rolling features
hist_data['Lag_Price_1h'] = hist_data['Price'].shift(1)
hist_data['Lag_Price_6h'] = hist_data['Price'].shift(6)
hist_data['Lag_Price_24h'] = hist_data['Price'].shift(24)
# Added a 168-hour lag for weekly pattern
hist_data['Lag_Price_168h'] = hist_data['Price'].shift(168)

hist_data['Rolling_Temp_6h'] = hist_data['temperature_2m'].rolling(window=6, min_periods=1).mean()
hist_data['Rolling_Temp_24h'] = hist_data['temperature_2m'].rolling(window=24, min_periods=1).mean()
hist_data['Rolling_Wind_6h'] = hist_data['wind_speed_100m (km/h)'].rolling(window=6, min_periods=1).mean()
hist_data['Rolling_Wind_24h'] = hist_data['wind_speed_100m (km/h)'].rolling(window=24, min_periods=1).mean()
hist_data['Rolling_Load_6h'] = hist_data['total-consumption'].rolling(window=6, min_periods=1).mean()
hist_data['Rolling_Load_24h'] = hist_data['total-consumption'].rolling(window=24, min_periods=1).mean()

# Interpolate and fill missing numeric values
for col in hist_data.columns:
    if hist_data[col].dtype in [np.float64, np.int64]:
        hist_data[col] = hist_data[col].interpolate(method='linear', limit_direction='both')
        mean_val = hist_data[col].mean()
        hist_data[col] = hist_data[col].fillna(mean_val if not pd.isna(mean_val) else 0)

#####################################
# FIX FOR LOG(0) OR NEGATIVE PRICES
#####################################

# Clip prices to avoid zero or negative for log transform
hist_data['Price'] = hist_data['Price'].clip(lower=0.01)
hist_data.replace([np.inf, -np.inf], np.nan, inplace=True)
hist_data.dropna(subset=['Price'], inplace=True)

#####################################
# 3. Define Features and Target
#####################################

features = [
    'temperature_2m', 'precipitation (mm)', 'rain (mm)',
    'snowfall (cm)', 'weather_code (wmo code)',
    'wind_speed_100m (km/h)', 'relative_humidity_2m (%)',
    'total-consumption',
    'Hour_sin', 'Hour_cos', 'DOW_sin', 'DOW_cos',
    'Lag_Price_1h', 'Lag_Price_6h', 'Lag_Price_24h',
    'Lag_Price_168h',
    'Rolling_Temp_6h', 'Rolling_Temp_24h',
    'Rolling_Wind_6h', 'Rolling_Wind_24h',
    'Rolling_Load_6h', 'Rolling_Load_24h'
]
target = 'Price'

# Drop rows with missing feature or target values
hist_data.dropna(subset=features + [target], inplace=True)

X = hist_data[features]
y = hist_data[target]

# --- LOG TRANSFORM the target to reduce large-range issues ---
y_log = np.log1p(y)  # log(Price + 1)

#####################################
# 4. Scaling and Sequence Preparation
#####################################

scaler_X = RobustScaler()
scaler_y = RobustScaler()

X_scaled = scaler_X.fit_transform(X)
y_log_scaled = scaler_y.fit_transform(y_log.values.reshape(-1, 1)).flatten()

# Increase time_steps to 168 to capture weekly patterns
time_steps = 168

def create_sequences(X, y, time_steps=168):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i+time_steps])
        y_seq.append(y[i+time_steps])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X_scaled, y_log_scaled, time_steps)

# Split into training (80%) and test sets
train_size = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:train_size], X_seq[train_size:]
y_train, y_test = y_seq[:train_size], y_seq[train_size:]

#####################################
# 5. Hyperparameter Tuning with KerasTuner
#####################################

def build_lstm_model_hp(hp):
    model = Sequential()
    
    # number of LSTM layers
    lstm_layers = hp.Int('lstm_layers', min_value=1, max_value=3, step=1)
    
    # units for the first LSTM layer
    lstm_units = hp.Choice('lstm_units', values=[32, 64, 128, 256])
    model.add(LSTM(lstm_units, return_sequences=(lstm_layers > 1),
                   input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(hp.Float('dropout_1', 0.1, 0.5, step=0.1)))
    
    # Additional LSTM layers if needed
    for i in range(1, lstm_layers):
        units_i = hp.Choice(f'lstm_units_{i+1}', values=[32, 64, 128])
        return_seq = (i < (lstm_layers - 1))
        model.add(LSTM(units_i, return_sequences=return_seq))
        model.add(Dropout(hp.Float(f'dropout_{i+1}', 0.1, 0.5, step=0.1)))
    
    # Dense layers
    dense_units = hp.Choice('dense_units', values=[16, 32, 64])
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(1))  # single output for log-price
    
    # learning rate
    lr = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')
    return model

tuner = RandomSearch(
    build_lstm_model_hp,
    objective='val_loss',
    max_trials=20,  # bumped up from 10 to 20
    executions_per_trial=1,
    overwrite=True,
    directory='lstm_tuning',
    project_name='cyclical_lstm_no_wavelet'
)

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

tuner.search(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best Hyperparameters:", best_hps.values)

#####################################
# 6. Build Final Model with Best Hyperparameters
#####################################

model_hp = tuner.hypermodel.build(best_hps)
early_stop_final = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)

history_final = model_hp.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop_final, reduce_lr],
    verbose=1
)

#####################################
# 7. Evaluate on Test Data
#####################################

y_pred_log_scaled = model_hp.predict(X_test).flatten()
# Invert the scaling
y_pred_log = scaler_y.inverse_transform(y_pred_log_scaled.reshape(-1, 1)).flatten()
# Invert the log transform
y_pred = np.expm1(y_pred_log)

# Do the same for y_test
y_test_log = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_true = np.expm1(y_test_log)

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
mape = safe_mape(y_true, y_pred, eps=1.0)

print("\nEvaluation Metrics on Historical Data:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.4f}")
print(f"MAPE: {mape:.2f}%")

plt.figure(figsize=(10,6))
plt.plot(y_true, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.title("Historical Data: Actual vs Predicted (LSTM, 168h Window, Log-Transform)")
plt.show()

#####################################
# 8. Future Forecasting (Recursive Prediction)
#####################################

# Load and process future data similarly
future_df = pd.read_csv('data/future-data.csv')
future_df.rename(columns={
    'start date/time': 'StartDateTime',
    'grid_load': 'total-consumption',
    'day of the week': 'DayOfWeek'
}, inplace=True)

# Clean numeric columns in future data
for col in numeric_cols:
    if col in future_df.columns and col != 'Price':
        future_df[col] = clean_numeric_column(future_df[col])

future_df['StartDateTime'] = pd.to_datetime(
    future_df['StartDateTime'],
    format='%d/%m/%Y %H:%M',
    errors='coerce'
)
future_df['DayOfWeek'] = future_df['DayOfWeek'].str.title().map(day_map)
future_df.dropna(subset=['StartDateTime', 'DayOfWeek'], inplace=True)
future_df.sort_values('StartDateTime', inplace=True)

forecast_start_dt = pd.to_datetime('06/01/2025 00:00:00', format='%d/%m/%Y %H:%M:%S')
forecast_end_dt = pd.to_datetime('12/01/2025 23:00:00', format='%d/%m/%Y %H:%M:%S')
future_df = future_df[(future_df['StartDateTime'] >= forecast_start_dt) &
                      (future_df['StartDateTime'] <= forecast_end_dt)].copy()

for col in future_df.columns:
    if future_df[col].dtype in [np.float64, np.int64]:
        future_df[col] = future_df[col].interpolate(method='linear', limit_direction='both')
        mean_val = future_df[col].mean()
        future_df[col] = future_df[col].fillna(mean_val if not pd.isna(mean_val) else 0)

future_df.reset_index(drop=True, inplace=True)

# Add cyclical features for future data
future_df['Hour'] = future_df['StartDateTime'].dt.hour
future_df['Hour_sin'] = np.sin(2 * np.pi * future_df['Hour'] / 24)
future_df['Hour_cos'] = np.cos(2 * np.pi * future_df['Hour'] / 24)
future_df['DOW_sin'] = np.sin(2 * np.pi * future_df['DayOfWeek'] / 7)
future_df['DOW_cos'] = np.cos(2 * np.pi * future_df['DayOfWeek'] / 7)

# Use the last 168 time steps from historical data for recursive predictions
last_historical_data = hist_data.tail(time_steps)
initial_sequence = last_historical_data[features].values
initial_sequence_scaled = scaler_X.transform(initial_sequence)

predictions = []

for idx, row in future_df.iterrows():
    dt = row['StartDateTime']
    
    # For lag features, use previous prediction if available; else historical Price
    lag_1 = predictions[-1] if len(predictions) >= 1 else last_historical_data['Price'].iloc[-1]
    lag_6 = predictions[-6] if len(predictions) >= 6 else last_historical_data['Price'].iloc[-6]
    lag_24 = predictions[-24] if len(predictions) >= 24 else last_historical_data['Price'].iloc[-24]
    lag_168 = predictions[-168] if len(predictions) >= 168 else last_historical_data['Price'].iloc[-168]
    
    # Build feature vector for prediction (order must match training features)
    features_for_pred = [
        row['temperature_2m'],
        row['precipitation (mm)'],
        row['rain (mm)'],
        row['snowfall (cm)'],
        row['weather_code (wmo code)'],
        row['wind_speed_100m (km/h)'],
        row['relative_humidity_2m (%)'],
        row['total-consumption'],
        np.sin(2 * np.pi * dt.hour / 24),     # Hour_sin
        np.cos(2 * np.pi * dt.hour / 24),       # Hour_cos
        np.sin(2 * np.pi * dt.dayofweek / 7),   # DOW_sin
        np.cos(2 * np.pi * dt.dayofweek / 7),   # DOW_cos
        lag_1,
        lag_6,
        lag_24,
        lag_168,
        np.mean(initial_sequence_scaled[-6:, features.index('temperature_2m')]),
        np.mean(initial_sequence_scaled[-24:, features.index('temperature_2m')]),
        np.mean(initial_sequence_scaled[-6:, features.index('wind_speed_100m (km/h)')]),
        np.mean(initial_sequence_scaled[-24:, features.index('wind_speed_100m (km/h)')]),
        np.mean(initial_sequence_scaled[-6:, features.index('total-consumption')]),
        np.mean(initial_sequence_scaled[-24:, features.index('total-consumption')])
    ]
    
    # Create DataFrame with same columns as features
    pred_df = pd.DataFrame([features_for_pred], columns=features)
    # Use .values to avoid feature name warning
    features_scaled = scaler_X.transform(pred_df.values)
    
    # Update sliding window: roll and insert new row
    initial_sequence_scaled = np.roll(initial_sequence_scaled, -1, axis=0)
    initial_sequence_scaled[-1] = features_scaled[0]
    
    # Reshape for LSTM: (1, time_steps, num_features)
    input_seq = initial_sequence_scaled.reshape(1, time_steps, initial_sequence_scaled.shape[1])
    pred_scaled = model_hp.predict(input_seq).flatten()
    # Invert log-scale
    pred_price_log = scaler_y.inverse_transform(pred_scaled.reshape(-1,1)).flatten()
    pred_price = np.expm1(pred_price_log)[0]
    predictions.append(pred_price)

future_df['Predicted Price [Euro/MWh]'] = predictions

# (Optional) Merge with actual future prices if available
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
    actual_df,
    left_on='StartDateTime',
    right_on='start date/time',
    how='inner'
)

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

pred_prices = merged_df['Predicted Price [Euro/MWh]'].values
actual_prices = merged_df['actual_price'].values

lstm_rmse = np.sqrt(mean_squared_error(actual_prices, pred_prices))
lstm_mae = mean_absolute_error(actual_prices, pred_prices)
lstm_r2 = r2_score(actual_prices, pred_prices)
lstm_mape = safe_mape(actual_prices, pred_prices, eps=1.0)

print("\nLSTM Prediction Performance on Forecast Period:")
print(f"RMSE: {lstm_rmse:.2f}")
print(f"MAE: {lstm_mae:.2f}")
print(f"R²: {lstm_r2:.4f}")
print(f"MAPE: {lstm_mape:.2f}%")

# print("\nPredictions for 06-Jan-2025 to 12-Jan-2025:")
# print(future_df[['StartDateTime', 'Predicted Price [Euro/MWh]']].to_string(index=False))