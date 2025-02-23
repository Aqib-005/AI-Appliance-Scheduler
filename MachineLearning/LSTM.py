import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from prophet import Prophet

# 1. Enhanced Data Loading and Preprocessing
def load_and_preprocess_data(path):
    data = pd.read_csv(path)
    
    # Column standardization
    data.rename(columns={
        'start date/time': 'StartDateTime',
        'day of the week': 'DayOfWeek',
        'day-price': 'Price'
    }, inplace=True)
    
    # Numeric conversion with error handling
    numeric_cols = ['Price', 'total-consumption']
    for col in numeric_cols:
        data[col] = data[col].astype(str).str.replace(',', '', regex=False)
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Datetime handling
    data['StartDateTime'] = pd.to_datetime(data['StartDateTime'], dayfirst=True)
    data = data.sort_values('StartDateTime').reset_index(drop=True)
    
    # Temporal feature engineering
    data['Day'] = data['StartDateTime'].dt.day
    data['Hour'] = data['StartDateTime'].dt.hour
    data['DayOfWeek'] = data['DayOfWeek'].map({
        "Monday": 0, "Tuesday": 1, "Wednesday": 2,
        "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
    })
    
    return data.dropna()

# 2. Advanced Feature Engineering
def create_features(data, lookback=72):
    # Multiple lagged features
    for lag in [1, 24, 48]:
        data[f'Lag_Price_{lag}'] = data['Price'].shift(lag)
    
    # Rolling features with dynamic window
    rolling_windows = {
        'temperature_2m': 24,
        'wind_speed_100m (km/h)': 12,
        'total-consumption': 24
    }
    
    for col, window in rolling_windows.items():
        data[f'Rolling_{col.split(" ")[0]}_{window}h'] = data[col].rolling(window).mean()
        data[f'Rolling_{col.split(" ")[0]}_{window}h_std'] = data[col].rolling(window).std()
    
    # Cyclical encoding for temporal features
    data['Hour_sin'] = np.sin(2 * np.pi * data['Hour']/24)
    data['Hour_cos'] = np.cos(2 * np.pi * data['Hour']/24)
    data['DayOfWeek_sin'] = np.sin(2 * np.pi * data['DayOfWeek']/7)
    data['DayOfWeek_cos'] = np.cos(2 * np.pi * data['DayOfWeek']/7)
    
    return data.dropna()

# 3. Model Architecture with Regularization
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, 
             input_shape=input_shape,
             kernel_regularizer=L1L2(l1=1e-5, l2=1e-4)),
        Dropout(0.3),
        LSTM(64, return_sequences=False,
             kernel_regularizer=L1L2(l1=1e-5, l2=1e-4)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    return model

# 4. Learning Rate Schedule
def lr_scheduler(epoch, lr):
    if epoch < 30:
        return lr
    elif epoch < 60:
        return lr * 0.5
    else:
        return lr * 0.1
    
def prepare_future_data(future_data, historical_data):
    """Add engineered features to future data using historical context"""
    # Cyclical encoding
    future_data['Hour_sin'] = np.sin(2 * np.pi * future_data['Hour']/24)
    future_data['Hour_cos'] = np.cos(2 * np.pi * future_data['Hour']/24)
    future_data['DayOfWeek_sin'] = np.sin(2 * np.pi * future_data['DayOfWeek']/7)
    future_data['DayOfWeek_cos'] = np.cos(2 * np.pi * future_data['DayOfWeek']/7)
    
    # Lagged prices initialization
    last_prices = historical_data['Price'].iloc[-48:].values
    for lag in [1, 24, 48]:
        future_data[f'Lag_Price_{lag}'] = np.concatenate([
            last_prices[-lag:], 
            np.zeros(len(future_data) - lag)  # Fill remaining with 0s
        ])[:len(future_data)]
    
    # Rolling features initialization
    rolling_config = {
        'temperature_2m': 24,
        'wind_speed_100m (km/h)': 12,
        'total-consumption': 24
    }
    
    for col, window in rolling_config.items():
        col_base = col.split(" ")[0]
        # Initialize with historical rolling values
        historical_vals = historical_data[f'Rolling_{col_base}_{window}h'][-window:].values
        future_vals = future_data[col].rolling(window, min_periods=1).mean().values
        future_data[f'Rolling_{col_base}_{window}h'] = np.concatenate([
            historical_vals,
            future_vals[window:]
        ])[:len(future_data)]
        
    return future_data.ffill().bfill()

# Main Execution Flow
if __name__ == "__main__":
    # Load and preprocess data
    data = load_and_preprocess_data("data/merged-data.csv")
    data = create_features(data)
    
    # Feature Selection
    features = [
        'temperature_2m', 'wind_speed_100m (km/h)', 'total-consumption',
        'Hour_sin', 'Hour_cos', 'DayOfWeek_sin', 'DayOfWeek_cos',
        'Lag_Price_1', 'Lag_Price_24', 'Lag_Price_48',
        'Rolling_temperature_2m_24h', 'Rolling_wind_speed_100m_12h',
        'Rolling_total-consumption_24h'
    ]
    target = 'Price'
    
    # Data Scaling
    scaler_X = StandardScaler()
    scaler_y = MinMaxScaler()
    
    X = scaler_X.fit_transform(data[features])
    y = scaler_y.fit_transform(data[target].values.reshape(-1, 1))
    
    # Sequence Creation
    def create_sequences(X, y, time_steps=24):
        X_seq, y_seq = [], []
        for i in range(len(X) - time_steps):
            X_seq.append(X[i:i+time_steps])
            y_seq.append(y[i+time_steps])
        return np.array(X_seq), np.array(y_seq)
    
    X_seq, y_seq = create_sequences(X, y)
    train_size = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:train_size], X_seq[train_size:]
    y_train, y_test = y_seq[:train_size], y_seq[train_size:]
    
    # Model Configuration
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    # Training with callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    lr_callback = LearningRateScheduler(lr_scheduler)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=150,
        batch_size=64,
        callbacks=[early_stop, lr_callback],
        verbose=1
    )
    
    # Evaluation
    y_pred = model.predict(X_test)
    y_test_actual = scaler_y.inverse_transform(y_test)
    y_pred_actual = scaler_y.inverse_transform(y_pred)
    
    print("\nEvaluation Metrics:")
    print(f"MSE: {mean_squared_error(y_test_actual, y_pred_actual):.2f}")
    print(f"MAE: {mean_absolute_error(y_test_actual, y_pred_actual):.2f}")
    print(f"R²: {r2_score(y_test_actual, y_pred_actual):.2f}")

# 19. Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# ----- Forecasting for the Coming Week using Prophet -----

def forecast_feature(data, feature, start_date, end_date):
    """Forecast a feature between specific dates using Prophet"""
    feature_data = data[['StartDateTime', feature]].rename(columns={'StartDateTime': 'ds', feature: 'y'})
    model_prophet = Prophet()
    model_prophet.fit(feature_data)
    
    future = pd.date_range(start=start_date, end=end_date, freq='h')  # Changed to 'h'
    forecast = model_prophet.predict(pd.DataFrame({'ds': future}))
    return forecast[['ds', 'yhat']]

# Define custom date range
start_date = '2025-01-20 00:00:00'
end_date = '2025-01-26 23:00:00'
future_dates = pd.date_range(start=start_date, end=end_date, freq='h')

# Create future DataFrame
future_data = pd.DataFrame(index=future_dates)
future_data['Year'] = future_data.index.year
future_data['Month'] = future_data.index.month
future_data['Day'] = future_data.index.day
future_data['Hour'] = future_data.index.hour
future_data['DayOfWeek'] = future_data.index.dayofweek

# Forecast features for custom dates
temp_forecast = forecast_feature(data, 'temperature_2m', start_date, end_date)
wind_forecast = forecast_feature(data, 'wind_speed_100m (km/h)', start_date, end_date)
load_forecast = forecast_feature(data, 'total-consumption', start_date, end_date)

# Merge forecasts
future_data = future_data.merge(temp_forecast, left_index=True, right_on='ds')
future_data = future_data.merge(wind_forecast, on='ds', suffixes=('', '_wind'))
future_data = future_data.merge(load_forecast, on='ds', suffixes=('', '_load'))
future_data.rename(columns={
    'yhat': 'temperature_2m',
    'yhat_wind': 'wind_speed_100m (km/h)',
    'yhat_load': 'total-consumption'
}, inplace=True)
future_data.set_index('ds', inplace=True)
future_data = prepare_future_data(future_data, data)

# Add rolling averages
future_data['Rolling_Temp_24h'] = future_data['temperature_2m'].rolling(24).mean()
future_data['Rolling_Wind_24h'] = future_data['wind_speed_100m (km/h)'].rolling(24).mean()
future_data['Rolling_Load_24h'] = future_data['total-consumption'].rolling(24).mean()
future_data['Lag_Price'] = data['Price'].iloc[-1]
future_data.bfill(inplace=True)

# Scale features
future_data_scaled = scaler_X.transform(future_data[features])

# Prepare sequences
future_seq = []
for i in range(len(future_data_scaled) - time_steps + 1):
    future_seq.append(future_data_scaled[i:i+time_steps])
future_seq = np.array(future_seq)

# Predict prices
future_predictions_scaled = model.predict(future_seq)
future_predictions = scaler_y.inverse_transform(future_predictions_scaled)

# Save predictions
future_predictions_df = pd.DataFrame({
    'StartDateTime': future_dates[time_steps-1:],
    'Predicted Price [Euro/MWh]': future_predictions.flatten()
})
# future_predictions_df.to_csv('custom_predictions.csv', index=False)

# ----- Comparison with Actual Data -----
actual_data = pd.DataFrame({
    'StartDateTime': [
        # Jan 20, 2025
        '2025-01-20 00:00:00', '2025-01-20 01:00:00', '2025-01-20 02:00:00', 
        '2025-01-20 03:00:00', '2025-01-20 04:00:00', '2025-01-20 05:00:00', 
        '2025-01-20 06:00:00', '2025-01-20 07:00:00', '2025-01-20 08:00:00', 
        '2025-01-20 09:00:00', '2025-01-20 10:00:00', '2025-01-20 11:00:00', 
        '2025-01-20 12:00:00', '2025-01-20 13:00:00', '2025-01-20 14:00:00', 
        '2025-01-20 15:00:00', '2025-01-20 16:00:00', '2025-01-20 17:00:00', 
        '2025-01-20 18:00:00', '2025-01-20 19:00:00', '2025-01-20 20:00:00', 
        '2025-01-20 21:00:00', '2025-01-20 22:00:00', '2025-01-20 23:00:00',
        # Jan 21, 2025
        '2025-01-21 00:00:00', '2025-01-21 01:00:00', '2025-01-21 02:00:00', 
        '2025-01-21 03:00:00', '2025-01-21 04:00:00', '2025-01-21 05:00:00', 
        '2025-01-21 06:00:00', '2025-01-21 07:00:00', '2025-01-21 08:00:00', 
        '2025-01-21 09:00:00', '2025-01-21 10:00:00', '2025-01-21 11:00:00', 
        '2025-01-21 12:00:00', '2025-01-21 13:00:00', '2025-01-21 14:00:00', 
        '2025-01-21 15:00:00', '2025-01-21 16:00:00', '2025-01-21 17:00:00', 
        '2025-01-21 18:00:00', '2025-01-21 19:00:00', '2025-01-21 20:00:00', 
        '2025-01-21 21:00:00', '2025-01-21 22:00:00', '2025-01-21 23:00:00',
        # Jan 22, 2025
        '2025-01-22 00:00:00', '2025-01-22 01:00:00', '2025-01-22 02:00:00', 
        '2025-01-22 03:00:00', '2025-01-22 04:00:00', '2025-01-22 05:00:00', 
        '2025-01-22 06:00:00', '2025-01-22 07:00:00', '2025-01-22 08:00:00', 
        '2025-01-22 09:00:00', '2025-01-22 10:00:00', '2025-01-22 11:00:00', 
        '2025-01-22 12:00:00', '2025-01-22 13:00:00', '2025-01-22 14:00:00', 
        '2025-01-22 15:00:00', '2025-01-22 16:00:00', '2025-01-22 17:00:00', 
        '2025-01-22 18:00:00', '2025-01-22 19:00:00', '2025-01-22 20:00:00', 
        '2025-01-22 21:00:00', '2025-01-22 22:00:00', '2025-01-22 23:00:00',
        # Jan 23, 2025
        '2025-01-23 00:00:00', '2025-01-23 01:00:00', '2025-01-23 02:00:00', 
        '2025-01-23 03:00:00', '2025-01-23 04:00:00', '2025-01-23 05:00:00', 
        '2025-01-23 06:00:00', '2025-01-23 07:00:00', '2025-01-23 08:00:00', 
        '2025-01-23 09:00:00', '2025-01-23 10:00:00', '2025-01-23 11:00:00', 
        '2025-01-23 12:00:00', '2025-01-23 13:00:00', '2025-01-23 14:00:00', 
        '2025-01-23 15:00:00', '2025-01-23 16:00:00', '2025-01-23 17:00:00', 
        '2025-01-23 18:00:00', '2025-01-23 19:00:00', '2025-01-23 20:00:00', 
        '2025-01-23 21:00:00', '2025-01-23 22:00:00', '2025-01-23 23:00:00',
        # Jan 24, 2025
        '2025-01-24 00:00:00', '2025-01-24 01:00:00', '2025-01-24 02:00:00', 
        '2025-01-24 03:00:00', '2025-01-24 04:00:00', '2025-01-24 05:00:00', 
        '2025-01-24 06:00:00', '2025-01-24 07:00:00', '2025-01-24 08:00:00', 
        '2025-01-24 09:00:00', '2025-01-24 10:00:00', '2025-01-24 11:00:00', 
        '2025-01-24 12:00:00', '2025-01-24 13:00:00', '2025-01-24 14:00:00', 
        '2025-01-24 15:00:00', '2025-01-24 16:00:00', '2025-01-24 17:00:00', 
        '2025-01-24 18:00:00', '2025-01-24 19:00:00', '2025-01-24 20:00:00', 
        '2025-01-24 21:00:00', '2025-01-24 22:00:00', '2025-01-24 23:00:00',
        # Jan 25, 2025
        '2025-01-25 00:00:00', '2025-01-25 01:00:00', '2025-01-25 02:00:00', 
        '2025-01-25 03:00:00', '2025-01-25 04:00:00', '2025-01-25 05:00:00', 
        '2025-01-25 06:00:00', '2025-01-25 07:00:00', '2025-01-25 08:00:00', 
        '2025-01-25 09:00:00', '2025-01-25 10:00:00', '2025-01-25 11:00:00', 
        '2025-01-25 12:00:00', '2025-01-25 13:00:00', '2025-01-25 14:00:00', 
        '2025-01-25 15:00:00', '2025-01-25 16:00:00', '2025-01-25 17:00:00', 
        '2025-01-25 18:00:00', '2025-01-25 19:00:00', '2025-01-25 20:00:00', 
        '2025-01-25 21:00:00', '2025-01-25 22:00:00', '2025-01-25 23:00:00',
        # Jan 26, 2025
        '2025-01-26 00:00:00', '2025-01-26 01:00:00', '2025-01-26 02:00:00', 
        '2025-01-26 03:00:00', '2025-01-26 04:00:00', '2025-01-26 05:00:00', 
        '2025-01-26 06:00:00', '2025-01-26 07:00:00', '2025-01-26 08:00:00', 
        '2025-01-26 09:00:00', '2025-01-26 10:00:00', '2025-01-26 11:00:00', 
        '2025-01-26 12:00:00', '2025-01-26 13:00:00', '2025-01-26 14:00:00', 
        '2025-01-26 15:00:00', '2025-01-26 16:00:00', '2025-01-26 17:00:00', 
        '2025-01-26 18:00:00', '2025-01-26 19:00:00', '2025-01-26 20:00:00', 
        '2025-01-26 21:00:00', '2025-01-26 22:00:00', '2025-01-26 23:00:00'
    ],
    'Actual Price [Euro/MWh]': [
        # Jan 20, 2025
        122.27, 119.44, 116.56, 114.41, 115.45, 127.54, 161.71, 276.48, 431.99, 
        291.70, 236.29, 187.48, 176.00, 171.39, 191.85, 277.17, 402.12, 583.40, 
        473.28, 295.57, 220.00, 170.00, 152.51, 137.98,
        # Jan 21, 2025
        127.52, 121.65, 116.67, 113.86, 113.54, 122.34, 142.20, 190.06, 248.32, 
        211.68, 198.05, 161.50, 147.44, 142.83, 150.05, 173.43, 212.13, 301.15, 
        251.93, 202.68, 174.84, 155.19, 138.60, 125.29,
        # Jan 22, 2025
        128.00, 125.06, 122.22, 121.66, 123.16, 128.32, 156.92, 208.25, 238.60, 
        199.41, 179.06, 172.94, 165.00, 170.00, 179.92, 195.67, 199.04, 208.99, 
        180.43, 167.08, 134.49, 127.78, 128.09, 114.68,
        # Jan 23, 2025
        113.70, 108.88, 105.58, 100.01, 97.59, 100.01, 106.41, 136.60, 159.03, 
        155.42, 130.94, 116.10, 94.93, 88.56, 85.86, 89.80, 89.87, 106.75, 112.00, 
        101.87, 86.36, 74.28, 75.68, 58.23,
        # Jan 24, 2025
        69.03, 58.16, 45.05, 40.60, 50.17, 73.20, 85.44, 109.16, 116.23, 96.34, 
        88.90, 78.44, 76.48, 74.69, 74.51, 74.51, 75.25, 83.02, 88.59, 91.78, 
        84.99, 80.79, 91.97, 86.29,
        # Jan 25, 2025
        59.04, 64.96, 63.55, 67.17, 76.77, 76.89, 79.32, 76.89, 88.99, 87.31, 
        86.32, 88.15, 82.55, 81.98, 89.00, 111.07, 132.97, 145.85, 151.00, 147.53, 
        137.60, 134.61, 132.26, 122.03,
        # Jan 26, 2025
        126.21, 115.79, 111.30, 106.85, 105.43, 107.86, 110.60, 117.62, 124.17, 
        121.97, 105.51, 102.57, 96.73, 90.02, 92.52, 111.63, 125.92, 132.77, 118.94, 
        90.32, 78.93, 68.26, 49.25, 23.89
    ]
})

actual_data['StartDateTime'] = pd.to_datetime(actual_data['StartDateTime'])

comparison_df = pd.merge(future_predictions_df, actual_data, on='StartDateTime', how='inner')

if comparison_df.empty:
    print("No matching actual data available for the forecast period. Skipping future evaluation metrics.")
else:
    mse_future = mean_squared_error(comparison_df['Actual Price [Euro/MWh]'], comparison_df['Predicted Price [Euro/MWh]'])
    mae_future = mean_absolute_error(comparison_df['Actual Price [Euro/MWh]'], comparison_df['Predicted Price [Euro/MWh]'])
    r2_future = r2_score(comparison_df['Actual Price [Euro/MWh]'], comparison_df['Predicted Price [Euro/MWh]'])
    rmse_future = np.sqrt(mse_future)

    print("Future Evaluation Metrics:")
    print(f"Mean Squared Error: {mse_future:.4f}")
    print(f"Mean Absolute Error: {mae_future:.4f}")
    print(f"R-squared: {r2_future:.4f}")
    print(f"Root Mean Squared Error: {rmse_future:.4f}")

    # Plot predicted vs actual prices
    plt.figure(figsize=(12, 6))
    plt.plot(comparison_df['StartDateTime'], comparison_df['Predicted Price [Euro/MWh]'], label='Predicted', marker='o')
    plt.plot(comparison_df['StartDateTime'], comparison_df['Actual Price [Euro/MWh]'], label='Actual', marker='x')
    plt.title('Predicted vs Actual Hourly Prices')
    plt.xlabel('StartDateTime')
    plt.ylabel('Price [Euro/MWh]')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot residuals
    comparison_df['Residuals'] = comparison_df['Actual Price [Euro/MWh]'] - comparison_df['Predicted Price [Euro/MWh]']
    plt.figure(figsize=(12, 6))
    plt.plot(comparison_df['StartDateTime'], comparison_df['Residuals'], marker='o', color='red')
    plt.axhline(0, color='black', linestyle='--')
    plt.title('Residuals (Actual - Predicted)')
    plt.xlabel('StartDateTime')
    plt.ylabel('Residuals [Euro/MWh]')
    plt.grid(True)
    plt.show()