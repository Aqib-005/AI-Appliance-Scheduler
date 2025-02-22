import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from prophet import Prophet

# 1. Load your CSV
data = pd.read_csv("data/merged-data.csv")

# 2. Rename columns for consistency
data.rename(columns={
    'start date/time': 'StartDateTime',
    'day of the week': 'DayOfWeek',
    'day-price': 'Price'
}, inplace=True)

# 3. Convert columns to float
data['Price'] = data['Price'].replace({',': ''}, regex=True).astype(float)
data['total-consumption'] = data['total-consumption'].replace({',': ''}, regex=True).astype(float)

# 4. Convert to datetime and extract temporal features
data['StartDateTime'] = pd.to_datetime(data['StartDateTime'], dayfirst=True)
data['Day'] = data['StartDateTime'].dt.day  # Extract day from datetime
data['Hour'] = data['StartDateTime'].dt.hour  # Extract hour from datetime

# 5. Map DayOfWeek from string to numeric
day_map = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2,
    "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
}
data['DayOfWeek'] = data['DayOfWeek'].map(day_map)

# 6-8. Feature engineering (lagged price and rolling averages)
data['Lag_Price'] = data['Price'].shift(1)
data['Rolling_Temp_24h'] = data['temperature_2m'].rolling(24).mean()
data['Rolling_Wind_24h'] = data['wind_speed_100m (km/h)'].rolling(24).mean()
data['Rolling_Load_24h'] = data['total-consumption'].rolling(24).mean()
data.dropna(inplace=True)

# 9. Updated feature selection
features = [
    'temperature_2m',
    'wind_speed_100m (km/h)',
    'total-consumption',
    'Day', 'Hour', 'DayOfWeek',
    'Rolling_Temp_24h', 'Rolling_Wind_24h', 'Rolling_Load_24h',
    'Lag_Price'
]
target = 'Price'

# 10. Subset data into X (features) and y (target)
X = data[features]
y = data[target]

# 11. Scale features and target
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# 12. Prepare sequences for LSTM
def create_sequences(X, y, time_steps=24):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i+time_steps])
        y_seq.append(y[i+time_steps])
    return np.array(X_seq), np.array(y_seq)

time_steps = 24  # 24-hour lookback window
X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)

# 13. Split into training and testing sets (80% train, 20% test)
train_size = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:train_size], X_seq[train_size:]
y_train, y_test = y_seq[:train_size], y_seq[train_size:]

# 14. Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    LSTM(25, return_sequences=False),
    Dropout(0.3),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# 15. Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 16. Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# 17. Make predictions on the test set
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test)

# 18. Calculate evaluation metrics
mse_val = mean_squared_error(y_test_actual, y_pred)
mae_val = mean_absolute_error(y_test_actual, y_pred)
r2_val = r2_score(y_test_actual, y_pred)
rmse_val = np.sqrt(mse_val)

print("Evaluation Metrics:")
print(f"Mean Squared Error: {mse_val:.4f}")
print(f"Mean Absolute Error: {mae_val:.4f}")
print(f"R-squared: {r2_val:.4f}")
print(f"Root Mean Squared Error: {rmse_val:.4f}")

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
start_date = '2024-10-01 00:00:00'
end_date = '2024-10-07 23:00:00'
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
future_predictions_df.to_csv('custom_predictions.csv', index=False)

# ----- Comparison with Actual Data -----
actual_data = pd.DataFrame({
     'StartDateTime': [ 
        '2024-10-01 00:00:00', '2024-10-01 01:00:00', '2024-10-01 02:00:00', 
        '2024-10-01 03:00:00', '2024-10-01 04:00:00', '2024-10-01 05:00:00', 
        '2024-10-01 06:00:00', '2024-10-01 07:00:00', '2024-10-01 08:00:00', 
        '2024-10-01 09:00:00', '2024-10-01 10:00:00', '2024-10-01 11:00:00', 
        '2024-10-01 12:00:00', '2024-10-01 13:00:00', '2024-10-01 14:00:00', 
        '2024-10-01 15:00:00', '2024-10-01 16:00:00', '2024-10-01 17:00:00', 
        '2024-10-01 18:00:00', '2024-10-01 19:00:00', '2024-10-01 20:00:00', 
        '2024-10-01 21:00:00', '2024-10-01 22:00:00', '2024-10-01 23:00:00', 
        '2024-10-02 00:00:00', '2024-10-02 01:00:00', '2024-10-02 02:00:00', 
        '2024-10-02 03:00:00', '2024-10-02 04:00:00', '2024-10-02 05:00:00', 
        '2024-10-02 06:00:00', '2024-10-02 07:00:00', '2024-10-02 08:00:00', 
        '2024-10-02 09:00:00', '2024-10-02 10:00:00', '2024-10-02 11:00:00', 
        '2024-10-02 12:00:00', '2024-10-02 13:00:00', '2024-10-02 14:00:00', 
        '2024-10-02 15:00:00', '2024-10-02 16:00:00', '2024-10-02 17:00:00', 
        '2024-10-02 18:00:00', '2024-10-02 19:00:00', '2024-10-02 20:00:00', 
        '2024-10-02 21:00:00', '2024-10-02 22:00:00', '2024-10-02 23:00:00', 
        '2024-10-03 00:00:00', '2024-10-03 01:00:00', '2024-10-03 02:00:00', 
        '2024-10-03 03:00:00', '2024-10-03 04:00:00', '2024-10-03 05:00:00', 
        '2024-10-03 06:00:00', '2024-10-03 07:00:00', '2024-10-03 08:00:00', 
        '2024-10-03 09:00:00', '2024-10-03 10:00:00', '2024-10-03 11:00:00', 
        '2024-10-03 12:00:00', '2024-10-03 13:00:00', '2024-10-03 14:00:00', 
        '2024-10-03 15:00:00', '2024-10-03 16:00:00', '2024-10-03 17:00:00', 
        '2024-10-03 18:00:00', '2024-10-03 19:00:00', '2024-10-03 20:00:00', 
        '2024-10-03 21:00:00', '2024-10-03 22:00:00', '2024-10-03 23:00:00', 
        '2024-10-04 00:00:00', '2024-10-04 01:00:00', '2024-10-04 02:00:00', 
        '2024-10-04 03:00:00', '2024-10-04 04:00:00', '2024-10-04 05:00:00', 
        '2024-10-04 06:00:00', '2024-10-04 07:00:00', '2024-10-04 08:00:00', 
        '2024-10-04 09:00:00', '2024-10-04 10:00:00', '2024-10-04 11:00:00', 
        '2024-10-04 12:00:00', '2024-10-04 13:00:00', '2024-10-04 14:00:00', 
        '2024-10-04 15:00:00', '2024-10-04 16:00:00', '2024-10-04 17:00:00', 
        '2024-10-04 18:00:00', '2024-10-04 19:00:00', '2024-10-04 20:00:00', 
        '2024-10-04 21:00:00', '2024-10-04 22:00:00', '2024-10-04 23:00:00', 
        '2024-10-05 00:00:00', '2024-10-05 01:00:00', '2024-10-05 02:00:00', 
        '2024-10-05 03:00:00', '2024-10-05 04:00:00', '2024-10-05 05:00:00', 
        '2024-10-05 06:00:00', '2024-10-05 07:00:00', '2024-10-05 08:00:00', 
        '2024-10-05 09:00:00', '2024-10-05 10:00:00', '2024-10-05 11:00:00', 
        '2024-10-05 12:00:00', '2024-10-05 13:00:00', '2024-10-05 14:00:00', 
        '2024-10-05 15:00:00', '2024-10-05 16:00:00', '2024-10-05 17:00:00', 
        '2024-10-05 18:00:00', '2024-10-05 19:00:00', '2024-10-05 20:00:00', 
        '2024-10-05 21:00:00', '2024-10-05 22:00:00', '2024-10-05 23:00:00', 
        '2024-10-06 00:00:00', '2024-10-06 01:00:00', '2024-10-06 02:00:00', 
        '2024-10-06 03:00:00', '2024-10-06 04:00:00', '2024-10-06 05:00:00', 
        '2024-10-06 06:00:00', '2024-10-06 07:00:00', '2024-10-06 08:00:00', 
        '2024-10-06 09:00:00', '2024-10-06 10:00:00', '2024-10-06 11:00:00', 
        '2024-10-06 12:00:00', '2024-10-06 13:00:00', '2024-10-06 14:00:00', 
        '2024-10-06 15:00:00', '2024-10-06 16:00:00', '2024-10-06 17:00:00', 
        '2024-10-06 18:00:00', '2024-10-06 19:00:00', '2024-10-06 20:00:00', 
        '2024-10-06 21:00:00', '2024-10-06 22:00:00', '2024-10-06 23:00:00', 
        '2024-10-07 00:00:00', '2024-10-07 01:00:00', '2024-10-07 02:00:00', 
        '2024-10-07 03:00:00', '2024-10-07 04:00:00', '2024-10-07 05:00:00', 
        '2024-10-07 06:00:00', '2024-10-07 07:00:00', '2024-10-07 08:00:00', 
        '2024-10-07 09:00:00', '2024-10-07 10:00:00', '2024-10-07 11:00:00', 
        '2024-10-07 12:00:00', '2024-10-07 13:00:00', '2024-10-07 14:00:00', 
        '2024-10-07 15:00:00', '2024-10-07 16:00:00', '2024-10-07 17:00:00', 
        '2024-10-07 18:00:00', '2024-10-07 19:00:00', '2024-10-07 20:00:00', 
        '2024-10-07 21:00:00', '2024-10-07 22:00:00', '2024-10-07 23:00:00'
    ],
   'Actual Price [Euro/MWh]': [ 
        3.21, 0.07, 0.05, 0.02, 0.09, 6.80, 63.96, 103.35, 114.98, 100.41, 
        76.48, 68.21, 58.60, 55.66, 56.51, 62.18, 98.94, 109.58, 133.90, 
        136.51, 118.54, 92.30, 91.45, 76.24, 85.44, 80.88, 77.09, 74.93, 
        77.14, 81.10, 96.53, 118.83, 135.70, 117.59, 103.21, 96.52, 90.45, 
        86.44, 78.87, 82.07, 81.51, 100.10, 117.01, 130.07, 114.37, 96.54, 
        93.30, 85.00, 67.33, 65.86, 65.14, 64.18, 63.98, 64.40, 65.43, 
        73.37, 76.67, 72.90, 69.28, 63.04, 47.09, 33.98, 39.14, 51.36, 
        66.04, 89.70, 110.07, 115.32, 108.28, 98.76, 90.96, 81.46, 73.90, 
        72.52, 72.10, 72.00, 72.33, 77.10, 103.00, 126.54, 141.42, 116.27, 
        100.09, 85.00, 75.07, 73.49, 75.82, 79.07, 89.80, 107.10, 131.97, 
        149.02, 123.70, 99.90, 99.90, 90.00, 93.10, 84.64, 81.40, 75.54, 
        75.72, 79.82, 86.20, 100.40, 110.62, 103.00, 82.42, 68.27, 51.44, 
        38.88, 43.49, 62.92, 77.12, 99.48, 148.65, 143.28, 107.23, 96.56, 
        85.23, 67.00, 67.04, 63.97, 62.83, 63.35, 62.71, 63.97, 63.41, 
        72.81, 77.20, 66.06, 35.28, 16.68, 5.25, -0.01, -0.01, 0.20, 
        59.60, 90.94, 106.30, 97.22, 72.98, 59.37, 58.69, 51.71, 34.58, 
        35.34, 33.25, 30.15, 36.09, 46.73, 67.59, 100.92, 108.32, 91.86, 
        66.09, 60.22, 54.11, 43.29, 55.00, 67.01, 97.90, 120.71, 237.65, 
        229.53, 121.98, 99.93, 91.91, 79.12
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