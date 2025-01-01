import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from prophet import Prophet
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("data/merged-data.csv")

# Data cleaning and preprocessing
data['Price Germany/Luxembourg [Euro/MWh]'] = data['Price Germany/Luxembourg [Euro/MWh]'].replace({',': ''}, regex=True).astype(float)
data['Total (grid consumption) [MWh]'] = data['Total (grid consumption) [MWh]'].replace({',': ''}, regex=True).astype(float)

# Convert Start date/time to datetime and extract useful time features
data['Start date/time'] = pd.to_datetime(data['Start date/time'], dayfirst=True)
data['Year'] = data['Start date/time'].dt.year
data['Month'] = data['Start date/time'].dt.month
data['Day'] = data['Start date/time'].dt.day
data['Hour'] = data['Start date/time'].dt.hour
data['DayOfWeek'] = data['Start date/time'].dt.dayofweek  # Add day of the week

# Create lagged features for target variable
data['Lag_Price'] = data['Price Germany/Luxembourg [Euro/MWh]'].shift(1)

# Add rolling averages for features
data['Rolling_Temp_24h'] = data['temperature_2m (°C)'].rolling(window=24).mean()
data['Rolling_Wind_24h'] = data['wind_speed_100m (km/h)'].rolling(window=24).mean()
data['Rolling_Load_24h'] = data['Total (grid consumption) [MWh]'].rolling(window=24).mean()

# Drop rows with missing values after lagging and rolling
data = data.dropna()

# Feature selection based on correlation and engineering
features = [
    'temperature_2m (°C)', 
    'wind_speed_100m (km/h)', 
    'Total (grid consumption) [MWh]', 
    'Day', 
    'Hour', 
    'DayOfWeek',  # Added day of the week
    'Rolling_Temp_24h',  # Added rolling averages
    'Rolling_Wind_24h',
    'Rolling_Load_24h',
    'Lag_Price'
]
target = 'Price Germany/Luxembourg [Euro/MWh]'

# Split data chronologically for train-test
train_data = data[data['Start date/time'] < '2023-09-30']
test_data = data[data['Start date/time'] >= '2023-09-30']

X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 1: Train Prophet model
prophet_train_data = train_data[['Start date/time', target]].rename(columns={'Start date/time': 'ds', target: 'y'})
prophet_model = Prophet(
    changepoint_prior_scale=0.01,  # Reduce to make trends less flexible
    seasonality_prior_scale=10.0,  # Increase to capture stronger seasonality
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=True
)
prophet_model.fit(prophet_train_data)

# Generate predictions for the training data
prophet_train_pred = prophet_model.predict(prophet_train_data)['yhat'].values

# Generate predictions for the test data
prophet_test_pred = prophet_model.predict(test_data[['Start date/time']].rename(columns={'Start date/time': 'ds'}))['yhat'].values

# Add Prophet predictions as a feature to XGBoost
X_train_scaled = np.hstack([X_train_scaled, prophet_train_pred.reshape(-1, 1)])  # Add Prophet predictions
X_test_scaled = np.hstack([X_test_scaled, prophet_test_pred.reshape(-1, 1)])  # Add Prophet predictions

# Step 2: Hyperparameter tuning for XGBoost with TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)  # Time-series cross-validation

# Expanded hyperparameter grid
xgb_param_grid = {
    'n_estimators': [50, 100, 200, 300, 500],
    'max_depth': [3, 6, 9, 12, 15],
    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4],
    'min_child_weight': [1, 3, 5, 7],
    'reg_alpha': [0, 0.1, 1, 10],
    'reg_lambda': [0, 0.1, 1, 10]
}

xgb_random = RandomizedSearchCV(
    estimator=xgb.XGBRegressor(random_state=42),
    param_distributions=xgb_param_grid,
    n_iter=50,  # Increased number of iterations
    cv=tscv,  # Use TimeSeriesSplit for cross-validation
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Train the model with the Prophet predictions as an additional feature
xgb_random.fit(X_train_scaled, y_train)
best_xgb = xgb_random.best_estimator_
print(f"Best XGBoost Parameters: {xgb_random.best_params_}")

# Predict with the best XGBoost model
xgb_test_pred = best_xgb.predict(X_test_scaled)

# Evaluate the hybrid model
mse_hybrid = mean_squared_error(y_test, xgb_test_pred)
mae_hybrid = mean_absolute_error(y_test, xgb_test_pred)
r2_hybrid = r2_score(y_test, xgb_test_pred)

print("Hybrid Model (Prophet + XGBoost) Evaluation:")
print(f"Mean Squared Error: {mse_hybrid:.4f}")
print(f"Mean Absolute Error: {mae_hybrid:.4f}")
print(f"R-squared: {r2_hybrid:.4f}")

# Step 3: Forecast target variable using Prophet
prophet_test_pred = prophet_model.predict(test_data[['Start date/time']].rename(columns={'Start date/time': 'ds'}))['yhat']
mse_prophet = mean_squared_error(y_test, prophet_test_pred)
print(f"Prophet MSE: {mse_prophet}")

# Step 4: Feature Importance Analysis
xgb.plot_importance(best_xgb)
plt.show()

# Step 5: Ensemble Approach
ensemble_pred = (prophet_test_pred + xgb_test_pred) / 2
mse_ensemble = mean_squared_error(y_test, ensemble_pred)
print(f"Ensemble MSE: {mse_ensemble}")

# Step 6: Use Plain XGBoost (if Prophet is not adding value)
# Train XGBoost without Prophet predictions
xgb_random.fit(X_train_scaled[:, :-1], y_train)  # Exclude Prophet predictions
best_xgb = xgb_random.best_estimator_
xgb_test_pred = best_xgb.predict(X_test_scaled[:, :-1])

# Evaluate plain XGBoost
mse_xgb = mean_squared_error(y_test, xgb_test_pred)
print(f"Plain XGBoost MSE: {mse_xgb}")

# Predict for the coming week
# Assuming 'data' contains the latest available data
last_date = data['Start date/time'].max()
future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=7*24, freq='h')  # Use 'h' instead of 'H'

# Create future DataFrame
future_data = pd.DataFrame(index=future_dates)
future_data['Year'] = future_data.index.year
future_data['Month'] = future_data.index.month
future_data['Day'] = future_data.index.day
future_data['Hour'] = future_data.index.hour
future_data['DayOfWeek'] = future_data.index.dayofweek  # Add day of the week

# Initialize Lag_Price with the last observed price
future_data['Lag_Price'] = data['Price Germany/Luxembourg [Euro/MWh]'].iloc[-1]

# Step 7: Forecast weather data (temperature and wind speed) using Prophet
def forecast_feature(data, feature, periods=7*24):
    # Prepare data for Prophet
    feature_data = data[['Start date/time', feature]].rename(
        columns={'Start date/time': 'ds', feature: 'y'}
    )
    
    # Train Prophet model
    model = Prophet()
    model.fit(feature_data)
    
    # Forecast for future dates
    future = model.make_future_dataframe(periods=periods, freq='h')  # Use 'h' instead of 'H'
    forecast = model.predict(future)
    
    return forecast['yhat'].tail(periods).values

# Forecast temperature and wind speed
future_data['temperature_2m (°C)'] = forecast_feature(data, 'temperature_2m (°C)')
future_data['wind_speed_100m (km/h)'] = forecast_feature(data, 'wind_speed_100m (km/h)')

# Step 8: Forecast grid load using Prophet
future_data['Total (grid consumption) [MWh]'] = forecast_feature(data, 'Total (grid consumption) [MWh]')

# Add rolling averages for future data
future_data['Rolling_Temp_24h'] = future_data['temperature_2m (°C)'].rolling(window=24).mean()
future_data['Rolling_Wind_24h'] = future_data['wind_speed_100m (km/h)'].rolling(window=24).mean()
future_data['Rolling_Load_24h'] = future_data['Total (grid consumption) [MWh]'].rolling(window=24).mean()

# Fill NaN values in rolling averages (first 23 rows)
future_data = future_data.bfill()  # Use bfill() instead of fillna(method='bfill')

# Step 9: Forecast target variable using Prophet
prophet_future_pred = prophet_model.predict(future_data.reset_index().rename(columns={'index': 'ds'}))['yhat'].values

# Add Prophet predictions as a feature to XGBoost
future_data_scaled = scaler.transform(future_data[features])  # Exclude Prophet predictions
future_data_scaled = np.hstack([future_data_scaled, prophet_future_pred.reshape(-1, 1)])  # Add Prophet predictions

# Predict future prices using the hybrid model
xgb_future_pred = best_xgb.predict(future_data_scaled)

# Create a DataFrame for the predictions
future_predictions_df = pd.DataFrame({
    'Start date/time': future_dates,  # The future dates generated earlier
    'Predicted Price [Euro/MWh]': xgb_future_pred  # The predicted prices
})

# Print the predictions with dates
print("Predictions for the coming week:")
print(future_predictions_df)

# Load actual data
actual_data = pd.DataFrame({
    'Start date/time': [
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
actual_data['Start date/time'] = pd.to_datetime(actual_data['Start date/time'])

# Merge predictions and actual data
comparison_df = pd.merge(future_predictions_df, actual_data, on='Start date/time', suffixes=('_predicted', '_actual'))

# Calculate evaluation metrics
mse = mean_squared_error(comparison_df['Actual Price [Euro/MWh]'], comparison_df['Predicted Price [Euro/MWh]'])
mae = mean_absolute_error(comparison_df['Actual Price [Euro/MWh]'], comparison_df['Predicted Price [Euro/MWh]'])
r2 = r2_score(comparison_df['Actual Price [Euro/MWh]'], comparison_df['Predicted Price [Euro/MWh]'])
rmse = np.sqrt(mse)

print("Evaluation Metrics:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R-squared: {r2:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")

# Plot predicted vs actual prices
plt.figure(figsize=(12, 6))
plt.plot(comparison_df['Start date/time'], comparison_df['Predicted Price [Euro/MWh]'], label='Predicted', marker='o')
plt.plot(comparison_df['Start date/time'], comparison_df['Actual Price [Euro/MWh]'], label='Actual', marker='x')
plt.title('Predicted vs Actual Hourly Prices')
plt.xlabel('Date/Time')
plt.ylabel('Price [Euro/MWh]')
plt.legend()
plt.grid(True)
plt.show()

# Plot residuals
comparison_df['Residuals'] = comparison_df['Actual Price [Euro/MWh]'] - comparison_df['Predicted Price [Euro/MWh]']
plt.figure(figsize=(12, 6))
plt.plot(comparison_df['Start date/time'], comparison_df['Residuals'], marker='o', color='red')
plt.axhline(0, color='black', linestyle='--')
plt.title('Residuals (Actual - Predicted)')
plt.xlabel('Date/Time')
plt.ylabel('Residuals [Euro/MWh]')
plt.grid(True)
plt.show()