import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
from prophet import Prophet
import holidays

# Load and clean data ---------------------------------------------------------
data = pd.read_csv("data/merged-data.csv")

# Clean numeric columns
numeric_cols = ['Price Germany/Luxembourg [Euro/MWh]', 'Total (grid consumption) [MWh]']
for col in numeric_cols:
    data[col] = data[col].replace({',': '', '': np.nan}, regex=True)
    data[col] = data[col].astype(float)
    data[col] = data[col].ffill().bfill()  # Forward then backward fill

# Convert datetime
data['Start date/time'] = pd.to_datetime(data['Start date/time'], dayfirst=True)

# Hybrid Model: Prophet + XGBoost ---------------------------------------------

# 1. Prophet Trend Component
prophet_df = data[['Start date/time', 'Price Germany/Luxembourg [Euro/MWh]']].rename(
    columns={'Start date/time': 'ds', 'Price Germany/Luxembourg [Euro/MWh]': 'y'}
)

# Configure Prophet with German holidays
m = Prophet(
    yearly_seasonality=False,  # Disable yearly seasonality to speed up
    weekly_seasonality=True,
    daily_seasonality=True,
    changepoint_prior_scale=0.05
)
m.add_country_holidays(country_name='DE')
m.fit(prophet_df)

# Generate trend predictions
future = m.make_future_dataframe(periods=0, freq='h')
forecast = m.predict(future)
data = data.merge(forecast[['ds', 'trend', 'yhat']], left_on='Start date/time', right_on='ds')
data.rename(columns={'trend': 'prophet_trend', 'yhat': 'prophet_prediction'}, inplace=True)

# 2. Residual Calculation with Safety Checks
data['residuals'] = data['Price Germany/Luxembourg [Euro/MWh]'] - data['prophet_prediction']
data['residuals'] = data['residuals'].fillna(0).replace([np.inf, -np.inf], 0)

# Feature Engineering ---------------------------------------------------------
# Add time features
data['Hour'] = data['Start date/time'].dt.hour
data['DayOfWeek'] = data['Start date/time'].dt.dayofweek
data['Month'] = data['Start date/time'].dt.month

# Create rolling features with NaN handling
for col in ['temperature_2m (°C)', 'wind_speed_100m (km/h)', 'Total (grid consumption) [MWh]']:
    data[f'Rolling_24h_{col}'] = data[col].rolling(24, min_periods=1).mean().ffill().bfill()

features = [
    'temperature_2m (°C)',
    'wind_speed_100m (km/h)', 
    'Total (grid consumption) [MWh]',
    'Rolling_24h_temperature_2m (°C)',
    'Rolling_24h_wind_speed_100m (km/h)',
    'Rolling_24h_Total (grid consumption) [MWh]',
    'prophet_trend',
    'Hour',
    'DayOfWeek',
    'Month'
]

# Final Data Preparation ------------------------------------------------------
data = data.dropna(subset=features + ['Price Germany/Luxembourg [Euro/MWh]'])
X = data[features]
y = data['residuals']

# Normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# XGBoost Model ---------------------------------------------------------------
# Simplified hyperparameter grid for faster tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1],
    'reg_alpha': [0, 10],
    'reg_lambda': [0, 10]
}

model = xgb.XGBRegressor(random_state=42)
randomized_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=10,  # Number of random combinations to try
    cv=2,  # Fewer folds for faster tuning
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    random_state=42
)
randomized_search.fit(X_scaled, y)

best_model = randomized_search.best_estimator_

# Generate Predictions --------------------------------------------------------
data['xgb_residual_pred'] = best_model.predict(X_scaled)
data['hybrid_prediction'] = data['prophet_prediction'] + data['xgb_residual_pred']

# Final Safety Check
data['hybrid_prediction'] = data['hybrid_prediction'].fillna(data['prophet_prediction'])
data['hybrid_prediction'] = data['hybrid_prediction'].replace([np.inf, -np.inf], data['prophet_prediction'])

# Evaluation -------------------------------------------------------------------
print("\nHybrid Model Training Metrics:")
print(f"RMSE: {np.sqrt(mean_squared_error(data['Price Germany/Luxembourg [Euro/MWh]'], data['hybrid_prediction'])):.2f}")
print(f"MAE: {mean_absolute_error(data['Price Germany/Luxembourg [Euro/MWh]'], data['hybrid_prediction']):.2f}")
print(f"R²: {r2_score(data['Price Germany/Luxembourg [Euro/MWh]'], data['hybrid_prediction']):.2f}")

# Future Predictions ----------------------------------------------------------
def create_future_features(last_date, periods=168):
    """Create future features for forecasting."""
    future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=periods, freq='h')
    future_df = pd.DataFrame({'ds': future_dates})
    
    # Prophet forecast
    prophet_forecast = m.predict(future_df)
    
    # Weather/load forecasts with forward filling
    future_df['temperature_2m (°C)'] = data['temperature_2m (°C)'].rolling(24, min_periods=1).mean().ffill().iloc[-1]
    future_df['wind_speed_100m (km/h)'] = data['wind_speed_100m (km/h)'].rolling(24, min_periods=1).mean().ffill().iloc[-1]
    future_df['Total (grid consumption) [MWh]'] = data['Total (grid consumption) [MWh]'].rolling(24, min_periods=1).mean().ffill().iloc[-1]
    
    # Add rolling features
    for col in ['temperature_2m (°C)', 'wind_speed_100m (km/h)', 'Total (grid consumption) [MWh]']:
        future_df[f'Rolling_24h_{col}'] = future_df[col].rolling(24, min_periods=1).mean().ffill().bfill()
    
    # Time features
    future_df['Hour'] = future_df['ds'].dt.hour
    future_df['DayOfWeek'] = future_df['ds'].dt.dayofweek
    future_df['Month'] = future_df['ds'].dt.month
    future_df['prophet_trend'] = prophet_forecast['trend']
    future_df['prophet_prediction'] = prophet_forecast['yhat']
    
    return future_df

# Generate future predictions
last_date = data['Start date/time'].max()
future_df = create_future_features(last_date)

# Ensure all features are present
assert all(feature in future_df.columns for feature in features), "Missing features in future_df"

# Scale and predict
future_X = scaler.transform(future_df[features])
future_df['xgb_residual'] = best_model.predict(future_X)
future_df['final_price'] = future_df['prophet_prediction'] + future_df['xgb_residual']

# Clean future predictions
future_df['final_price'] = future_df['final_price'].fillna(future_df['prophet_prediction'])
future_df['final_price'] = future_df.apply(
    lambda row: row['prophet_prediction'] if np.isinf(row['final_price']) else row['final_price'],
    axis=1
)

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

# Convert 'Start date/time' in actual_data to datetime
actual_data['Start date/time'] = pd.to_datetime(actual_data['Start date/time'])

# Merge the dataframes
comparison_df = pd.merge(
    future_df.rename(columns={'ds': 'Start date/time'}),
    actual_data,
    on='Start date/time',
    how='inner'
)

# Clean merged data
comparison_df = comparison_df.dropna()

# Calculate evaluation metrics
mse = mean_squared_error(comparison_df['Actual Price [Euro/MWh]'], comparison_df['final_price'])
mae = mean_absolute_error(comparison_df['Actual Price [Euro/MWh]'], comparison_df['final_price'])
r2 = r2_score(comparison_df['Actual Price [Euro/MWh]'], comparison_df['final_price'])
rmse = np.sqrt(mse)

print("\nFuture Evaluation Metrics:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.2f}")

# Plot diagnostics
fig, ax = plt.subplots(2, 1, figsize=(12, 10))
ax[0].scatter(comparison_df['final_price'], comparison_df['Actual Price [Euro/MWh]'], alpha=0.6)
ax[0].plot([0, 250], [0, 250], '--r')
ax[0].set_title('Actual vs Predicted Prices')
ax[0].set_xlabel('Predicted')
ax[0].set_ylabel('Actual')

ax[1].plot(comparison_df['Start date/time'], comparison_df['final_price'] - comparison_df['Actual Price [Euro/MWh]'], 'o')
ax[1].axhline(0, color='black', linestyle='--')
ax[1].set_title('Residuals Over Time')
ax[1].set_ylabel('Residuals')

plt.tight_layout()
plt.show()