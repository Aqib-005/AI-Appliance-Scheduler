import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler
import joblib

# 1. Load the data
data = pd.read_csv("data/merged-data.csv")

# (Optional) Fix mis-encoded characters in column names
data.columns = data.columns.str.replace("Ã¸", "°")

# 2. Data cleaning and preprocessing
# Rename columns for consistency:
#   - "start date/time" -> "StartDateTime"
#   - "day of the week" -> "DayOfWeek"
#   - "germany/luxembourg [?/mwh]" -> "Price"   (this is our target)
#   - "total (grid load) [mwh]" -> "total-consumption"
#   - "temperature_2m (°C)" -> "temperature_2m"
data.rename(columns={
    'start date/time': 'StartDateTime',
    'day of the week': 'DayOfWeek',
    'germany/luxembourg [?/mwh]': 'Price',
    'total (grid load) [mwh]': 'total-consumption',
    'temperature_2m (°c)': 'temperature_2m',
    'temperature_2m (°C)': 'temperature_2m'
}, inplace=True)

# Convert Price and total-consumption to numeric (remove commas if needed)
data['Price'] = data['Price'].replace({',': ''}, regex=True).astype(float)
data['total-consumption'] = data['total-consumption'].replace({',': ''}, regex=True).astype(float)

# Convert StartDateTime to datetime and sort
data['StartDateTime'] = pd.to_datetime(data['StartDateTime'], dayfirst=True)
data = data.sort_values('StartDateTime').reset_index(drop=True)

# Extract time features
data['Year'] = data['StartDateTime'].dt.year
data['Month'] = data['StartDateTime'].dt.month
data['Day'] = data['StartDateTime'].dt.day
data['Hour'] = data['StartDateTime'].dt.hour
data['DayOfWeek'] = data['StartDateTime'].dt.dayofweek  # Monday=0, Sunday=6

# Create lagged feature for Price
data['Lag_Price'] = data['Price'].shift(1)

# Add rolling averages for features (24-hour window)
data['Rolling_Temp_24h'] = data['temperature_2m'].rolling(window=24).mean()
data['Rolling_Wind_24h'] = data['wind_speed_100m (km/h)'].rolling(window=24).mean()
data['Rolling_Load_24h'] = data['total-consumption'].rolling(window=24).mean()

# Drop rows with missing values after lagging/rolling
data = data.dropna()

# 3. Feature selection
features = [
    'temperature_2m', 
    'wind_speed_100m (km/h)', 
    'total-consumption', 
    'Day', 
    'Hour', 
    'DayOfWeek',
    'Rolling_Temp_24h',
    'Rolling_Wind_24h',
    'Rolling_Load_24h',
    'Lag_Price'
]
target = 'Price'

# Subset the data
X = data[features]
y = data[target]

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use TimeSeriesSplit for cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# 4. Define the objective function for Optuna tuning
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 0.2),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
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
study.optimize(objective, n_trials=50)
best_params = study.best_params
print(f"Best Parameters: {best_params}")

# Train final XGBoost model with best parameters
best_xgb = xgb.XGBRegressor(**best_params, random_state=42)
best_xgb.fit(X_scaled, y)

# Evaluate using the last fold of TimeSeriesSplit
train_index, test_index = list(tscv.split(X_scaled))[-1]
X_train, X_test = X_scaled[train_index], X_scaled[test_index]
y_train, y_test = y.iloc[train_index], y.iloc[test_index]
y_pred = best_xgb.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Evaluation Metrics:")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R²: {r2}")

# Analyze feature importance
feature_importance = best_xgb.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()

# ----- Future Forecasting Using Merged Dataset -----
# We assume merged-data.csv includes both historical and future rows.
# The future rows already have forecasted regressors for weather and consumption, and Price is NaN.
# We filter the desired forecast period and predict Price for those rows.

# Load the merged dataset
merged_data = pd.read_csv("data/merged-data.csv")
merged_data.rename(columns={'start date/time': 'StartDateTime'}, inplace=True)
merged_data['StartDateTime'] = pd.to_datetime(merged_data['StartDateTime'], dayfirst=True)

# Define the forecast period: 2025-01-20 00:00:00 to 2025-01-26 23:00:00
forecast_start = '2025-01-20 00:00:00'
forecast_end = '2025-01-26 23:00:00'
future_data = merged_data[(merged_data['StartDateTime'] >= forecast_start) & 
                          (merged_data['StartDateTime'] <= forecast_end)]

# Select only future rows where Price is missing (NaN)
future_data = future_data[future_data['Price'].isna()]

# Ensure these future rows contain the same engineered features as used in training
future_features = future_data[features]

# Scale future features using the same scaler
future_features_scaled = scaler.transform(future_features)

# Predict future prices using the trained XGBoost model
future_predictions = best_xgb.predict(future_features_scaled)

# Add predictions to future_data DataFrame
future_data['Predicted Price [Euro/MWh]'] = future_predictions

# Save predictions for the future period
future_predictions_df = future_data[['StartDateTime', 'Predicted Price [Euro/MWh]']]
future_predictions_df.to_csv('future_predictions_xgb.csv', index=False)
print("Future predictions saved to future_predictions_xgb.csv")

# Optionally, if you have actual future data, merge and evaluate:
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
    mse = mean_squared_error(comparison_df['Actual Price [Euro/MWh]'], comparison_df['Predicted Price [Euro/MWh]'])
    mae = mean_absolute_error(comparison_df['Actual Price [Euro/MWh]'], comparison_df['Predicted Price [Euro/MWh]'])
    r2 = r2_score(comparison_df['Actual Price [Euro/MWh]'], comparison_df['Predicted Price [Euro/MWh]'])
    rmse = np.sqrt(mse)
    print("Evaluation Metrics on Future Data:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R-squared: {r2:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    
    plt.figure(figsize=(12, 6))
    plt.plot(comparison_df['StartDateTime'], comparison_df['Predicted Price [Euro/MWh]'], label='Predicted', marker='o')
    plt.plot(comparison_df['StartDateTime'], comparison_df['Actual Price [Euro/MWh]'], label='Actual', marker='x')
    plt.title('Predicted vs Actual Hourly Prices')
    plt.xlabel('StartDateTime')
    plt.ylabel('Price [Euro/MWh]')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    comparison_df['Residuals'] = comparison_df['Actual Price [Euro/MWh]'] - comparison_df['Predicted Price [Euro/MWh]']
    plt.figure(figsize=(12, 6))
    plt.plot(comparison_df['StartDateTime'], comparison_df['Residuals'], marker='o', color='red')
    plt.axhline(0, color='black', linestyle='--')
    plt.title('Residuals (Actual - Predicted)')
    plt.xlabel('StartDateTime')
    plt.ylabel('Residuals [Euro/MWh]')
    plt.grid(True)
    plt.show()