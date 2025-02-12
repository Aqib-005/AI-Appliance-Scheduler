import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from prophet import Prophet
import optuna
from optuna.samplers import TPESampler
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import holidays
import joblib

# -----------------------
# 1. Data Loading & Preprocessing
# -----------------------
data = pd.read_csv("data/merged-data.csv")

# Clean numeric fields: remove commas and convert to float
data['Price Germany/Luxembourg [Euro/MWh]'] = data['Price Germany/Luxembourg [Euro/MWh]'].replace({',': ''}, regex=True).astype(float)
data['Total (grid consumption) [MWh]'] = data['Total (grid consumption) [MWh]'].replace({',': ''}, regex=True).astype(float)

# Convert Start date/time to datetime and extract time features
data['Start date/time'] = pd.to_datetime(data['Start date/time'], dayfirst=True)
data['Year'] = data['Start date/time'].dt.year
data['Month'] = data['Start date/time'].dt.month
data['Day'] = data['Start date/time'].dt.day
data['Hour'] = data['Start date/time'].dt.hour
data['DayOfWeek'] = data['Start date/time'].dt.dayofweek

# Add holiday indicator for Germany
de_holidays = holidays.CountryHoliday('DE')
data['is_holiday'] = data['Start date/time'].dt.date.isin(list(de_holidays.keys())).astype(int)

# Lagged target feature
data['Lag_Price'] = data['Price Germany/Luxembourg [Euro/MWh]'].shift(1)

# Rolling averages (24-hour windows) for exogenous features
data['Rolling_Temp_24h'] = data['temperature_2m (°C)'].rolling(window=24).mean()
data['Rolling_Wind_24h'] = data['wind_speed_100m (km/h)'].rolling(window=24).mean()
data['Rolling_Load_24h'] = data['Total (grid consumption) [MWh]'].rolling(window=24).mean()

# Create cyclic features for Hour, DayOfWeek, and Month
data['Hour_sin'] = np.sin(2 * np.pi * data['Hour'] / 24)
data['Hour_cos'] = np.cos(2 * np.pi * data['Hour'] / 24)
data['DoW_sin'] = np.sin(2 * np.pi * data['DayOfWeek'] / 7)
data['DoW_cos'] = np.cos(2 * np.pi * data['DayOfWeek'] / 7)
data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)

# Drop rows with missing values after lagging/rolling
data = data.dropna()

# Check target statistics before and after clipping
print("Target statistics before clipping:")
print(data['Price Germany/Luxembourg [Euro/MWh]'].describe())
low_clip = data['Price Germany/Luxembourg [Euro/MWh]'].quantile(0.01)
high_clip = data['Price Germany/Luxembourg [Euro/MWh]'].quantile(0.99)
data['Price Germany/Luxembourg [Euro/MWh]'] = data['Price Germany/Luxembourg [Euro/MWh]'].clip(lower=low_clip, upper=high_clip)
print("Target statistics after clipping:")
print(data['Price Germany/Luxembourg [Euro/MWh]'].describe())

# -----------------------
# 2. Feature Selection and Target Transformation
# -----------------------
features = [
    'temperature_2m (°C)', 
    'wind_speed_100m (km/h)', 
    'Total (grid consumption) [MWh]', 
    'Day', 'Hour', 'DayOfWeek',
    'Rolling_Temp_24h', 'Rolling_Wind_24h', 'Rolling_Load_24h',
    'Lag_Price',
    'is_holiday',
    'Hour_sin', 'Hour_cos', 'DoW_sin', 'DoW_cos', 'Month_sin', 'Month_cos'
]
target = 'Price Germany/Luxembourg [Euro/MWh]'

X = data[features]
y = data[target].values.reshape(-1, 1)

# Scale features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Because our target still has negatives (after clipping, min is around -11),
# we shift it upward so that we can apply a log1p transform.
shift = abs(data[target].min()) + 1  # e.g., if min=-11, shift=12
y_shifted = y + shift

# Apply log1p transformation
y_transformer = PowerTransformer(method='yeo-johnson')  # Yeo-Johnson can handle negatives, but we already shifted
y_trans = y_transformer.fit_transform(y_shifted).ravel()

# -----------------------
# 3. Hyperparameter Tuning & Training with XGBoost
# -----------------------
tscv = TimeSeriesSplit(n_splits=5)

# Compute sample weights to emphasize extreme deviations
# Here, weight = 1 + (abs(y - median)/std)
y_median = np.median(y)
y_std = np.std(y)
sample_weights = 1 + (np.abs(y - y_median) / y_std).ravel()

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 0.3),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'random_state': 42,
        # Use the robust default objective (we try reg:squarederror here)
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'early_stopping_rounds': 50
    }
    scores = []
    for train_index, val_index in tscv.split(X_scaled):
        X_train, X_val = X_scaled[train_index], X_scaled[val_index]
        y_train, y_val = y_trans[train_index], y_trans[val_index]
        weights_train = sample_weights[train_index]
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, sample_weight=weights_train,
                  eval_set=[(X_val, y_val)], verbose=False)
        y_pred = model.predict(X_val)
        scores.append(mean_squared_error(y_val, y_pred))
    return np.mean(scores)

study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=100)
best_params = study.best_params
print("Best Parameters:", best_params)

# Train final model on the entire dataset using sample weights
best_xgb = xgb.XGBRegressor(**best_params, random_state=42)
best_xgb.fit(X_scaled, y_trans, sample_weight=sample_weights)

# Evaluate on the last fold of TimeSeriesSplit and invert target transformation
train_index, test_index = list(tscv.split(X_scaled))[-1]
X_train, X_test = X_scaled[train_index], X_scaled[test_index]
y_train_trans, y_test_trans = y_trans[train_index], y_trans[test_index]
y_pred_trans = best_xgb.predict(X_test)
# Inverse transform: first invert the Yeo-Johnson, then subtract the shift
y_pred_shifted = y_transformer.inverse_transform(y_pred_trans.reshape(-1, 1)).ravel()
y_pred_orig = y_pred_shifted - shift
y_test_shifted = y_transformer.inverse_transform(y_test_trans.reshape(-1, 1)).ravel()
y_test_orig = y_test_shifted - shift

rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
mae = mean_absolute_error(y_test_orig, y_pred_orig)
r2 = r2_score(y_test_orig, y_pred_orig)
print("Evaluation Metrics on CV (Last fold):")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

# -----------------------
# 4. Feature Importance
# -----------------------
feature_importance = best_xgb.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()

# -----------------------
# 5. Future Price Forecasting (Iterative Prediction)
# -----------------------
# Define future forecast dates (for one week: December 2, 2024 00:00 to December 8, 2024 23:00)
future_dates = pd.date_range(start='2024-12-02 00:00:00', end='2024-12-08 23:00:00', freq='h')

# Create a future DataFrame with time features and add cyclic and holiday features
future_data = pd.DataFrame(index=future_dates)
future_data['Year'] = future_data.index.year
future_data['Month'] = future_data.index.month
future_data['Day'] = future_data.index.day
future_data['Hour'] = future_data.index.hour
future_data['DayOfWeek'] = future_data.index.dayofweek
future_data['is_holiday'] = future_data.index.to_series().apply(lambda x: int(x.date() in de_holidays))
future_data['Hour_sin'] = np.sin(2 * np.pi * future_data['Hour'] / 24)
future_data['Hour_cos'] = np.cos(2 * np.pi * future_data['Hour'] / 24)
future_data['DoW_sin'] = np.sin(2 * np.pi * future_data['DayOfWeek'] / 7)
future_data['DoW_cos'] = np.cos(2 * np.pi * future_data['DayOfWeek'] / 7)
future_data['Month_sin'] = np.sin(2 * np.pi * future_data['Month'] / 12)
future_data['Month_cos'] = np.cos(2 * np.pi * future_data['Month'] / 12)

# Initialize Lag_Price using the last observed price from training data (manually set here)
future_data['Lag_Price'] = 85.00

# Forecast exogenous features for the future period using Prophet
def forecast_feature(data, feature, start_date='2024-12-02', end_date='2024-12-08'):
    future = pd.DataFrame({'ds': pd.date_range(start=start_date, end=end_date, freq='H')})
    model = Prophet(daily_seasonality=True)
    feature_data = data[['Start date/time', feature]].rename(columns={'Start date/time': 'ds', feature: 'y'})
    model.fit(feature_data)
    forecast = model.predict(future)
    return forecast.set_index('ds')['yhat']

future_data['temperature_2m (°C)'] = forecast_feature(data, 'temperature_2m (°C)')
future_data['wind_speed_100m (km/h)'] = forecast_feature(data, 'wind_speed_100m (km/h)')
future_data['Total (grid consumption) [MWh]'] = forecast_feature(data, 'Total (grid consumption) [MWh]')

# Compute rolling averages for future data
future_data['Rolling_Temp_24h'] = future_data['temperature_2m (°C)'].rolling(window=24).mean()
future_data['Rolling_Wind_24h'] = future_data['wind_speed_100m (km/h)'].rolling(window=24).mean()
future_data['Rolling_Load_24h'] = future_data['Total (grid consumption) [MWh]'].rolling(window=24).mean()
future_data.fillna(method='bfill', inplace=True)

# Prepare future features in the same order as training features
future_X = future_data[features]
future_X_scaled = scaler.transform(future_X)

# Iteratively forecast the price: update Lag_Price with the model prediction
xgb_future_pred = []
for i in range(len(future_data)):
    current_features = future_data.iloc[i:i+1][features]
    current_scaled = scaler.transform(current_features)
    pred_trans = best_xgb.predict(current_scaled)
    pred_shifted = y_transformer.inverse_transform(pred_trans.reshape(-1, 1)).ravel()[0]
    pred_orig = pred_shifted - shift
    xgb_future_pred.append(pred_orig)
    if i < len(future_data) - 1:
        future_data.at[future_data.index[i+1], 'Lag_Price'] = pred_orig

future_predictions_df = pd.DataFrame({
    'Start date/time': future_dates,
    'Predicted Price [Euro/MWh]': xgb_future_pred
})

print("Predictions for the coming week:")
print(future_predictions_df.head())

# -----------------------
# 6. Actual Data and Evaluation
# -----------------------
actual_dates = pd.date_range(start='2024-12-02 00:00:00', end='2024-12-08 23:00:00', freq='h')
actual_prices = [
    85.00, 74.77, 65.78, 61.17, 46.44, 50.21, 69.53, 92.08, 125.25, 
    125.20, 123.63, 121.27, 115.80, 115.09, 127.60, 136.74, 144.04, 
    144.75, 147.68, 147.90, 138.00, 120.01, 112.68, 111.72, 102.21, 
    110.73, 103.05, 96.28, 93.22, 100.48, 108.09, 121.76, 142.39, 
    154.62, 150.09, 141.37, 139.23, 135.62, 142.41, 150.02, 168.94, 
    205.18, 210.00, 197.88, 189.83, 170.04, 153.61, 142.08, 130.81, 
    120.99, 116.66, 112.79, 110.19, 112.26, 120.91, 145.20, 209.76, 
    276.66, 250.76, 212.21, 198.95, 192.25, 199.82, 218.44, 242.87, 
    267.35, 287.76, 230.22, 197.59, 165.68, 148.34, 128.33, 113.27, 
    114.77, 105.74, 103.27, 104.50, 103.27, 104.12, 135.88, 143.54, 
    141.29, 131.78, 124.92, 107.31, 102.67, 102.93, 102.89, 100.87, 
    101.50, 118.31, 112.16, 105.36, 87.13, 77.30, 74.17, 43.14, 10.64, 
    4.60, 3.87, 2.31, 2.99, 10.35, 63.49, 92.71, 110.17, 110.16, 
    104.08, 98.37, 91.90, 100.70, 109.00, 116.92, 131.55, 142.99, 
    151.07, 142.48, 130.00, 119.42, 121.08, 102.14, 86.65, 74.32, 
    68.00, 63.87, 58.09, 50.16, 39.46, 43.28, 58.06, 62.86, 64.92, 
    64.92, 68.00, 72.84, 74.24, 82.50, 96.45, 96.00, 87.92, 82.41, 
    78.92, 85.03, 82.50, 74.80, 93.91, 86.83, 86.73, 83.95, 84.44, 
    91.13, 91.25, 91.29, 100.84, 109.40, 110.84, 117.39, 118.36, 
    116.74, 111.91, 121.38, 105.94, 113.61, 98.55, 97.69, 91.29, 
    90.56, 88.60, 75.01
]
actual_prices = actual_prices[:len(actual_dates)]
actual_data = pd.DataFrame({
    'Start date/time': actual_dates,
    'Actual Price [Euro/MWh]': actual_prices
})
actual_data['Start date/time'] = pd.to_datetime(actual_data['Start date/time'])

# Merge predictions with actual data
comparison_df = pd.merge(future_predictions_df, actual_data, on='Start date/time', suffixes=('_predicted', '_actual'))

# Compute evaluation metrics
mse = mean_squared_error(comparison_df['Actual Price [Euro/MWh]'], comparison_df['Predicted Price [Euro/MWh]'])
mae = mean_absolute_error(comparison_df['Actual Price [Euro/MWh]'], comparison_df['Predicted Price [Euro/MWh]'])
r2 = r2_score(comparison_df['Actual Price [Euro/MWh]'], comparison_df['Predicted Price [Euro/MWh]'])
rmse = np.sqrt(mse)
print("Evaluation Metrics on Future Period:")
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
