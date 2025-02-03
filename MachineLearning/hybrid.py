import pandas as pd
import numpy as np
import holidays
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import xgboost as xgb
import lightgbm as lgb  # pip install lightgbm if needed
from prophet import Prophet
import optuna
from optuna.samplers import TPESampler

# -----------------------
# 1. Data Loading & Preprocessing
# -----------------------
data = pd.read_csv("data/merged-data.csv")

# Clean numeric fields
data['Price Germany/Luxembourg [Euro/MWh]'] = data['Price Germany/Luxembourg [Euro/MWh]']\
    .replace({',': ''}, regex=True).astype(float)
data['Total (grid consumption) [MWh]'] = data['Total (grid consumption) [MWh]']\
    .replace({',': ''}, regex=True).astype(float)

# Convert datetime and create time features
data['Start date/time'] = pd.to_datetime(data['Start date/time'], dayfirst=True)
data['Year'] = data['Start date/time'].dt.year
data['Month'] = data['Start date/time'].dt.month
data['Day'] = data['Start date/time'].dt.day
data['Hour'] = data['Start date/time'].dt.hour
data['DayOfWeek'] = data['Start date/time'].dt.dayofweek

# Create a holiday flag for Germany
de_holidays = holidays.CountryHoliday('DE')
data['is_holiday'] = data['Start date/time'].dt.date.astype('datetime64[ns]')\
    .isin(list(de_holidays.keys())).astype(int)

# Create lag features
data['Lag_Price_1'] = data['Price Germany/Luxembourg [Euro/MWh]'].shift(1)
data['Lag_Price_24'] = data['Price Germany/Luxembourg [Euro/MWh]'].shift(24)
data['Lag_Load_1'] = data['Total (grid consumption) [MWh]'].shift(1)
data['Lag_Load_24'] = data['Total (grid consumption) [MWh]'].shift(24)

# Create rolling averages (24-hour windows)
data['Rolling_Price_24h'] = data['Price Germany/Luxembourg [Euro/MWh]'].rolling(window=24).mean()
data['Rolling_Load_24h'] = data['Total (grid consumption) [MWh]'].rolling(window=24).mean()
data['Rolling_Temp_24h'] = data['temperature_2m (°C)'].rolling(window=24).mean()
data['Rolling_Wind_24h'] = data['wind_speed_100m (km/h)'].rolling(window=24).mean()

# Drop initial rows with missing values
data = data.dropna().reset_index(drop=True)

# Rename columns for convenience
data.rename(columns={
    'temperature_2m (°C)': 'temperature_2m_C',
    'wind_speed_100m (km/h)': 'wind_speed_100m_kmh',
    'Total (grid consumption) [MWh]': 'Total_grid_consumption_MWh'
}, inplace=True)

# -----------------------
# 2. Prophet Model for Trend & Seasonality (Improved)
# -----------------------
# Prepare dataframe for Prophet (include regressors)
prophet_df = data[['Start date/time', 'Price Germany/Luxembourg [Euro/MWh]', 
                   'temperature_2m_C', 'Total_grid_consumption_MWh', 
                   'wind_speed_100m_kmh', 'is_holiday']].copy()
prophet_df.rename(columns={'Start date/time': 'ds',
                           'Price Germany/Luxembourg [Euro/MWh]': 'y'}, inplace=True)

# Configure Prophet with improved settings:
m = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=True,         # Enable daily seasonality for hourly data
    changepoint_prior_scale=0.1,      # More flexibility
    seasonality_mode='multiplicative',
    interval_width=0.95
)
m.add_country_holidays(country_name='DE')
m.add_regressor('temperature_2m_C')
m.add_regressor('Total_grid_consumption_MWh')
m.add_regressor('wind_speed_100m_kmh')
m.add_regressor('is_holiday')
# Optionally, add custom seasonality (e.g., an 8-hour cycle)
m.add_seasonality(name='8hour', period=8, fourier_order=3)

# Fit Prophet
m.fit(prophet_df)

# Define forecast period to match actual data (October 1 to October 7, 2024)
forecast_start = pd.to_datetime("2024-10-01 00:00:00")
forecast_end   = pd.to_datetime("2024-10-07 23:00:00")
forecast_dates = pd.date_range(start=forecast_start, end=forecast_end, freq='h')

future = pd.DataFrame({'ds': forecast_dates})
# Use the last available regressor values (or use forecasts if available)
latest_regressors = data[['Start date/time', 'temperature_2m_C', 'Total_grid_consumption_MWh', 
                          'wind_speed_100m_kmh', 'is_holiday']].iloc[-1]
for col in ['temperature_2m_C', 'Total_grid_consumption_MWh', 'wind_speed_100m_kmh', 'is_holiday']:
    future[col] = latest_regressors[col]

# Generate Prophet forecast for the period
forecast = m.predict(future)
# Use the trend component as our forecast (alternatively, you could use yhat)
forecast = forecast[['ds', 'trend']].rename(columns={'trend': 'prophet_trend'})

# Merge the Prophet forecast back into the data (if needed)
data = data.merge(forecast, left_on='Start date/time', right_on='ds', how='left')
data['prophet_prediction'] = data['prophet_trend']  # Use trend for hybrid

# -----------------------
# 3. Residual Modeling (Stacked Ensemble)
# -----------------------
# Calculate residuals from Prophet
data['residuals'] = data['Price Germany/Luxembourg [Euro/MWh]'] - data['prophet_prediction']
data['residuals'] = data['residuals'].fillna(0).replace([np.inf, -np.inf], 0)

# Build features for the residual model
residual_features = [
    'temperature_2m_C',
    'wind_speed_100m_kmh',
    'Total_grid_consumption_MWh',
    'Day',
    'Hour',
    'DayOfWeek',
    'is_holiday',
    'Lag_Price_1',
    'Lag_Price_24',
    'Rolling_Price_24h',
    'Rolling_Temp_24h',
    'Rolling_Wind_24h',
    'Rolling_Load_24h',
    'prophet_trend'
]
X_residual = data[residual_features]
y_residual = data['residuals']

scaler_residual = StandardScaler()
X_residual_scaled = scaler_residual.fit_transform(X_residual)

tscv = TimeSeriesSplit(n_splits=5)

# --- 3a. Tune and Train XGBoost for Residuals ---
def objective_xgb(trial):
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
        'eval_metric': 'rmse'
    }
    errors = []
    for train_idx, val_idx in tscv.split(X_residual_scaled):
        X_train, X_val = X_residual_scaled[train_idx], X_residual_scaled[val_idx]
        y_train, y_val = y_residual.iloc[train_idx], y_residual.iloc[val_idx]
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        pred = model.predict(X_val)
        errors.append(mean_squared_error(y_val, pred))
    return np.mean(errors)

study_xgb = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
study_xgb.optimize(objective_xgb, n_trials=50)
best_params_xgb = study_xgb.best_params

best_xgb_res = xgb.XGBRegressor(**best_params_xgb, random_state=42)
best_xgb_res.fit(X_residual_scaled, y_residual)

# --- 3b. Tune and Train LightGBM for Residuals ---
def objective_lgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'random_state': 42
    }
    errors = []
    for train_idx, val_idx in tscv.split(X_residual_scaled):
        X_train, X_val = X_residual_scaled[train_idx], X_residual_scaled[val_idx]
        y_train, y_val = y_residual.iloc[train_idx], y_residual.iloc[val_idx]
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        pred = model.predict(X_val)
        errors.append(mean_squared_error(y_val, pred))
    return np.mean(errors)

study_lgb = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
study_lgb.optimize(objective_lgb, n_trials=50)
best_params_lgb = study_lgb.best_params

best_lgb_res = lgb.LGBMRegressor(**best_params_lgb, random_state=42)
best_lgb_res.fit(X_residual_scaled, y_residual)

# --- 3c. Ensemble Residual Predictions ---
data['xgb_res_pred'] = best_xgb_res.predict(X_residual_scaled)
data['lgb_res_pred'] = best_lgb_res.predict(X_residual_scaled)
data['ensemble_res_pred'] = (data['xgb_res_pred'] + data['lgb_res_pred']) / 2

# -----------------------
# 4. Final Hybrid Prediction
# -----------------------
# Clean predictions (fill NaNs if any)
data['prophet_trend'] = data['prophet_trend'].ffill()
data['ensemble_res_pred'] = data['ensemble_res_pred'].fillna(0)

# Calculate hybrid prediction
data['hybrid_prediction'] = data['prophet_trend'] + data['ensemble_res_pred']

# Remove any remaining NaNs from the target columns
valid_data = data.dropna(subset=['Price Germany/Luxembourg [Euro/MWh]', 'hybrid_prediction'])

rmse = np.sqrt(mean_squared_error(valid_data['Price Germany/Luxembourg [Euro/MWh]'], valid_data['hybrid_prediction']))
mae = mean_absolute_error(valid_data['Price Germany/Luxembourg [Euro/MWh]'], valid_data['hybrid_prediction'])
r2 = r2_score(valid_data['Price Germany/Luxembourg [Euro/MWh]'], valid_data['hybrid_prediction'])

print("Final Hybrid Model Evaluation Metrics:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R-squared: {r2:.4f}")

# Plot feature importance for the XGB residual model
feat_imp = pd.DataFrame({'Feature': residual_features, 'Importance': best_xgb_res.feature_importances_})
feat_imp = feat_imp.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10,6))
plt.barh(feat_imp['Feature'], feat_imp['Importance'])
plt.xlabel("Importance")
plt.title("XGB Residual Model Feature Importance")
plt.show()

# -----------------------
# 5. Future Predictions
# -----------------------
def create_future_features(last_date, data, forecast_dates):
    """Create features for future predictions."""
    future_df = pd.DataFrame({'ds': forecast_dates})
    # Use the last known regressor values
    last_reg = data.iloc[-1]
    for col in ['temperature_2m_C', 'Total_grid_consumption_MWh', 'wind_speed_100m_kmh', 'is_holiday']:
        future_df[col] = last_reg[col]
    # Get Prophet forecast for future dates
    prophet_future = m.predict(future_df)
    future_df = future_df.merge(prophet_future[['ds', 'trend']], on='ds', how='left')
    future_df.rename(columns={'trend': 'prophet_trend'}, inplace=True)
    # Add time features
    future_df['Hour'] = future_df['ds'].dt.hour
    future_df['Day'] = future_df['ds'].dt.day
    future_df['DayOfWeek'] = future_df['ds'].dt.dayofweek
    # Use last known lag/rolling features (for demonstration)
    for col in ['Lag_Price_1', 'Lag_Price_24', 'Rolling_Price_24h', 'Rolling_Temp_24h', 
                'Rolling_Wind_24h', 'Rolling_Load_24h']:
        future_df[col] = last_reg[col]
    return future_df

forecast_dates = pd.date_range(start=forecast_start, end=forecast_end, freq='h')
future_df = create_future_features(data['Start date/time'].max(), data, forecast_dates)

# Ensure all residual features are present in future_df
missing_feats = [f for f in residual_features if f not in future_df.columns]
if missing_feats:
    raise ValueError(f"Missing features in future_df: {missing_feats}")

future_X = scaler_residual.transform(future_df[residual_features])
future_df['xgb_res_pred'] = best_xgb_res.predict(future_X)
future_df['lgb_res_pred'] = best_lgb_res.predict(future_X)
future_df['ensemble_res_pred'] = (future_df['xgb_res_pred'] + future_df['lgb_res_pred']) / 2

# Final hybrid prediction for future dates
future_df['final_price'] = future_df['prophet_trend'] + future_df['ensemble_res_pred']
future_df['final_price'] = np.where(future_df['final_price'] < 0, 0, future_df['final_price'])
future_df['final_price'] = future_df['final_price'].rolling(3, center=True, min_periods=1).mean()

print("Predictions for the coming week:")
print(future_df[['ds', 'final_price']].tail(10))

# -----------------------
# 6. Compare Predictions to Actual Data
# -----------------------
# Prepare prediction dataframe with matching column names
future_predictions_df = future_df[['ds', 'final_price']].copy()
future_predictions_df.rename(columns={'ds': 'Start date/time',
                                        'final_price': 'Predicted Price [Euro/MWh]'}, inplace=True)
# Ensure the 'Start date/time' columns are datetime
future_predictions_df['Start date/time'] = pd.to_datetime(future_predictions_df['Start date/time'])

# Actual data for the same period (ensure these dates match the forecast dates)
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

# Merge predictions and actual data on 'Start date/time'
comparison_df = pd.merge(future_predictions_df, actual_data, on='Start date/time', suffixes=('_predicted', '_actual'))

# Check if the merged DataFrame is empty
if comparison_df.empty:
    raise ValueError("No matching dates found between predictions and actual data. Please verify your date ranges.")

# Drop any rows missing predicted or actual values
comparison_df = comparison_df.dropna(subset=['Predicted Price [Euro/MWh]', 'Actual Price [Euro/MWh]'])

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

# Plot predicted vs. actual prices
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