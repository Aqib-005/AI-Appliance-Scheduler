import pandas as pd
import numpy as np
import holidays
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import xgboost as xgb
import lightgbm as lgb
from prophet import Prophet

# -----------------------
# 1. Data Loading & Preprocessing
# -----------------------
data = pd.read_csv("data/merged-data.csv")

# Clean numeric fields: remove commas and convert to float
data['Price Germany/Luxembourg [Euro/MWh]'] = data['Price Germany/Luxembourg [Euro/MWh]']\
    .replace({',': ''}, regex=True).astype(float)
data['Total (grid consumption) [MWh]'] = data['Total (grid consumption) [MWh]']\
    .replace({',': ''}, regex=True).astype(float)

# Convert date/time and extract time features
data['Start date/time'] = pd.to_datetime(data['Start date/time'], dayfirst=True)
data['Year'] = data['Start date/time'].dt.year
data['Month'] = data['Start date/time'].dt.month
data['Day'] = data['Start date/time'].dt.day
data['Hour'] = data['Start date/time'].dt.hour
data['DayOfWeek'] = data['Start date/time'].dt.dayofweek

# Create holiday flag for Germany
de_holidays = holidays.CountryHoliday('DE')
data['is_holiday'] = data['Start date/time'].dt.date.astype('datetime64[ns]').isin(list(de_holidays.keys())).astype(int)

# Create lag features
data['Lag_Price_1'] = data['Price Germany/Luxembourg [Euro/MWh]'].shift(1)
data['Lag_Price_24'] = data['Price Germany/Luxembourg [Euro/MWh]'].shift(24)
data['Lag_Load_1'] = data['Total (grid consumption) [MWh]'].shift(1)
data['Lag_Load_24'] = data['Total (grid consumption) [MWh]'].shift(24)

# Create rolling averages
data['Rolling_Temp_24h'] = data['temperature_2m (°C)'].rolling(window=24).mean()
data['Rolling_Wind_24h'] = data['wind_speed_100m (km/h)'].rolling(window=24).mean()
data['Rolling_Load_24h'] = data['Total (grid consumption) [MWh]'].rolling(window=24).mean()

data.dropna(inplace=True)

# Rename columns for convenience
data.rename(columns={
    'temperature_2m (°C)': 'temperature_2m_C',
    'wind_speed_100m (km/h)': 'wind_speed_100m_kmh',
    'Total (grid consumption) [MWh]': 'Total_grid_consumption_MWh'
}, inplace=True)

# -----------------------
# 2. Prophet Model for Trend & Seasonality
# -----------------------
# Prepare data for Prophet (including regressors)
prophet_df = data[['Start date/time', 'Price Germany/Luxembourg [Euro/MWh]', 'temperature_2m_C',
                   'Total_grid_consumption_MWh', 'wind_speed_100m_kmh', 'is_holiday']].copy()
prophet_df.rename(columns={'Start date/time': 'ds', 'Price Germany/Luxembourg [Euro/MWh]': 'y'}, inplace=True)

m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True,
            changepoint_prior_scale=0.1, seasonality_mode='multiplicative', interval_width=0.95)
m.add_country_holidays(country_name='DE')
m.add_regressor('temperature_2m_C')
m.add_regressor('Total_grid_consumption_MWh')
m.add_regressor('wind_speed_100m_kmh')
m.add_regressor('is_holiday')
m.add_seasonality(name='8hour', period=8, fourier_order=3)

m.fit(prophet_df)
# Get Prophet's trend forecast (this serves as our base prediction)
prophet_hist = m.predict(prophet_df)[['ds', 'trend']].rename(columns={'trend': 'prophet_trend'})
data = data.merge(prophet_hist, left_on='Start date/time', right_on='ds', how='left')
data['prophet_prediction'] = data['prophet_trend']

# -----------------------
# 3. Residual Modeling (Stacked Ensemble)
# -----------------------
# Compute residuals
data['residuals'] = data['Price Germany/Luxembourg [Euro/MWh]'] - data['prophet_prediction']
data['residuals'].fillna(0, inplace=True)

# Define features for residual modeling (include time, lag, and rolling features)
residual_features = [
    'temperature_2m_C', 'wind_speed_100m_kmh', 'Total_grid_consumption_MWh',
    'Day', 'Hour', 'DayOfWeek', 'is_holiday', 'Lag_Price_1', 'Lag_Price_24',
    'Rolling_Temp_24h', 'Rolling_Wind_24h', 'Rolling_Load_24h', 'prophet_trend'
]
X_residual = data[residual_features]
y_residual = data['residuals']

scaler_residual = StandardScaler()
X_residual_scaled = scaler_residual.fit_transform(X_residual)

# Train ensemble residual models
best_xgb_res = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                                subsample=0.8, colsample_bytree=0.8, random_state=42).fit(X_residual_scaled, y_residual)
best_lgb_res = lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                                subsample=0.8, colsample_bytree=0.8, random_state=42).fit(X_residual_scaled, y_residual)

data['ensemble_res_pred'] = (best_xgb_res.predict(X_residual_scaled) + best_lgb_res.predict(X_residual_scaled)) / 2
data['final_prediction'] = data['prophet_prediction'] + data['ensemble_res_pred']

# -----------------------
# 4. Compare Future Predictions to Actual Data
# -----------------------
# Define the forecast period (October 1–7, 2024)
forecast_start = "2024-10-01 00:00:00"
forecast_end = "2024-10-07 23:00:00"

# Extract future data from your dataset (make sure your dataset contains actual values for this period)
future_data = data[data['Start date/time'].between(forecast_start, forecast_end)]
if future_data.empty:
    raise ValueError("Future data is empty. Please verify the forecast period and data availability.")

# Compute ensemble residual predictions for the future period
future_features = future_data[residual_features]
future_scaled = scaler_residual.transform(future_features)
future_residual_pred = best_xgb_res.predict(future_scaled) + best_lgb_res.predict(future_scaled)

# In our hybrid approach, final future prediction = Prophet prediction + half the ensemble residual
future_predictions = future_data['prophet_prediction'] + future_residual_pred / 2

future_predictions_df = pd.DataFrame({
    'Start date/time': future_data['Start date/time'],
    'Predicted Price [Euro/MWh]': future_predictions
})

# For actual values during the forecast period, extract them from your dataset
actual_data = future_data[['Start date/time', 'Price Germany/Luxembourg [Euro/MWh]']].copy()
actual_data.rename(columns={'Price Germany/Luxembourg [Euro/MWh]': 'Actual Price [Euro/MWh]'}, inplace=True)

# Merge predictions with actual data for comparison
comparison_df = future_predictions_df.merge(actual_data, on='Start date/time', how='left')

# Plot the predicted vs actual prices
plt.figure(figsize=(12, 6))
plt.plot(comparison_df['Start date/time'], comparison_df['Predicted Price [Euro/MWh]'], label='Predicted Price', linestyle='dashed', marker='o')
plt.plot(comparison_df['Start date/time'], comparison_df['Actual Price [Euro/MWh]'], label='Actual Price', linestyle='solid', marker='x', alpha=0.7)
plt.xlabel("Date/Time")
plt.ylabel("Price [Euro/MWh]")
plt.title("Comparison of Predicted vs Actual Prices (Forecast Period)")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()

mse_future = mean_squared_error(comparison_df['Actual Price [Euro/MWh]'], comparison_df['Predicted Price [Euro/MWh]'])
mae_future = mean_absolute_error(comparison_df['Actual Price [Euro/MWh]'], comparison_df['Predicted Price [Euro/MWh]'])
r2_future = r2_score(comparison_df['Actual Price [Euro/MWh]'], comparison_df['Predicted Price [Euro/MWh]'])
rmse_future = np.sqrt(mse_future)

print("Evaluation Metrics on Future Predictions:")
print(f"Mean Squared Error: {mse_future:.4f}")
print(f"Mean Absolute Error: {mae_future:.4f}")
print(f"R-squared: {r2_future:.4f}")
print(f"Root Mean Squared Error: {rmse_future:.4f}")
