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
data['Price Germany/Luxembourg [Euro/MWh]'] = (
    data['Price Germany/Luxembourg [Euro/MWh]']
    .replace({',': ''}, regex=True)
    .astype(float)
)
data['Total (grid consumption) [MWh]'] = (
    data['Total (grid consumption) [MWh]']
    .replace({',': ''}, regex=True)
    .astype(float)
)

# Convert Start date/time to datetime and extract features
data['Start date/time'] = pd.to_datetime(data['Start date/time'], dayfirst=True)
data['Year'] = data['Start date/time'].dt.year
data['Month'] = data['Start date/time'].dt.month
data['Day'] = data['Start date/time'].dt.day
data['Hour'] = data['Start date/time'].dt.hour
data['DayOfWeek'] = data['Start date/time'].dt.dayofweek

# Create holiday flag for Germany
de_holidays = holidays.CountryHoliday('DE')
# Ensure that the Series uses the same index as the DataFrame:
data['is_holiday'] = pd.Series(data['Start date/time'].dt.date, index=data.index) \
    .isin(list(de_holidays.keys())).astype(int)

# Create lag features using original column names
data['Lag_Price_1'] = data['Price Germany/Luxembourg [Euro/MWh]'].shift(1)
data['Lag_Price_24'] = data['Price Germany/Luxembourg [Euro/MWh]'].shift(24)
data['Lag_Load_1'] = data['Total (grid consumption) [MWh]'].shift(1)
data['Lag_Load_24'] = data['Total (grid consumption) [MWh]'].shift(24)

# Create rolling averages using original column names
data['Rolling_Temp_24h'] = data['temperature_2m (°C)'].rolling(window=24).mean()
data['Rolling_Wind_24h'] = data['wind_speed_100m (km/h)'].rolling(window=24).mean()
data['Rolling_Load_24h'] = data['Total (grid consumption) [MWh]'].rolling(window=24).mean()

# Rename columns after creating lags/rollings
data.rename(columns={
    'temperature_2m (°C)': 'temperature_2m_C',
    'wind_speed_100m (km/h)': 'wind_speed_100m_kmh',
    'Total (grid consumption) [MWh]': 'Total_grid_consumption_MWh'
}, inplace=True)

data.dropna(inplace=True)

# -----------------------
# 2. Prophet Model for Trend & Seasonality
# -----------------------
prophet_df = data[['Start date/time', 'Price Germany/Luxembourg [Euro/MWh]',
                   'temperature_2m_C', 'Total_grid_consumption_MWh',
                   'wind_speed_100m_kmh', 'is_holiday']].copy()
prophet_df.rename(columns={'Start date/time': 'ds',
                           'Price Germany/Luxembourg [Euro/MWh]': 'y'}, inplace=True)

m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True,
            changepoint_prior_scale=0.1, seasonality_mode='multiplicative', interval_width=0.95)
m.add_country_holidays(country_name='DE')
m.add_regressor('temperature_2m_C')
m.add_regressor('Total_grid_consumption_MWh')
m.add_regressor('wind_speed_100m_kmh')
m.add_regressor('is_holiday')
m.add_seasonality(name='8hour', period=8, fourier_order=3)
m.fit(prophet_df)

# Obtain Prophet's trend forecast as the base prediction
prophet_hist = m.predict(prophet_df)[['ds', 'trend']].rename(columns={'trend': 'prophet_trend'})
data = data.merge(prophet_hist, left_on='Start date/time', right_on='ds', how='left')
data['prophet_prediction'] = data['prophet_trend']

# -----------------------
# 3. Residual Modeling (Stacked Ensemble)
# -----------------------
data['residuals'] = data['Price Germany/Luxembourg [Euro/MWh]'] - data['prophet_prediction']
data['residuals'] = data['residuals'].fillna(0)

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
                                subsample=0.8, colsample_bytree=0.8, random_state=42)
best_xgb_res.fit(X_residual_scaled, y_residual)

best_lgb_res = lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                                subsample=0.8, colsample_bytree=0.8, random_state=42)
best_lgb_res.fit(X_residual_scaled, y_residual)

data['ensemble_res_pred'] = (best_xgb_res.predict(X_residual_scaled) + best_lgb_res.predict(X_residual_scaled)) / 2
data['final_prediction'] = data['prophet_prediction'] + data['ensemble_res_pred']

# -----------------------
# 4. Future Predictions & Comparison with Hardcoded Actual Data
# -----------------------
# Define forecast period (October 1–7, 2024)
forecast_start = "2024-10-01 00:00:00"
forecast_end = "2024-10-07 23:00:00"
future_dates = pd.date_range(start=forecast_start, end=forecast_end, freq='h')

# Create future DataFrame and set time features
future_data = pd.DataFrame(index=future_dates)
future_data['Start date/time'] = future_data.index
future_data['Year'] = future_data.index.year
future_data['Month'] = future_data.index.month
future_data['Day'] = future_data.index.day
future_data['Hour'] = future_data.index.hour
future_data['DayOfWeek'] = future_data.index.dayofweek
# Set holiday flag properly (make sure to pass the index)
future_data['is_holiday'] = pd.Series(future_data.index.date, index=future_data.index) \
    .isin(list(de_holidays.keys())).astype(int)

# Initialize lag features using last observed values from historical data
future_data['Lag_Price_1'] = data['Price Germany/Luxembourg [Euro/MWh]'].iloc[-1]
future_data['Lag_Price_24'] = data['Price Germany/Luxembourg [Euro/MWh]'].shift(24).iloc[-1]

# Forecast weather and grid consumption features using Prophet.
def forecast_feature(df, feature, periods=7*24):
    # Use the renamed column from data
    feature_data = df[['Start date/time', feature]].rename(
        columns={'Start date/time': 'ds', feature: 'y'}
    )
    model = Prophet(daily_seasonality=True)
    model.fit(feature_data)
    future = model.make_future_dataframe(periods=periods, freq='h')
    forecast = model.predict(future)
    return forecast['yhat'].tail(periods).values

# Forecast features using renamed columns
future_data['temperature_2m_C'] = forecast_feature(data, 'temperature_2m_C')
future_data['wind_speed_100m_kmh'] = forecast_feature(data, 'wind_speed_100m_kmh')
future_data['Total_grid_consumption_MWh'] = forecast_feature(data, 'Total_grid_consumption_MWh')

# Add rolling averages for future data
future_data['Rolling_Temp_24h'] = future_data['temperature_2m_C'].rolling(window=24, min_periods=1).mean()
future_data['Rolling_Wind_24h'] = future_data['wind_speed_100m_kmh'].rolling(window=24, min_periods=1).mean()
future_data['Rolling_Load_24h'] = future_data['Total_grid_consumption_MWh'].rolling(window=24, min_periods=1).mean()
future_data = future_data.bfill()  # backfill to handle any NaNs

# Build a separate DataFrame for Prophet future trend prediction with all regressors
future_prophet_df = pd.DataFrame({
    'ds': future_dates,
    'temperature_2m_C': forecast_feature(data, 'temperature_2m_C'),
    'Total_grid_consumption_MWh': forecast_feature(data, 'Total_grid_consumption_MWh'),
    'wind_speed_100m_kmh': forecast_feature(data, 'wind_speed_100m_kmh'),
    'is_holiday': np.repeat(data['is_holiday'].iloc[-1], len(future_dates))
})
prophet_future = m.predict(future_prophet_df)[['ds', 'trend']].rename(columns={'trend': 'prophet_trend'})

# Merge Prophet trend forecast into future_data
future_data = future_data.merge(prophet_future, left_index=True, right_on='ds', how='left')
future_data.drop(columns=['ds'], inplace=True)

# Check that all residual features are available in future_data
missing_feats = [f for f in residual_features if f not in future_data.columns]
if missing_feats:
    raise ValueError(f"Missing features in future_data: {missing_feats}")

# Prepare features for residual prediction and scale them
future_X = scaler_residual.transform(future_data[residual_features])
future_residual_pred = best_xgb_res.predict(future_X) + best_lgb_res.predict(future_X)

# Final hybrid future prediction: Prophet trend + half the ensemble residual
future_final_pred = future_data['prophet_trend'] + future_residual_pred / 2

future_predictions_df = pd.DataFrame({
    'Start date/time': future_dates,
    'Predicted Price [Euro/MWh]': future_final_pred
})

print("Predictions for the coming week:")
print(future_predictions_df.head())

# -----------------------
# 5. Compare Future Predictions to Hardcoded Actual Data
# -----------------------
# Hardcode actual data for the forecast period
actual_data = pd.DataFrame({
    'Start date/time': pd.date_range(start=forecast_start, end=forecast_end, freq='h'),
    'Actual Price [Euro/MWh]': (
        [3.21, 0.07, 0.05, 0.02, 0.09, 6.80, 63.96, 103.35, 114.98, 100.41, 
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
         229.53, 121.98, 99.93, 91.91, 79.12]
    )[:168]
})
actual_data['Start date/time'] = pd.to_datetime(actual_data['Start date/time'])

# Merge predictions and actual data on 'Start date/time'
comparison_df = pd.merge(future_predictions_df, actual_data, on='Start date/time', how='left', suffixes=('_predicted', '_actual'))

mse = mean_squared_error(comparison_df['Actual Price [Euro/MWh]'], comparison_df['Predicted Price [Euro/MWh]'])
mae = mean_absolute_error(comparison_df['Actual Price [Euro/MWh]'], comparison_df['Predicted Price [Euro/MWh]'])
r2 = r2_score(comparison_df['Actual Price [Euro/MWh]'], comparison_df['Predicted Price [Euro/MWh]'])
rmse = np.sqrt(mse)

print("Evaluation Metrics on Future Predictions:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R-squared: {r2:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")

plt.figure(figsize=(12, 6))
plt.plot(comparison_df['Start date/time'], comparison_df['Predicted Price [Euro/MWh]'], label='Predicted', marker='o')
plt.plot(comparison_df['Start date/time'], comparison_df['Actual Price [Euro/MWh]'], label='Actual', marker='x')
plt.title('Predicted vs Actual Hourly Prices')
plt.xlabel('Date/Time')
plt.ylabel('Price [Euro/MWh]')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
comparison_df['Residuals'] = comparison_df['Actual Price [Euro/MWh]'] - comparison_df['Predicted Price [Euro/MWh]']
plt.plot(comparison_df['Start date/time'], comparison_df['Residuals'], marker='o', color='red')
plt.axhline(0, color='black', linestyle='--')
plt.title('Residuals (Actual - Predicted)')
plt.xlabel('Date/Time')
plt.ylabel('Residuals [Euro/MWh]')
plt.grid(True)
plt.show()
