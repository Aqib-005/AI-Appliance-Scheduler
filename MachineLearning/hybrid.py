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

data['Price Germany/Luxembourg [Euro/MWh]'] = data['Price Germany/Luxembourg [Euro/MWh]'].replace({',': ''}, regex=True).astype(float)
data['Total (grid consumption) [MWh]'] = data['Total (grid consumption) [MWh]'].replace({',': ''}, regex=True).astype(float)

data['Start date/time'] = pd.to_datetime(data['Start date/time'], dayfirst=True)
data['Year'] = data['Start date/time'].dt.year
data['Month'] = data['Start date/time'].dt.month
data['Day'] = data['Start date/time'].dt.day
data['Hour'] = data['Start date/time'].dt.hour
data['DayOfWeek'] = data['Start date/time'].dt.dayofweek

de_holidays = holidays.CountryHoliday('DE')
data['is_holiday'] = data['Start date/time'].dt.date.astype('datetime64[ns]').isin(list(de_holidays.keys())).astype(int)

# Lag features
data['Lag_Price_1'] = data['Price Germany/Luxembourg [Euro/MWh]'].shift(1)
data['Lag_Price_24'] = data['Price Germany/Luxembourg [Euro/MWh]'].shift(24)
data['Lag_Load_1'] = data['Total (grid consumption) [MWh]'].shift(1)
data['Lag_Load_24'] = data['Total (grid consumption) [MWh]'].shift(24)

# Rolling averages
data['Rolling_Price_24h'] = data['Price Germany/Luxembourg [Euro/MWh]'].rolling(window=24).mean()
data['Rolling_Load_24h'] = data['Total (grid consumption) [MWh]'].rolling(window=24).mean()
data['Rolling_Temp_24h'] = data['temperature_2m (°C)'].rolling(window=24).mean()
data['Rolling_Wind_24h'] = data['wind_speed_100m (km/h)'].rolling(window=24).mean()

data.dropna(inplace=True)

data.rename(columns={
    'temperature_2m (°C)': 'temperature_2m_C',
    'wind_speed_100m (km/h)': 'wind_speed_100m_kmh',
    'Total (grid consumption) [MWh]': 'Total_grid_consumption_MWh'
}, inplace=True)

# -----------------------
# 2. Prophet Model for Trend & Seasonality
# -----------------------
prophet_df = data[['Start date/time', 'Price Germany/Luxembourg [Euro/MWh]', 'temperature_2m_C', 'Total_grid_consumption_MWh', 'wind_speed_100m_kmh', 'is_holiday']].copy()
prophet_df.rename(columns={'Start date/time': 'ds', 'Price Germany/Luxembourg [Euro/MWh]': 'y'}, inplace=True)

m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True, changepoint_prior_scale=0.1, seasonality_mode='multiplicative', interval_width=0.95)
m.add_country_holidays(country_name='DE')
m.add_regressor('temperature_2m_C')
m.add_regressor('Total_grid_consumption_MWh')
m.add_regressor('wind_speed_100m_kmh')
m.add_regressor('is_holiday')
m.add_seasonality(name='8hour', period=8, fourier_order=3)

m.fit(prophet_df)
prophet_hist = m.predict(prophet_df)[['ds', 'trend']].rename(columns={'trend': 'prophet_trend'})
data = data.merge(prophet_hist, left_on='Start date/time', right_on='ds', how='left')
data['prophet_prediction'] = data['prophet_trend']

# -----------------------
# 3. Residual Modeling (Stacked Ensemble)
# -----------------------
data['residuals'] = data['Price Germany/Luxembourg [Euro/MWh]'] - data['prophet_prediction']
data['residuals'].fillna(0, inplace=True)

residual_features = ['temperature_2m_C', 'wind_speed_100m_kmh', 'Total_grid_consumption_MWh', 'Day', 'Hour', 'DayOfWeek', 'is_holiday', 'Lag_Price_1', 'Lag_Price_24', 'Rolling_Price_24h', 'Rolling_Temp_24h', 'Rolling_Wind_24h', 'Rolling_Load_24h', 'prophet_trend']
X_residual = data[residual_features]
y_residual = data['residuals']

scaler_residual = StandardScaler()
X_residual_scaled = scaler_residual.fit_transform(X_residual)

best_xgb_res = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42).fit(X_residual_scaled, y_residual)
best_lgb_res = lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42).fit(X_residual_scaled, y_residual)

data['ensemble_res_pred'] = (best_xgb_res.predict(X_residual_scaled) + best_lgb_res.predict(X_residual_scaled)) / 2
data['final_prediction'] = data['prophet_prediction'] + data['ensemble_res_pred']

# -----------------------
# 4. Compare Future Predictions to Actual Data
# -----------------------
forecast_start = "2024-10-01"
forecast_end = "2024-10-07"
future_predictions_df = pd.DataFrame({
    'Start date/time': pd.date_range(start=forecast_start, end=forecast_end, freq='h'),
    'Predicted Price [Euro/MWh]': np.random.uniform(50, 150, size=168)  # Replace with real predictions
})

actual_data = pd.DataFrame({
    'Start date/time': pd.date_range(start=forecast_start, end=forecast_end, freq='h'),
    'Actual Price [Euro/MWh]': np.random.uniform(40, 160, size=168)  # Replace with actual prices
})

comparison_df = future_predictions_df.merge(actual_data, on='Start date/time', how='left')

plt.figure(figsize=(12, 6))
plt.plot(comparison_df['Start date/time'], comparison_df['Predicted Price [Euro/MWh]'], label='Predicted Price', linestyle='dashed')
plt.plot(comparison_df['Start date/time'], comparison_df['Actual Price [Euro/MWh]'], label='Actual Price', alpha=0.7)
plt.legend()
plt.xlabel("Date")
plt.ylabel("Price [Euro/MWh]")
plt.title("Comparison of Predicted vs Actual Prices")
plt.xticks(rotation=45)
plt.show()
