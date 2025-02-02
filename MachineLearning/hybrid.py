import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
from prophet import Prophet
import optuna
from optuna.samplers import TPESampler
import holidays

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
data['DayOfWeek'] = data['Start date/time'].dt.dayofweek

# Create lagged features for target variable
data['Lag_Price'] = data['Price Germany/Luxembourg [Euro/MWh]'].shift(1)

# Add rolling averages for features
data['Rolling_Temp_24h'] = data['temperature_2m (°C)'].rolling(window=24).mean()
data['Rolling_Wind_24h'] = data['wind_speed_100m (km/h)'].rolling(window=24).mean()
data['Rolling_Load_24h'] = data['Total (grid consumption) [MWh]'].rolling(window=24).mean()

# Drop rows with missing values after lagging and rolling
data = data.dropna()

# Rename features to remove special characters
data = data.rename(columns={
    'temperature_2m (°C)': 'temperature_2m_C',
    'wind_speed_100m (km/h)': 'wind_speed_100m_kmh',
    'Total (grid consumption) [MWh]': 'Total_grid_consumption_MWh'
})

# Feature selection for XGBoost
features = [
    'temperature_2m_C', 
    'wind_speed_100m_kmh', 
    'Total_grid_consumption_MWh', 
    'Day', 
    'Hour', 
    'DayOfWeek',
    'Rolling_Temp_24h',
    'Rolling_Wind_24h',
    'Rolling_Load_24h',
    'Lag_Price'
]
target = 'Price Germany/Luxembourg [Euro/MWh]'

# Subset the data
X = data[features]
y = data[target]

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# TimeSeriesSplit for cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Define the objective function for Optuna
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

        # Initialize the model
        model = xgb.XGBRegressor(**params)

        # Fit the model with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        y_pred = model.predict(X_val)
        scores.append(mean_squared_error(y_val, y_pred))

    return np.mean(scores)

# Run Bayesian Optimization with Optuna
study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=100)

# Get the best parameters
best_params = study.best_params
print(f"Best Parameters: {best_params}")

# Train the final XGBoost model with the best parameters
best_xgb = xgb.XGBRegressor(**best_params, random_state=42)
best_xgb.fit(X_scaled, y)

# Evaluate the model using the last fold of TimeSeriesSplit
train_index, test_index = list(tscv.split(X_scaled))[-1]
X_train, X_test = X_scaled[train_index], X_scaled[test_index]
y_train, y_test = y.iloc[train_index], y.iloc[test_index]

y_pred = best_xgb.predict(X_test)

# Evaluation metrics
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

# Hybrid Model: Prophet + XGBoost ---------------------------------------------

# 1. Prophet Model for Trend and Seasonality
prophet_df = data.rename(columns={'Start date/time': 'ds'})[['ds', 'Price Germany/Luxembourg [Euro/MWh]', 
                  'temperature_2m_C', 'Total_grid_consumption_MWh', 
                  'wind_speed_100m_kmh']].rename(
    columns={'Price Germany/Luxembourg [Euro/MWh]': 'y'}
)

# Configure Prophet
m = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=True,
    changepoint_prior_scale=0.02,
    seasonality_mode='multiplicative',
    holidays_prior_scale=10,
    interval_width=0.95
)
m.add_country_holidays(country_name='DE')
m.add_regressor('temperature_2m_C')
m.add_regressor('Total_grid_consumption_MWh')
m.add_regressor('wind_speed_100m_kmh')

# Fit Prophet model
m.fit(prophet_df)

# Generate future dataframe (FIX 1: Use 'h' instead of 'H')
future = m.make_future_dataframe(periods=168, freq='h')

# Merge regressors using original datetime column (FIX 2: Correct column reference)
future = future.merge(
    data[['Start date/time', 'temperature_2m_C', 'Total_grid_consumption_MWh', 'wind_speed_100m_kmh']].rename(
        columns={'Start date/time': 'ds'}
    ),
    on='ds',
    how='left'
)

# Forward-fill missing regressor values
for col in ['temperature_2m_C', 'Total_grid_consumption_MWh', 'wind_speed_100m_kmh']:
    future[col] = future[col].fillna(method='ffill')

# Generate Prophet forecast
forecast = m.predict(future)

# Merge Prophet results into future_df and RENAME 'trend' to 'prophet_trend'
future_df = future.merge(
    forecast[['ds', 'trend']], 
    on='ds',
    how='left'
).rename(columns={'trend': 'prophet_trend'})  # Critical fix here

# Merge Prophet results back to main dataframe
data = data.merge(
    forecast[['ds', 'trend', 'yhat']], 
    left_on='Start date/time', 
    right_on='ds',
    how='left'
)
data.rename(columns={'trend': 'prophet_trend', 'yhat': 'prophet_prediction'}, inplace=True)

# 2. Residual Calculation
data['residuals'] = data['Price Germany/Luxembourg [Euro/MWh]'] - data['prophet_prediction']
data['residuals'] = data['residuals'].fillna(0).replace([np.inf, -np.inf], 0)

# Feature Engineering for Residual Model
residual_features = [
    'temperature_2m_C', 
    'wind_speed_100m_kmh', 
    'Total_grid_consumption_MWh', 
    'Day', 
    'Hour', 
    'DayOfWeek',
    'Rolling_Temp_24h',
    'Rolling_Wind_24h',
    'Rolling_Load_24h',
    'Lag_Price',
    'prophet_trend'
]

# Subset the data for residual model
X_residual = data[residual_features]
y_residual = data['residuals']

# Normalize the features
scaler_residual = StandardScaler()
X_residual_scaled = scaler_residual.fit_transform(X_residual)

# Train the residual model
best_xgb_residual = xgb.XGBRegressor(**best_params, random_state=42)
best_xgb_residual.fit(X_residual_scaled, y_residual)

# Generate hybrid predictions
data['xgb_residual_pred'] = best_xgb_residual.predict(X_residual_scaled)
data['hybrid_prediction'] = data['prophet_prediction'] + data['xgb_residual_pred']

# Evaluate hybrid model
rmse_hybrid = np.sqrt(mean_squared_error(data['Price Germany/Luxembourg [Euro/MWh]'], data['hybrid_prediction']))
mae_hybrid = mean_absolute_error(data['Price Germany/Luxembourg [Euro/MWh]'], data['hybrid_prediction'])
r2_hybrid = r2_score(data['Price Germany/Luxembourg [Euro/MWh]'], data['hybrid_prediction'])

print("Hybrid Model Evaluation Metrics:")
print(f"RMSE: {rmse_hybrid}")
print(f"MAE: {mae_hybrid}")
print(f"R²: {r2_hybrid}")

# Future Predictions ----------------------------------------------------------
def create_future_features(last_date, data, periods=168):
    """Create future features for forecasting with proper regressor handling"""
    future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=periods, freq='H')
    future_df = pd.DataFrame({'ds': future_dates})
    
    # Carry forward the last known values of regressors
    last_values = data[['temperature_2m_C', 'Total_grid_consumption_MWh', 
                       'wind_speed_100m_kmh']].iloc[-1]
    
    for col in ['temperature_2m_C', 'Total_grid_consumption_MWh', 'wind_speed_100m_kmh']:
        future_df[col] = last_values[col]
    
    # Generate Prophet forecast for future dates
    prophet_forecast = m.predict(future_df)
    
    # Merge Prophet components
    future_df = future_df.merge(prophet_forecast[['ds', 'trend']], on='ds')
    
    # Add time features
    future_df['Hour'] = future_df['ds'].dt.hour
    future_df['Day'] = future_df['ds'].dt.day
    future_df['DayOfWeek'] = future_df['ds'].dt.dayofweek
    
    # Add rolling features (carry forward last known values)
    for col in ['Rolling_Temp_24h', 'Rolling_Wind_24h', 'Rolling_Load_24h']:
        future_df[col] = data[col].iloc[-1]
    
    # Initialize lagged price with the last known value
    future_df['Lag_Price'] = data['Price Germany/Luxembourg [Euro/MWh]'].iloc[-1]
    
    return future_df

# Generate future predictions
last_date = data['Start date/time'].max()
future_df = create_future_features(last_date, data)

# Ensure all features are present
missing_features = [feature for feature in residual_features if feature not in future_df.columns]
if missing_features:
    raise ValueError(f"Missing features in future_df: {missing_features}")

# Scale and predict
future_X_residual = scaler_residual.transform(future_df[residual_features])
future_df['xgb_residual'] = best_xgb_residual.predict(future_X_residual)
future_df['final_price'] = future_df['trend'] + future_df['xgb_residual']

# Post-processing
future_df['final_price'] = np.where(future_df['final_price'] < 0, 0, future_df['final_price'])
future_df['final_price'] = future_df['final_price'].rolling(3, center=True).mean()

# Print future predictions
print("Predictions for the coming week:")
print(future_df[['ds', 'final_price']])

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
comparison_df = pd.merge(future_df[['ds', 'final_price']], actual_data, left_on='ds', right_on='Start date/time', how='left')

# Handle NaN values in comparison_df
comparison_df = comparison_df.dropna(subset=['Actual Price [Euro/MWh]', 'final_price'])

# Calculate evaluation metrics
if not comparison_df.empty:
    mse = mean_squared_error(comparison_df['Actual Price [Euro/MWh]'], comparison_df['final_price'])
    mae = mean_absolute_error(comparison_df['Actual Price [Euro/MWh]'], comparison_df['final_price'])
    r2 = r2_score(comparison_df['Actual Price [Euro/MWh]'], comparison_df['final_price'])
    rmse = np.sqrt(mse)

    print("Hybrid Model Evaluation Metrics:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R-squared: {r2:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")

    # Plot predicted vs actual prices
    plt.figure(figsize=(12, 6))
    plt.plot(comparison_df['ds'], comparison_df['final_price'], label='Predicted', marker='o')
    plt.plot(comparison_df['ds'], comparison_df['Actual Price [Euro/MWh]'], label='Actual', marker='x')
    plt.title('Hybrid Model: Predicted vs Actual Hourly Prices')
    plt.xlabel('Date/Time')
    plt.ylabel('Price [Euro/MWh]')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot residuals
    comparison_df['Residuals'] = comparison_df['Actual Price [Euro/MWh]'] - comparison_df['final_price']
    plt.figure(figsize=(12, 6))
    plt.plot(comparison_df['ds'], comparison_df['Residuals'], marker='o', color='red')
    plt.axhline(0, color='black', linestyle='--')
    plt.title('Hybrid Model: Residuals (Actual - Predicted)')
    plt.xlabel('Date/Time')
    plt.ylabel('Residuals [Euro/MWh]')
    plt.grid(True)
    plt.show()
else:
    print("No overlapping data for comparison")