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

# Feature selection for XGBoost
features = [
    'temperature_2m (°C)', 
    'wind_speed_100m (km/h)', 
    'Total (grid consumption) [MWh]', 
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
study.optimize(objective, n_trials=50)

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
prophet_df = data[['Start date/time', 'Price Germany/Luxembourg [Euro/MWh]']].rename(
    columns={'Start date/time': 'ds', 'Price Germany/Luxembourg [Euro/MWh]': 'y'}
)

# Configure Prophet with custom parameters
m = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=True,
    changepoint_prior_scale=0.05,  # Reduced flexibility to prevent overfitting
    seasonality_mode='additive',
    holidays_prior_scale=5,
    interval_width=0.95
)
m.add_country_holidays(country_name='DE')
m.fit(prophet_df)

# Generate trend predictions
future = m.make_future_dataframe(periods=0, freq='h')
forecast = m.predict(future)
data = data.merge(forecast[['ds', 'trend', 'yhat']], left_on='Start date/time', right_on='ds')
data.rename(columns={'trend': 'prophet_trend', 'yhat': 'prophet_prediction'}, inplace=True)

# 2. Residual Calculation
data['residuals'] = data['Price Germany/Luxembourg [Euro/MWh]'] - data['prophet_prediction']
data['residuals'] = data['residuals'].fillna(0).replace([np.inf, -np.inf], 0)

# Feature Engineering for Residual Model
residual_features = [
    'temperature_2m (°C)', 
    'wind_speed_100m (km/h)', 
    'Total (grid consumption) [MWh]', 
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
def create_future_features(last_date, periods=168):
    """Create future features for forecasting with proper lag handling"""
    future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=periods, freq='h')
    future_df = pd.DataFrame({'ds': future_dates})
    
    # Add external regressors (use the last available values)
    future_df['temperature_2m (°C)'] = data['temperature_2m (°C)'].ffill().iloc[-1]
    future_df['wind_speed_100m (km/h)'] = data['wind_speed_100m (km/h)'].ffill().iloc[-1]
    future_df['Total (grid consumption) [MWh]'] = data['Total (grid consumption) [MWh]'].ffill().iloc[-1]
    
    # Prophet forecast
    prophet_forecast = m.predict(future_df)
    
    # Rolling features
    for col in ['temperature_2m (°C)', 'wind_speed_100m (km/h)', 'Total (grid consumption) [MWh]']:
        # Calculate rolling mean for the last 24 hours
        future_df[f'Rolling_24h_{col}'] = future_df[col].rolling(window=24, min_periods=1, closed='left').mean()
    
    # Lagged prices using weekly pattern
    future_df['Lag_168_Price'] = data['Price Germany/Luxembourg [Euro/MWh]'].shift(168).ffill().values[-periods:]
    future_df['Lag_24_Price'] = data['Price Germany/Luxembourg [Euro/MWh]'].shift(24).ffill().values[-periods:]
    future_df['Lag_1_Price'] = data['Price Germany/Luxembourg [Euro/MWh]'].shift(1).ffill().values[-periods:]
    
    # Time features
    future_df['Hour'] = future_df['ds'].dt.hour
    future_df['Month'] = future_df['ds'].dt.month
    future_df['DayOfWeek'] = future_df['ds'].dt.dayofweek
    future_df['Day'] = future_df['ds'].dt.day  # Add 'Day' feature
    
    # Prophet trend
    future_df['prophet_trend'] = prophet_forecast['trend']
    
    # Add missing rolling features
    future_df['Rolling_Temp_24h'] = future_df['temperature_2m (°C)'].rolling(window=24, min_periods=1, closed='left').mean()
    future_df['Rolling_Wind_24h'] = future_df['wind_speed_100m (km/h)'].rolling(window=24, min_periods=1, closed='left').mean()
    future_df['Rolling_Load_24h'] = future_df['Total (grid consumption) [MWh]'].rolling(window=24, min_periods=1, closed='left').mean()
    
    # Add Lag_Price feature
    future_df['Lag_Price'] = data['Price Germany/Luxembourg [Euro/MWh]'].shift(1).ffill().values[-periods:]
    
    return future_df

# Generate future predictions with autoregressive features
last_date = data['Start date/time'].max()
future_df = create_future_features(last_date)

# Ensure all features are present
missing_features = [feature for feature in residual_features if feature not in future_df.columns]
if missing_features:
    raise ValueError(f"Missing features in future_df: {missing_features}")

# Scale and predict
future_X_residual = scaler_residual.transform(future_df[residual_features])
future_df['xgb_residual'] = best_xgb_residual.predict(future_X_residual)
future_df['final_price'] = future_df['prophet_trend'] + future_df['xgb_residual']

# Post-processing
future_df['final_price'] = np.where(future_df['final_price'] < 0, 0, future_df['final_price'])
future_df['final_price'] = future_df['final_price'].rolling(3, center=True).mean()

# Print future predictions
print("Predictions for the coming week:")
print(future_df[['ds', 'final_price']])