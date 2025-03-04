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

#############################
# 1. Load and preprocess historical data (training data)
#############################

# Load historical data
hist_data = pd.read_csv("data/merged-data.csv")

# Clean column names and fix encoding issues
hist_data.columns = hist_data.columns.str.strip().str.replace("Ã¸", "°")

# Rename columns for consistency
hist_data.rename(columns={
    'start date/time': 'StartDateTime',
    'day_price': 'Price',
    'grid_load': 'total-consumption'
}, inplace=True)

# Define numeric columns (including weather code if applicable)
numeric_cols_hist = ['Price', 'total-consumption', 'temperature_2m',
                     'precipitation (mm)', 'rain (mm)', 'snowfall (cm)',
                     'wind_speed_100m (km/h)', 'relative_humidity_2m (%)',
                     'weather_code (wmo code)']

for col in numeric_cols_hist:
    hist_data[col] = hist_data[col].replace({',': ''}, regex=True)
    hist_data[col] = pd.to_numeric(hist_data[col], errors='coerce')

# Convert date column and sort
hist_data['StartDateTime'] = pd.to_datetime(hist_data['StartDateTime'], dayfirst=True)
hist_data = hist_data.sort_values('StartDateTime').reset_index(drop=True)

# Forward fill missing values
hist_data.ffill(inplace=True)

# Create time features
hist_data['Year'] = hist_data['StartDateTime'].dt.year
hist_data['Month'] = hist_data['StartDateTime'].dt.month
hist_data['Day'] = hist_data['StartDateTime'].dt.day
hist_data['Hour'] = hist_data['StartDateTime'].dt.hour
hist_data['DayOfWeek'] = hist_data['StartDateTime'].dt.dayofweek

# Create lag feature for Price
hist_data['Lag_Price'] = hist_data['Price'].shift(1)

# Compute rolling averages (24-hour window)
hist_data['Rolling_Temp_24h'] = hist_data['temperature_2m'].rolling(window=24).mean()
hist_data['Rolling_Wind_24h'] = hist_data['wind_speed_100m (km/h)'].rolling(window=24).mean()
hist_data['Rolling_Load_24h'] = hist_data['total-consumption'].rolling(window=24).mean()

# Drop any rows with NA after feature engineering
hist_data = hist_data.dropna()

#############################
# 2. Define features and target for training
#############################

features = [
    'temperature_2m',
    'precipitation (mm)',
    'rain (mm)',
    'snowfall (cm)',
    'weather_code (wmo code)',  # if available
    'wind_speed_100m (km/h)',
    'relative_humidity_2m (%)',
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

X = hist_data[features]
y = hist_data[target]

# Normalize features based on historical data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Set up time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)

#############################
# 3. Hyperparameter tuning using Optuna
#############################

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

#############################
# 4. Train final XGBoost model on historical data
#############################

best_xgb = xgb.XGBRegressor(**best_params, random_state=42)
best_xgb.fit(X_scaled, y)

# Evaluate on the last fold for reference
train_index, test_index = list(tscv.split(X_scaled))[-1]
X_train, X_test = X_scaled[train_index], X_scaled[test_index]
y_train, y_test = y.iloc[train_index], y.iloc[test_index]
y_pred = best_xgb.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Evaluation Metrics on Historical Data:")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R²: {r2}")

# Plot feature importance
feature_importance = best_xgb.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=True)
plt.figure(figsize=(10,6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()

# --------------------------
# 5. Future Forecasting using future-data.csv (Revised Dynamic Approach)
# --------------------------

# Read future data
try:
    future_df = pd.read_csv('data/future-data.csv', parse_dates=['start date/time'])
    future_df.columns = future_df.columns.str.strip()
    print("Stripped future data columns:", future_df.columns.tolist())
except Exception as e:
    print(f"Error reading future-data.csv: {e}")
    exit()

future_df.dropna(subset=['start date/time'], inplace=True)
future_df.sort_values('start date/time', inplace=True)
future_df['start date/time'] = pd.to_datetime(future_df['start date/time'], dayfirst=True, errors='coerce')
if future_df['start date/time'].isna().all():
    print("Error: Could not parse any dates in 'start date/time'.")
    exit()

# Rename columns for consistency
future_df.rename(columns={
    'start date/time': 'StartDateTime',
    'grid_load': 'total-consumption'
}, inplace=True)

# Ensure Price is set to NaN for future predictions
future_df['Price'] = np.nan

# Process numeric columns in future data similar to historical
numeric_cols_future = ['total-consumption', 'temperature_2m', 'precipitation (mm)',
                       'rain (mm)', 'snowfall (cm)', 'wind_speed_100m (km/h)',
                       'relative_humidity_2m (%)', 'weather_code (wmo code)']
for col in numeric_cols_future:
    if col in future_df.columns:
        future_df[col] = future_df[col].astype(str).str.replace(',', '').str.strip()
        future_df[col] = pd.to_numeric(future_df[col], errors='coerce')
future_df[numeric_cols_future] = future_df[numeric_cols_future].ffill().bfill()

# Define forecast period (using dayfirst=True because your dates are dd/mm/YYYY)
forecast_start_dt = pd.to_datetime('06/01/2025 00:00:00', dayfirst=True)
forecast_end_dt   = pd.to_datetime('12/01/2025 23:00:00', dayfirst=True)

# Filter future_df for the forecast period
future_df = future_df[(future_df['StartDateTime'] >= forecast_start_dt) & 
                      (future_df['StartDateTime'] <= forecast_end_dt)].copy()
if future_df.empty:
    print("No future data available for the forecast period after processing.")
    exit()

future_df.reset_index(drop=True, inplace=True)

# --- Set up dynamic feature calculation for recursive forecasting ---
# Get the last 23 rows from historical data to compute 24-hour rolling averages.
hist_tail = hist_data.tail(23)
temp_history = hist_tail['temperature_2m'].tolist()
wind_history = hist_tail['wind_speed_100m (km/h)'].tolist()
load_history = hist_tail['total-consumption'].tolist()

# For Lag_Price, get the last historical Price.
last_price = hist_data['Price'].iloc[-1]

# List to store predictions
predictions = []

for idx, row in future_df.iterrows():
    # Direct features from future row
    temperature    = row['temperature_2m']
    precipitation  = row['precipitation (mm)']
    rain           = row['rain (mm)']
    snowfall       = row['snowfall (cm)']
    weather_code   = row['weather_code (wmo code)']
    wind_speed     = row['wind_speed_100m (km/h)']
    rel_humidity   = row['relative_humidity_2m (%)']
    total_consumption = row['total-consumption']
    
    # Time features from StartDateTime
    dt = row['StartDateTime']
    Day = dt.day
    Hour = dt.hour
    DayOfWeek = dt.dayofweek
    
    # Update rolling lists with current future values
    temp_history.append(temperature)
    wind_history.append(wind_speed)
    load_history.append(total_consumption)
    if len(temp_history) > 24: temp_history.pop(0)
    if len(wind_history) > 24: wind_history.pop(0)
    if len(load_history) > 24: load_history.pop(0)
    
    # Ensure we have enough values for rolling features
    if len(temp_history) < 24 or len(wind_history) < 24 or len(load_history) < 24:
        print(f"Skipping prediction for {dt} due to insufficient rolling data.")
        predictions.append(np.nan)
        continue
    
    Rolling_Temp_24h = np.mean(temp_history)
    Rolling_Wind_24h = np.mean(wind_history)
    Rolling_Load_24h = np.mean(load_history)
    
    # ---- FIX: Compute Lag_Price with tolerance ----
    lag_time = dt - pd.Timedelta(hours=1)
    # Use a 1-minute tolerance window to catch the historical timestamp
    lag_row = hist_data[(hist_data['StartDateTime'] >= lag_time - pd.Timedelta(minutes=1)) &
                        (hist_data['StartDateTime'] <= lag_time + pd.Timedelta(minutes=1))]
    if not lag_row.empty:
        Lag_Price = lag_row['Price'].values[0]
    else:
        # If not found in historical, then if we're in future use previous prediction, else fallback to last historical price
        Lag_Price = predictions[-1] if idx > 0 else last_price
    # -----------------------------------------------
    
    # Assemble feature vector in the same order as training:
    feature_vector = [
        temperature, precipitation, rain, snowfall, weather_code,
        wind_speed, rel_humidity, total_consumption, Day, Hour, DayOfWeek,
        Rolling_Temp_24h, Rolling_Wind_24h, Rolling_Load_24h, Lag_Price
    ]
    
    # Skip prediction if any feature is NaN
    if any(np.isnan(x) for x in feature_vector):
        print(f"Skipping prediction for {dt} due to NaN in features.")
        predictions.append(np.nan)
        continue
    
    # Scale and predict.
    feature_vector_scaled = scaler.transform([feature_vector])
    pred_price = best_xgb.predict(feature_vector_scaled)[0]
    predictions.append(pred_price)
    
# Append predictions to future_df.
future_df['Predicted Price [Euro/MWh]'] = predictions

print("\nPredictions for 06-Jan-2025 to 12-Jan-2025:")
print(future_df[['StartDateTime', 'Predicted Price [Euro/MWh]']].to_string(index=False))
