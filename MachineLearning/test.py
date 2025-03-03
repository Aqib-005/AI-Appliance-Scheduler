import numpy as np
import pandas as pd
import pywt
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

# --------------------------
# 1. Data Loading & Preprocessing
# --------------------------
df = pd.read_csv('data/merged-data.csv', parse_dates=['start date/time'])
df['start date/time'] = pd.to_datetime(df['start date/time'], errors='coerce')
df.dropna(subset=['start date/time'], inplace=True)
df.sort_values('start date/time', inplace=True)

# Convert key numeric columns to numeric types
numeric_cols = ['temperature_2m', 'precipitation (mm)', 'wind_speed_100m (km/h)',
                'relative_humidity_2m (%)', 'grid_load']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Convert target column to numeric
df['day_price'] = pd.to_numeric(df['day_price'], errors='coerce')

# Fill missing values in essential columns
essential_cols = ['day_price'] + numeric_cols
df[essential_cols] = df[essential_cols].ffill().bfill()

# --------------------------
# 2. Wavelet Denoising on Target
# --------------------------
def wavelet_denoise(data, wavelet='db4', level=3):
    coeff = pywt.wavedec(data, wavelet, mode='per', level=level)
    sigma = (1 / 0.6745) * np.median(np.abs(coeff[-level] - np.median(coeff[-level])))
    uthresh = sigma * np.sqrt(2 * np.log(len(data)))
    coeff[1:] = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeff[1:]]
    return pywt.waverec(coeff, wavelet, mode='per')[:len(data)]

df['denoised_price'] = wavelet_denoise(df['day_price'])

# --------------------------
# 3. Feature Engineering
# --------------------------

# Create lag features
for lag in [24, 168]:
    df[f'price_lag_{lag}'] = df['denoised_price'].shift(lag)
    df[f'load_lag_{lag}'] = df['grid_load'].shift(lag)

# Add rolling window features (24-hour rolling mean and std)
df['price_roll_mean_24'] = df['denoised_price'].rolling(window=24).mean()
df['price_roll_std_24'] = df['denoised_price'].rolling(window=24).std()

# Drop the first 168 rows to ensure lag features are available
df = df.iloc[168:]

# Extract datetime features
df['Hour'] = df['start date/time'].dt.hour
df['day of week'] = df['start date/time'].dt.dayofweek

# Create cyclical features
df['hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
df['day_of_week_sin'] = np.sin(2 * np.pi * df['day of week'] / 7)

# Fill remaining NaNs in lag and rolling features
extra_cols = [f'price_lag_24', f'price_lag_168', f'load_lag_24', 'price_roll_mean_24', 'price_roll_std_24']
df[extra_cols] = df[extra_cols].ffill().bfill()
df[essential_cols] = df[essential_cols].ffill().bfill()

# Apply a log-transformation to the denoised target (with a small epsilon for stability)
epsilon = 1e-8
df['log_denoised_price'] = np.log(df['denoised_price'] + epsilon)


# --------------------------
# 4. Temporal Data Split
# --------------------------
# Define temporal cutoff
split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx]
test = df.iloc[split_idx:]

print(f"Training samples: {len(train)}, Testing samples: {len(test)}")

# --------------------------
# 5. Model Training + Bayesian Optimization with Optuna
# --------------------------
features = [
    'temperature_2m', 'precipitation (mm)', 'wind_speed_100m (km/h)',
    'relative_humidity_2m (%)', 'grid_load', 'price_lag_24', 'price_lag_168',
    'load_lag_24', 'hour_sin', 'hour_cos', 'day_of_week_sin'
]

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.05, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
        'verbosity': 0
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    rmses = []
    
    for train_idx, val_idx in tscv.split(train):
        X_train, X_val = train.iloc[train_idx][features], train.iloc[val_idx][features]
        y_train, y_val = train.iloc[train_idx]['denoised_price'], train.iloc[val_idx]['denoised_price']
        
        model = XGBRegressor(**params, early_stopping_rounds=50)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        rmses.append(rmse)
    
    return np.mean(rmses)

# Increase the number of trials for more robust hyperparameter tuning
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=200)
print("Best hyperparameters:", study.best_trial.params)

# --------------------------
# 6. Evaluation
# --------------------------
best_params = study.best_trial.params
final_model = XGBRegressor(**best_params, early_stopping_rounds=50, verbosity=0)
final_model.fit(train[features], train['denoised_price'],
                eval_set=[(test[features], test['denoised_price'])],
                verbose=False)

test_preds = final_model.predict(test[features])
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape

test_preds = final_model.predict(test[features])
mae, rmse, mape = calculate_metrics(test['denoised_price'], test_preds)
print(f"Test MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

# --------------------------
# 7. Future Prediction with Actual Features
# --------------------------
try:
    # Use the correct future-data CSV file if available
    future_df = pd.read_csv('data/future-data.csv', parse_dates=['start date/time'])
    # Don't overwrite future_df's dates with historical df's dates
    # future_df['start date/time'] = pd.to_datetime(df['start date/time'], errors='coerce')
    future_df.dropna(subset=['start date/time'], inplace=True)
    future_df.sort_values('start date/time', inplace=True)
    
    # Strip leading/trailing spaces from column names
    future_df.columns = future_df.columns.str.strip()
    print("Stripped column names:", future_df.columns.tolist())
    
    if 'start date/time' not in future_df.columns:
        print("Error: 'start date/time' column not found in the CSV file.")
        print("Available columns:", future_df.columns.tolist())
        exit()
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit()

# Let Pandas auto-parse if dates are in ISO format; remove forced format if not needed
future_df['start date/time'] = pd.to_datetime(future_df['start date/time'], dayfirst=True, errors='coerce')

# Drop unused columns if necessary
if 'day_price' in future_df.columns:
    future_df = future_df.drop(columns=['day_price'])

# Process numeric columns
numeric_cols = [
    'temperature_2m', 'precipitation (mm)', 'wind_speed_100m (km/h)',
    'relative_humidity_2m (%)', 'grid_load'
]

for col in numeric_cols:
    if col in future_df.columns:
        future_df[col] = (
            future_df[col]
            .astype(str)
            .str.replace(',', '')  # Remove commas
            .str.strip()
            .replace('', np.nan)
            .astype(float)
        )
    else:
        print(f"Warning: Column '{col}' not found in future_df.")

# Fill missing values
future_df[numeric_cols] = future_df[numeric_cols].ffill().bfill()

# Create features
future_df['Hour'] = future_df['start date/time'].dt.hour
future_df['day of week'] = future_df['start date/time'].dt.dayofweek
future_df['hour_sin'] = np.sin(2 * np.pi * future_df['Hour'] / 24)
future_df['hour_cos'] = np.cos(2 * np.pi * future_df['Hour'] / 24)
future_df['day_of_week_sin'] = np.sin(2 * np.pi * future_df['day of week'] / 7)

# Initialize lag columns
for col in ['price_lag_24', 'price_lag_168', 'load_lag_24']:
    future_df[col] = np.nan

# Prediction loop
features = [
    'temperature_2m', 'precipitation (mm)', 'wind_speed_100m (km/h)',
    'relative_humidity_2m (%)', 'grid_load', 'price_lag_24', 'price_lag_168',
    'load_lag_24', 'hour_sin', 'hour_cos', 'day_of_week_sin'
]

# Initialize the predicted_price column
future_df['predicted_price'] = np.nan

for idx in future_df.index:
    current_time = future_df.loc[idx, 'start date/time']
    
    # Calculate lag times
    lag_24_time = current_time - pd.Timedelta(hours=24)
    lag_168_time = current_time - pd.Timedelta(hours=168)
    
    # Set price_lag_24 and load_lag_24
    if lag_24_time <= df['start date/time'].max():
        # If the lag time falls within historical data
        lag_24_row = df[df['start date/time'] == lag_24_time]
        if not lag_24_row.empty:
            future_df.loc[idx, 'price_lag_24'] = lag_24_row['denoised_price'].values[0]
            future_df.loc[idx, 'load_lag_24'] = lag_24_row['grid_load'].values[0]
        else:
            future_df.loc[idx, 'price_lag_24'] = df['denoised_price'].iloc[-1]
            future_df.loc[idx, 'load_lag_24'] = df['grid_load'].iloc[-1]
    else:
        # If the lag time is in the future region, check if a predicted price exists
        lag_24_row = future_df[future_df['start date/time'] == lag_24_time]
        if not lag_24_row.empty and pd.notna(lag_24_row.iloc[0].get('predicted_price', np.nan)):
            future_df.loc[idx, 'price_lag_24'] = lag_24_row['predicted_price'].values[0]
            future_df.loc[idx, 'load_lag_24'] = lag_24_row['grid_load'].values[0]
        else:
            # Try to use the most recent predicted price from earlier rows, if available
            valid_preds = future_df.loc[:idx-1, 'predicted_price']
            if valid_preds.notna().any():
                last_pred = valid_preds.iloc[-1]
                future_df.loc[idx, 'price_lag_24'] = last_pred
            else:
                # Fallback to historical value if no prediction exists yet
                future_df.loc[idx, 'price_lag_24'] = df['denoised_price'].iloc[-1]
            # For load, use the last available grid_load from future data or fallback
            if idx > future_df.index.min():
                # Use the grid_load from the previous row in future_df if available
                future_df.loc[idx, 'load_lag_24'] = future_df.loc[idx-1, 'grid_load']
            else:
                future_df.loc[idx, 'load_lag_24'] = df['grid_load'].iloc[-1]
    
    # Handle 168-hour lag (using historical data only)
    lag_168_row = df[df['start date/time'] == lag_168_time]
    if not lag_168_row.empty:
        future_df.loc[idx, 'price_lag_168'] = lag_168_row['denoised_price'].values[0]
    else:
        future_df.loc[idx, 'price_lag_168'] = df['denoised_price'].iloc[-1]
    
    # Ensure features are numeric
    try:
        feature_values = future_df.loc[idx, features].values.astype(float)
    except ValueError as e:
        print(f"Error at index {idx}: {e}")
        print("Problematic row:", future_df.loc[idx])
        exit()
    
    # Make the prediction for the current row
    future_df.loc[idx, 'predicted_price'] = final_model.predict([feature_values])[0]

# Extract predictions
predictions_from_06 = future_df[future_df['start date/time'] >= '2025-01-06']
print(predictions_from_06[['start date/time', 'predicted_price']])

# --------------------------
# 8. Plot Actual vs Predicted Prices
# --------------------------
# Create a DataFrame for the actual prices using the provided values.
# You can replace this list with loading from a CSV or other source.
actual_data = [
    {"start date/time": "2025-01-06 00:00:00", "actual_price": 27.52},
    {"start date/time": "2025-01-06 01:00:00", "actual_price": 19.26},
    {"start date/time": "2025-01-06 02:00:00", "actual_price": 11.35},
    {"start date/time": "2025-01-06 03:00:00", "actual_price": 9.20},
    {"start date/time": "2025-01-06 04:00:00", "actual_price": 10.00},
    {"start date/time": "2025-01-06 05:00:00", "actual_price": 14.81},
    {"start date/time": "2025-01-06 06:00:00", "actual_price": 23.82},
    {"start date/time": "2025-01-06 07:00:00", "actual_price": 31.72},
    {"start date/time": "2025-01-06 08:00:00", "actual_price": 41.37},
    {"start date/time": "2025-01-06 09:00:00", "actual_price": 36.36},
    {"start date/time": "2025-01-06 10:00:00", "actual_price": 34.69},
    {"start date/time": "2025-01-06 11:00:00", "actual_price": 32.83},
    {"start date/time": "2025-01-06 12:00:00", "actual_price": 31.28},
    {"start date/time": "2025-01-06 13:00:00", "actual_price": 28.60},
    {"start date/time": "2025-01-06 14:00:00", "actual_price": 25.61},
    {"start date/time": "2025-01-06 15:00:00", "actual_price": 26.32},
    {"start date/time": "2025-01-06 16:00:00", "actual_price": 26.87},
    {"start date/time": "2025-01-06 17:00:00", "actual_price": 31.66},
    {"start date/time": "2025-01-06 18:00:00", "actual_price": 31.58},
    {"start date/time": "2025-01-06 19:00:00", "actual_price": 28.74},
    {"start date/time": "2025-01-06 20:00:00", "actual_price": 26.59},
    {"start date/time": "2025-01-06 21:00:00", "actual_price": 13.82},
    {"start date/time": "2025-01-06 22:00:00", "actual_price": 26.94},
    {"start date/time": "2025-01-06 23:00:00", "actual_price": 12.99},
    {"start date/time": "2025-01-07 00:00:00", "actual_price": 19.07},
    {"start date/time": "2025-01-07 01:00:00", "actual_price": 8.71},
    {"start date/time": "2025-01-07 02:00:00", "actual_price": 8.90},
    {"start date/time": "2025-01-07 03:00:00", "actual_price": 5.01},
    {"start date/time": "2025-01-07 04:00:00", "actual_price": 5.13},
    {"start date/time": "2025-01-07 05:00:00", "actual_price": 5.80},
    {"start date/time": "2025-01-07 06:00:00", "actual_price": 48.86},
    {"start date/time": "2025-01-07 07:00:00", "actual_price": 76.83},
    {"start date/time": "2025-01-07 08:00:00", "actual_price": 85.08},
    {"start date/time": "2025-01-07 09:00:00", "actual_price": 84.24},
    {"start date/time": "2025-01-07 10:00:00", "actual_price": 75.25},
    {"start date/time": "2025-01-07 11:00:00", "actual_price": 62.80},
    {"start date/time": "2025-01-07 12:00:00", "actual_price": 62.44},
    {"start date/time": "2025-01-07 13:00:00", "actual_price": 63.90},
    {"start date/time": "2025-01-07 14:00:00", "actual_price": 72.56},
    {"start date/time": "2025-01-07 15:00:00", "actual_price": 78.11},
    {"start date/time": "2025-01-07 16:00:00", "actual_price": 79.98},
    {"start date/time": "2025-01-07 17:00:00", "actual_price": 96.03},
    {"start date/time": "2025-01-07 18:00:00", "actual_price": 101.11},
    {"start date/time": "2025-01-07 19:00:00", "actual_price": 86.21},
    {"start date/time": "2025-01-07 20:00:00", "actual_price": 78.01},
    {"start date/time": "2025-01-07 21:00:00", "actual_price": 72.45},
    {"start date/time": "2025-01-07 22:00:00", "actual_price": 72.45},
    {"start date/time": "2025-01-07 23:00:00", "actual_price": 50.04},
    {"start date/time": "2025-01-08 00:00:00", "actual_price": 71.05},
    {"start date/time": "2025-01-08 01:00:00", "actual_price": 68.01},
    {"start date/time": "2025-01-08 02:00:00", "actual_price": 63.34},
    {"start date/time": "2025-01-08 03:00:00", "actual_price": 57.01},
    {"start date/time": "2025-01-08 04:00:00", "actual_price": 66.29},
    {"start date/time": "2025-01-08 05:00:00", "actual_price": 72.07},
    {"start date/time": "2025-01-08 06:00:00", "actual_price": 82.70},
    {"start date/time": "2025-01-08 07:00:00", "actual_price": 100.73},
    {"start date/time": "2025-01-08 08:00:00", "actual_price": 128.22},
    {"start date/time": "2025-01-08 09:00:00", "actual_price": 108.18},
    {"start date/time": "2025-01-08 10:00:00", "actual_price": 94.65},
    {"start date/time": "2025-01-08 11:00:00", "actual_price": 100.01},
    {"start date/time": "2025-01-08 12:00:00", "actual_price": 89.99},
    {"start date/time": "2025-01-08 13:00:00", "actual_price": 97.20},
    {"start date/time": "2025-01-08 14:00:00", "actual_price": 110.19},
    {"start date/time": "2025-01-08 15:00:00", "actual_price": 127.80},
    {"start date/time": "2025-01-08 16:00:00", "actual_price": 135.82},
    {"start date/time": "2025-01-08 17:00:00", "actual_price": 155.46},
    {"start date/time": "2025-01-08 18:00:00", "actual_price": 149.23},
    {"start date/time": "2025-01-08 19:00:00", "actual_price": 146.48},
    {"start date/time": "2025-01-08 20:00:00", "actual_price": 136.86},
    {"start date/time": "2025-01-08 21:00:00", "actual_price": 127.86},
    {"start date/time": "2025-01-08 22:00:00", "actual_price": 115.92},
    {"start date/time": "2025-01-08 23:00:00", "actual_price": 103.59},
    {"start date/time": "2025-01-09 00:00:00", "actual_price": 101.44},
    {"start date/time": "2025-01-09 01:00:00", "actual_price": 100.00},
    {"start date/time": "2025-01-09 02:00:00", "actual_price": 98.77},
    {"start date/time": "2025-01-09 03:00:00", "actual_price": 95.22},
    {"start date/time": "2025-01-09 04:00:00", "actual_price": 98.28},
    {"start date/time": "2025-01-09 05:00:00", "actual_price": 102.65},
    {"start date/time": "2025-01-09 06:00:00", "actual_price": 133.54},
    {"start date/time": "2025-01-09 07:00:00", "actual_price": 148.80},
    {"start date/time": "2025-01-09 08:00:00", "actual_price": 164.88},
    {"start date/time": "2025-01-09 09:00:00", "actual_price": 156.15},
    {"start date/time": "2025-01-09 10:00:00", "actual_price": 147.64},
    {"start date/time": "2025-01-09 11:00:00", "actual_price": 140.21},
    {"start date/time": "2025-01-09 12:00:00", "actual_price": 128.18},
    {"start date/time": "2025-01-09 13:00:00", "actual_price": 121.15},
    {"start date/time": "2025-01-09 14:00:00", "actual_price": 123.95},
    {"start date/time": "2025-01-09 15:00:00", "actual_price": 127.53},
    {"start date/time": "2025-01-09 16:00:00", "actual_price": 123.75},
    {"start date/time": "2025-01-09 17:00:00", "actual_price": 130.91},
    {"start date/time": "2025-01-09 18:00:00", "actual_price": 134.75},
    {"start date/time": "2025-01-09 19:00:00", "actual_price": 125.44},
    {"start date/time": "2025-01-09 20:00:00", "actual_price": 119.23},
    {"start date/time": "2025-01-09 21:00:00", "actual_price": 104.99},
    {"start date/time": "2025-01-09 22:00:00", "actual_price": 101.10},
    {"start date/time": "2025-01-09 23:00:00", "actual_price": 88.19},
    {"start date/time": "2025-01-10 00:00:00", "actual_price": 84.79},
    {"start date/time": "2025-01-10 01:00:00", "actual_price": 80.23},
    {"start date/time": "2025-01-10 02:00:00", "actual_price": 71.29},
    {"start date/time": "2025-01-10 03:00:00", "actual_price": 69.05},
    {"start date/time": "2025-01-10 04:00:00", "actual_price": 69.89},
    {"start date/time": "2025-01-10 05:00:00", "actual_price": 83.90},
    {"start date/time": "2025-01-10 06:00:00", "actual_price": 99.09},
    {"start date/time": "2025-01-10 07:00:00", "actual_price": 123.92},
    {"start date/time": "2025-01-10 08:00:00", "actual_price": 139.47},
    {"start date/time": "2025-01-10 09:00:00", "actual_price": 136.92},
    {"start date/time": "2025-01-10 10:00:00", "actual_price": 123.57},
    {"start date/time": "2025-01-10 11:00:00", "actual_price": 113.59},
    {"start date/time": "2025-01-10 12:00:00", "actual_price": 107.43},
    {"start date/time": "2025-01-10 13:00:00", "actual_price": 105.01},
    {"start date/time": "2025-01-10 14:00:00", "actual_price": 110.01},
    {"start date/time": "2025-01-10 15:00:00", "actual_price": 128.69},
    {"start date/time": "2025-01-10 16:00:00", "actual_price": 134.87},
    {"start date/time": "2025-01-10 17:00:00", "actual_price": 142.56},
    {"start date/time": "2025-01-10 18:00:00", "actual_price": 144.12},
    {"start date/time": "2025-01-10 19:00:00", "actual_price": 141.05},
    {"start date/time": "2025-01-10 20:00:00", "actual_price": 134.82},
    {"start date/time": "2025-01-10 21:00:00", "actual_price": 122.51},
    {"start date/time": "2025-01-10 22:00:00", "actual_price": 119.77},
    {"start date/time": "2025-01-10 23:00:00", "actual_price": 111.71},
    {"start date/time": "2025-01-11 00:00:00", "actual_price": 106.64},
    {"start date/time": "2025-01-11 01:00:00", "actual_price": 98.99},
    {"start date/time": "2025-01-11 02:00:00", "actual_price": 95.71},
    {"start date/time": "2025-01-11 03:00:00", "actual_price": 89.45},
    {"start date/time": "2025-01-11 04:00:00", "actual_price": 88.35},
    {"start date/time": "2025-01-11 05:00:00", "actual_price": 88.40},
    {"start date/time": "2025-01-11 06:00:00", "actual_price": 88.13},
    {"start date/time": "2025-01-11 07:00:00", "actual_price": 94.68},
    {"start date/time": "2025-01-11 08:00:00", "actual_price": 107.65},
    {"start date/time": "2025-01-11 09:00:00", "actual_price": 103.14},
    {"start date/time": "2025-01-11 10:00:00", "actual_price": 101.13},
    {"start date/time": "2025-01-11 11:00:00", "actual_price": 99.56},
    {"start date/time": "2025-01-11 12:00:00", "actual_price": 96.74},
    {"start date/time": "2025-01-11 13:00:00", "actual_price": 93.15},
    {"start date/time": "2025-01-11 14:00:00", "actual_price": 94.90},
    {"start date/time": "2025-01-11 15:00:00", "actual_price": 104.89},
    {"start date/time": "2025-01-11 16:00:00", "actual_price": 112.12},
    {"start date/time": "2025-01-11 17:00:00", "actual_price": 119.80},
    {"start date/time": "2025-01-11 18:00:00", "actual_price": 122.34},
    {"start date/time": "2025-01-11 19:00:00", "actual_price": 114.85},
    {"start date/time": "2025-01-11 20:00:00", "actual_price": 111.19},
    {"start date/time": "2025-01-11 21:00:00", "actual_price": 104.80},
    {"start date/time": "2025-01-11 22:00:00", "actual_price": 103.07},
    {"start date/time": "2025-01-11 23:00:00", "actual_price": 105.69},
    {"start date/time": "2025-01-12 00:00:00", "actual_price": 83.58},
]

actual_df = pd.DataFrame(actual_data)
actual_df['start date/time'] = pd.to_datetime(actual_df['start date/time'])

# Merge predictions with actual data
if not predictions_from_06.empty and 'predicted_price' in predictions_from_06.columns:
    comparison_df = pd.merge(predictions_from_06, actual_df, on='start date/time', how='inner')
    
    if not comparison_df.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(comparison_df['start date/time'], comparison_df['predicted_price'], label='Predicted', marker='o')
        plt.plot(comparison_df['start date/time'], comparison_df['actual_price'], label='Actual', marker='x')
        plt.xlabel('Date Time')
        plt.ylabel('Electricity Price')
        plt.title('Actual vs Predicted Electricity Prices (2025-01-06 to 2025-01-12)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print("No overlapping data for comparison")
else:
    print("No predictions available for plotting")