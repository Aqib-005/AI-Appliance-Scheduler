import numpy as np
import pandas as pd
import pywt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

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

# Convert target column to numeric.
df['day_price'] = pd.to_numeric(df['day_price'], errors='coerce')

# Fill missing values in essential columns (avoid dropping too many rows)
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
# Create lag features (24-hour and 168-hour lags)
for lag in [24, 168]:
    df[f'price_lag_{lag}'] = df['denoised_price'].shift(lag)
    df[f'load_lag_{lag}'] = df['grid_load'].shift(lag)

# Drop the first 168 rows (which will have NaN lag values)
df = df.iloc[168:]

# Extract datetime features
df['Hour'] = df['start date/time'].dt.hour
df['day of week'] = df['start date/time'].dt.dayofweek  # Monday=0, Sunday=6

# Create cyclical features
df['hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
df['day_of_week_sin'] = np.sin(2 * np.pi * df['day of week'] / 7)

# In case any NaNs remain in lag features, fill them.
lag_cols = [f'price_lag_24', f'price_lag_168', f'load_lag_24']
df[lag_cols] = df[lag_cols].ffill().bfill()
df[essential_cols] = df[essential_cols].ffill().bfill()

print("Total samples:", len(df))

# --------------------------
# 4. Train/Test Split (80/20 Chronological Split)
# --------------------------
split_date = df['start date/time'].quantile(0.8)
train = df[df['start date/time'] <= split_date].copy()
test = df[df['start date/time'] > split_date].copy()
print("Training samples:", len(train), "Testing samples:", len(test))
if len(train) == 0 or len(test) == 0:
    raise ValueError("After splitting, one of the datasets is empty. Check dataset size or lag feature settings.")

# --------------------------
# 5. Model Training
# --------------------------
features = [
    'temperature_2m', 'precipitation (mm)', 'wind_speed_100m (km/h)',
    'relative_humidity_2m (%)', 'grid_load', 'price_lag_24', 'price_lag_168',
    'load_lag_24', 'hour_sin', 'hour_cos', 'day_of_week_sin'
]

model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=50,
    verbosity=0
)

model.fit(
    train[features],
    train['denoised_price'],
    eval_set=[(test[features], test['denoised_price'])],
    verbose=False
)

# --------------------------
# 6. Evaluation
# --------------------------
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape

test_preds = model.predict(test[features])
mae, rmse, mape = calculate_metrics(test['denoised_price'], test_preds)
print(f"Test MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

# --------------------------
# 7. Future Prediction
# --------------------------
def create_future_dataset(start_date, end_date):
    # Use lowercase 'h' for frequency to avoid deprecation warnings.
    date_range = pd.date_range(start=start_date, end=end_date, freq='h')
    future = pd.DataFrame({'start date/time': date_range})
    
    future['Hour'] = future['start date/time'].dt.hour
    future['day of week'] = future['start date/time'].dt.dayofweek

    historical = df.copy()  # processed historical data

    # Function to compute average with fallback if no exact match.
    def get_feature_average(row, col):
        subset = historical[
            (historical['start date/time'].dt.month == row['start date/time'].month) &
            (historical['start date/time'].dt.day == row['start date/time'].day) &
            (historical['start date/time'].dt.hour == row['Hour'])
        ]
        if subset.empty:
            subset = historical[
                (historical['start date/time'].dt.month == row['start date/time'].month) &
                (historical['start date/time'].dt.hour == row['Hour'])
            ]
        return subset[col].mean()
    
    for col in ['temperature_2m', 'precipitation (mm)', 'wind_speed_100m (km/h)',
                'relative_humidity_2m (%)', 'grid_load']:
        future[col] = future.apply(lambda x: get_feature_average(x, col), axis=1)
    
    # Function to get lag features with fallback.
    def get_lag_value(row, col, lag):
        target_time = row['start date/time'] - pd.Timedelta(hours=lag)
        subset = historical[historical['start date/time'] == target_time]
        if subset.empty:
            return historical[col].mean()
        return subset[col].mean()
    
    for lag in [24, 168]:
        future[f'price_lag_{lag}'] = future.apply(lambda x: get_lag_value(x, 'denoised_price', lag), axis=1)
        future[f'load_lag_{lag}'] = future.apply(lambda x: get_lag_value(x, 'grid_load', lag), axis=1)
    
    future['hour_sin'] = np.sin(2 * np.pi * future['Hour'] / 24)
    future['hour_cos'] = np.cos(2 * np.pi * future['Hour'] / 24)
    future['day_of_week_sin'] = np.sin(2 * np.pi * future['day of week'] / 7)
    
    future.fillna(method='ffill', inplace=True)
    
    return future

# Generate future predictions for January 6, 2025 to January 12, 2025.
future_data = create_future_dataset('2025-01-06', '2025-01-12')
future_predictions = model.predict(future_data[features])
future_data['predicted_price'] = future_predictions

print(future_data[['start date/time', 'predicted_price']])

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
    # ... (continue adding rows for Jan 7, Jan 8, Jan 9, Jan 10, Jan 11, Jan 12 as provided)
    {"start date/time": "2025-01-12 00:00:00", "actual_price": 83.58}
]

actual_df = pd.DataFrame(actual_data)
actual_df['start date/time'] = pd.to_datetime(actual_df['start date/time'])

# Merge the future predictions with actual data on 'start date/time'
comparison_df = pd.merge(future_data, actual_df, on='start date/time', how='inner')

plt.figure(figsize=(12, 6))
plt.plot(comparison_df['start date/time'], comparison_df['predicted_price'], label='Predicted', marker='o')
plt.plot(comparison_df['start date/time'], comparison_df['actual_price'], label='Actual', marker='x')
plt.xlabel('Date Time')
plt.ylabel('Electricity Price')
plt.title('Actual vs Predicted Electricity Prices')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
