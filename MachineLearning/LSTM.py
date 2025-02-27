import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Layer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

# --- Custom Attention Layer ---
class AttentionLayer(Layer):
    def __init__(self, units: int, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        score = self.V(K.tanh(self.W1(inputs) + self.W2(inputs)))
        attention_weights = K.softmax(score, axis=1)
        context_vector = attention_weights * inputs
        # Sum across the time dimension to get a context vector
        return K.sum(context_vector, axis=1)

# --- Utility functions ---
def clean_column_name(col: str) -> str:
    return (col.strip().lower()
            .replace(' ', '_')
            .replace('-', '_')
            .replace('/', '_')
            .replace('(', '')
            .replace(')', ''))

def load_and_preprocess(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        print("Original columns detected:", df.columns.tolist())
        df.columns = [clean_column_name(col) for col in df.columns]
        print("Standardized columns:", df.columns.tolist())
        
        # Expected column mapping using cleaned names
        expected_columns = {
            'datetime': ['start_date_time', 'start_date', 'date_time', 'timestamp'],
            'dow': ['day_of_week', 'day_of_the_week', 'weekday', 'dow'],
            'price': ['day_price', 'price', 'energy_price'],
            'temperature_2m': ['temperature_2m', 'temp_2m', 'temperature'],
            'total_consumption': ['grid_load', 'total_cons', 'consumption', 'load'],
            'wind_speed_100m': ['wind_speed_100m', 'wind_speed_100m_km_h', 'wind_speed', 'wind_100m', 'wind_kmh']
        }
        
        final_columns = {}
        for target, patterns in expected_columns.items():
            # Look for an exact match or if the pattern is a substring
            matches = [col for col in df.columns if any(p == col or p in col for p in patterns)]
            if not matches:
                raise ValueError(f"Missing column matching patterns: {patterns}")
            final_columns[matches[0]] = target
        
        df = df.rename(columns=final_columns)[list(expected_columns.keys())]
        
        # Convert datetime column (assuming dayfirst format)
        df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True, errors='coerce')
        if df['datetime'].isnull().any():
            raise ValueError("Invalid datetime values – check format (DD/MM/YYYY HH:MM)")
        
        # Convert day-of-week to numeric
        dow_map = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
                   "friday": 4, "saturday": 5, "sunday": 6}
        df['dow'] = df['dow'].str.strip().str.lower().map(dow_map)
        if df['dow'].isnull().any():
            raise ValueError("Invalid day-of-week values found")
        
        # Convert numeric columns and drop rows with invalid values (e.g., "#NUM!")
        numeric_cols = ['price', 'temperature_2m', 'total_consumption', 'wind_speed_100m']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
        if df[numeric_cols].isnull().any().any():
            print(f"Warning: Dropping rows due to invalid numeric values in {numeric_cols}")
        df = df.dropna(subset=numeric_cols)
        
        df = df.sort_values('datetime').reset_index(drop=True)
        print("\nData validation successful!")
        print("Final columns:", df.columns.tolist())
        print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        print(f"Total records: {len(df)}")
        return df

    except Exception as e:
        print(f"\nDATA ERROR: {str(e)}")
        raise

def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Cyclical features for hour and day-of-week
    df['hour_sin'] = np.sin(2 * np.pi * df['datetime'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['datetime'].dt.hour / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
    # Lag features for price
    for lag in [1, 6, 24, 48]:
        df[f'price_lag_{lag}'] = df['price'].shift(lag)
    # Rolling mean features (24-hour window) for temperature and consumption
    for col, windows in {'temperature_2m': [24], 'total_consumption': [24]}.items():
        for w in windows:
            df[f'{col}_roll{w}_mean'] = df[col].rolling(w, min_periods=1).mean()
    # Interaction features
    df['temp_load'] = df['temperature_2m'] * df['total_consumption']
    df['wind_temp'] = df['wind_speed_100m'] / (df['temperature_2m'] + 1e-6)
    
    return df.dropna().reset_index(drop=True)

def create_sequences(X, y, window=24):
    X_seq, y_seq = [], []
    for i in range(len(X) - window):
        X_seq.append(X[i:i+window])
        y_seq.append(y[i+window])
    return np.array(X_seq), np.array(y_seq)

def build_temporal_model(input_shape):
    # Build LSTM model with attention; note the second LSTM has been removed
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape,
             kernel_regularizer=L1L2(1e-5, 1e-4)),
        Dropout(0.3),
        AttentionLayer(64),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
                  loss='mse', metrics=['mae'])
    return model

def forecast_future(model, df, FEATURES, TARGET, scaler_x, scaler_y, forecast_horizon=168, window=24):
    """
    Recursively forecast the next forecast_horizon hours.
    Assumes exogenous features (temperature, consumption, wind) stay constant at their last observed values.
    """
    last_date = df['datetime'].max()
    last_row = df.iloc[-1]
    const_temp = last_row['temperature_2m']
    const_cons = last_row['total_consumption']
    const_wind = last_row['wind_speed_100m']
    temp_roll24 = last_row['temperature_2m_roll24_mean']
    cons_roll24 = last_row['total_consumption_roll24_mean']
    
    # Get the last window of features (raw)
    last_window_raw = df[FEATURES].iloc[-window:].values
    current_seq = scaler_x.transform(last_window_raw)
    
    # Price history to compute lag features
    price_history = df[TARGET].iloc[-window:].tolist()
    forecast_timestamps = []
    predictions = []
    
    for i in range(forecast_horizon):
        X_input = current_seq.reshape(1, window, len(FEATURES))
        pred_scaled = model.predict(X_input)
        pred_price = scaler_y.inverse_transform(pred_scaled)[0, 0]
        
        future_time = last_date + pd.Timedelta(hours=i+1)
        forecast_timestamps.append(future_time)
        predictions.append(pred_price)
        
        # Update price history for lag features
        price_history.append(pred_price)
        hour = future_time.hour
        dow = future_time.dayofweek
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        dow_sin = np.sin(2 * np.pi * dow / 7)
        dow_cos = np.cos(2 * np.pi * dow / 7)
        lag1 = price_history[-1]
        lag6 = price_history[-6] if len(price_history) >= 6 else price_history[0]
        lag24 = price_history[-24] if len(price_history) >= 24 else price_history[0]
        
        # Construct new feature vector matching FEATURES order:
        # ['temperature_2m', 'total_consumption', 'wind_speed_100m',
        #  'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        #  'price_lag_1', 'price_lag_6', 'price_lag_24',
        #  'temperature_2m_roll24_mean', 'total_consumption_roll24_mean',
        #  'temp_load', 'wind_temp']
        new_features = np.array([
            const_temp,
            const_cons,
            const_wind,
            hour_sin,
            hour_cos,
            dow_sin,
            dow_cos,
            lag1,
            lag6,
            lag24,
            temp_roll24,
            cons_roll24,
            const_temp * const_cons,
            const_wind / (const_temp + 1e-6)
        ])
        new_features_scaled = scaler_x.transform(new_features.reshape(1, -1))[0]
        current_seq = np.vstack([current_seq[1:], new_features_scaled])
    
    forecast_df = pd.DataFrame({
        'datetime': forecast_timestamps,
        'predicted_price': predictions
    })
    return forecast_df

def main():
    try:
        DATA_PATH = "data/merged-data.csv"
        df = load_and_preprocess(DATA_PATH)
        df = create_temporal_features(df)
        
        # Define feature order (must match forecast_future())
        FEATURES = [
            'temperature_2m', 'total_consumption', 'wind_speed_100m',
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
            'price_lag_1', 'price_lag_6', 'price_lag_24',
            'temperature_2m_roll24_mean', 'total_consumption_roll24_mean',
            'temp_load', 'wind_temp'
        ]
        TARGET = 'price'
        
        # Scale features and target
        scaler_x = StandardScaler()
        scaler_y = MinMaxScaler()
        X = scaler_x.fit_transform(df[FEATURES])
        y = scaler_y.fit_transform(df[[TARGET]])
        
        window = 24
        X_seq, y_seq = create_sequences(X, y, window)
        print(f"Training on {X_seq.shape[0]} sequences of length {window}.")
        
        # Build and train the model
        model = build_temporal_model((window, len(FEATURES)))
        model.fit(
            X_seq, y_seq,
            epochs=100,
            batch_size=64,
            callbacks=[
                EarlyStopping(patience=15, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.2, patience=5)
            ],
            verbose=1
        )
        
        # Evaluate on training set
        train_pred = model.predict(X_seq)
        train_pred_inv = scaler_y.inverse_transform(train_pred)
        train_true_inv = scaler_y.inverse_transform(y_seq)
        print("\nTraining Metrics:")
        print(f"MAE: {mean_absolute_error(train_true_inv, train_pred_inv):.2f}")
        print(f"RMSE: {np.sqrt(mean_squared_error(train_true_inv, train_pred_inv)):.2f}")
        print(f"R²: {r2_score(train_true_inv, train_pred_inv):.2f}")
        
        # Forecast next week (168 hours)
        forecast_horizon = 168
        forecast_df = forecast_future(model, df, FEATURES, TARGET, scaler_x, scaler_y, forecast_horizon, window)
        forecast_df.to_csv('energy_price_predictions.csv', index=False)
        print("Future predictions saved to 'energy_price_predictions.csv'.")
    
    except Exception as e:
        print(f"\nFatal Error: {str(e)}")
        print("Troubleshooting Checklist:")
        print("1. Verify CSV file exists at specified path")
        print("2. Check column names match expected patterns")
        print("3. Validate datetime format (DD/MM/YYYY HH:MM)")
        print("4. Ensure numeric columns contain valid values")
        print("5. Confirm sufficient data points (at least 2 years)")
    
if __name__ == "__main__":
    main()

# # ----- Forecasting for the Coming Week using Prophet -----

# def forecast_feature(data, feature, start_date, end_date):
#     """Forecast a feature between specific dates using Prophet"""
#     feature_data = data[['StartDateTime', feature]].rename(columns={'StartDateTime': 'ds', feature: 'y'})
#     model_prophet = Prophet()
#     model_prophet.fit(feature_data)
    
#     future = pd.date_range(start=start_date, end=end_date, freq='h')  # Changed to 'h'
#     forecast = model_prophet.predict(pd.DataFrame({'ds': future}))
#     return forecast[['ds', 'yhat']]

# # Define custom date range
# start_date = '2025-01-20 00:00:00'
# end_date = '2025-01-26 23:00:00'
# future_dates = pd.date_range(start=start_date, end=end_date, freq='h')

# # Create future DataFrame
# future_data = pd.DataFrame(index=future_dates)
# future_data['Year'] = future_data.index.year
# future_data['Month'] = future_data.index.month
# future_data['Day'] = future_data.index.day
# future_data['Hour'] = future_data.index.hour
# future_data['DayOfWeek'] = future_data.index.dayofweek

# # Forecast features for custom dates
# temp_forecast = forecast_feature(data, 'temperature_2m', start_date, end_date)
# wind_forecast = forecast_feature(data, 'wind_speed_100m (km/h)', start_date, end_date)
# load_forecast = forecast_feature(data, 'total-consumption', start_date, end_date)

# # Merge forecasts
# future_data = future_data.merge(temp_forecast, left_index=True, right_on='ds')
# future_data = future_data.merge(wind_forecast, on='ds', suffixes=('', '_wind'))
# future_data = future_data.merge(load_forecast, on='ds', suffixes=('', '_load'))
# future_data.rename(columns={
#     'yhat': 'temperature_2m',
#     'yhat_wind': 'wind_speed_100m (km/h)',
#     'yhat_load': 'total-consumption'
# }, inplace=True)
# future_data.set_index('ds', inplace=True)
# future_data = prepare_future_data(future_data, data)

# # Add rolling averages
# future_data['Rolling_Temp_24h'] = future_data['temperature_2m'].rolling(24).mean()
# future_data['Rolling_Wind_24h'] = future_data['wind_speed_100m (km/h)'].rolling(24).mean()
# future_data['Rolling_Load_24h'] = future_data['total-consumption'].rolling(24).mean()
# future_data['Lag_Price'] = data['Price'].iloc[-1]
# future_data.bfill(inplace=True)

# # Scale features
# future_data_scaled = scaler_X.transform(future_data[features])

# # Prepare sequences
# time_steps = 24 
# future_seq = []
# for i in range(len(future_data_scaled) - time_steps + 1):
#     future_seq.append(future_data_scaled[i:i+time_steps])
# future_seq = np.array(future_seq)

# # Predict prices
# future_predictions_scaled = model.predict(future_seq)
# future_predictions = scaler_y.inverse_transform(future_predictions_scaled)

# # Save predictions
# future_predictions_df = pd.DataFrame({
#     'StartDateTime': future_dates[time_steps-1:],
#     'Predicted Price [Euro/MWh]': future_predictions.flatten()
# })
# # future_predictions_df.to_csv('custom_predictions.csv', index=False)

# # ----- Comparison with Actual Data -----
# actual_data = pd.DataFrame({
#     'StartDateTime': [
#         # Jan 20, 2025
#         '2025-01-20 00:00:00', '2025-01-20 01:00:00', '2025-01-20 02:00:00', 
#         '2025-01-20 03:00:00', '2025-01-20 04:00:00', '2025-01-20 05:00:00', 
#         '2025-01-20 06:00:00', '2025-01-20 07:00:00', '2025-01-20 08:00:00', 
#         '2025-01-20 09:00:00', '2025-01-20 10:00:00', '2025-01-20 11:00:00', 
#         '2025-01-20 12:00:00', '2025-01-20 13:00:00', '2025-01-20 14:00:00', 
#         '2025-01-20 15:00:00', '2025-01-20 16:00:00', '2025-01-20 17:00:00', 
#         '2025-01-20 18:00:00', '2025-01-20 19:00:00', '2025-01-20 20:00:00', 
#         '2025-01-20 21:00:00', '2025-01-20 22:00:00', '2025-01-20 23:00:00',
#         # Jan 21, 2025
#         '2025-01-21 00:00:00', '2025-01-21 01:00:00', '2025-01-21 02:00:00', 
#         '2025-01-21 03:00:00', '2025-01-21 04:00:00', '2025-01-21 05:00:00', 
#         '2025-01-21 06:00:00', '2025-01-21 07:00:00', '2025-01-21 08:00:00', 
#         '2025-01-21 09:00:00', '2025-01-21 10:00:00', '2025-01-21 11:00:00', 
#         '2025-01-21 12:00:00', '2025-01-21 13:00:00', '2025-01-21 14:00:00', 
#         '2025-01-21 15:00:00', '2025-01-21 16:00:00', '2025-01-21 17:00:00', 
#         '2025-01-21 18:00:00', '2025-01-21 19:00:00', '2025-01-21 20:00:00', 
#         '2025-01-21 21:00:00', '2025-01-21 22:00:00', '2025-01-21 23:00:00',
#         # Jan 22, 2025
#         '2025-01-22 00:00:00', '2025-01-22 01:00:00', '2025-01-22 02:00:00', 
#         '2025-01-22 03:00:00', '2025-01-22 04:00:00', '2025-01-22 05:00:00', 
#         '2025-01-22 06:00:00', '2025-01-22 07:00:00', '2025-01-22 08:00:00', 
#         '2025-01-22 09:00:00', '2025-01-22 10:00:00', '2025-01-22 11:00:00', 
#         '2025-01-22 12:00:00', '2025-01-22 13:00:00', '2025-01-22 14:00:00', 
#         '2025-01-22 15:00:00', '2025-01-22 16:00:00', '2025-01-22 17:00:00', 
#         '2025-01-22 18:00:00', '2025-01-22 19:00:00', '2025-01-22 20:00:00', 
#         '2025-01-22 21:00:00', '2025-01-22 22:00:00', '2025-01-22 23:00:00',
#         # Jan 23, 2025
#         '2025-01-23 00:00:00', '2025-01-23 01:00:00', '2025-01-23 02:00:00', 
#         '2025-01-23 03:00:00', '2025-01-23 04:00:00', '2025-01-23 05:00:00', 
#         '2025-01-23 06:00:00', '2025-01-23 07:00:00', '2025-01-23 08:00:00', 
#         '2025-01-23 09:00:00', '2025-01-23 10:00:00', '2025-01-23 11:00:00', 
#         '2025-01-23 12:00:00', '2025-01-23 13:00:00', '2025-01-23 14:00:00', 
#         '2025-01-23 15:00:00', '2025-01-23 16:00:00', '2025-01-23 17:00:00', 
#         '2025-01-23 18:00:00', '2025-01-23 19:00:00', '2025-01-23 20:00:00', 
#         '2025-01-23 21:00:00', '2025-01-23 22:00:00', '2025-01-23 23:00:00',
#         # Jan 24, 2025
#         '2025-01-24 00:00:00', '2025-01-24 01:00:00', '2025-01-24 02:00:00', 
#         '2025-01-24 03:00:00', '2025-01-24 04:00:00', '2025-01-24 05:00:00', 
#         '2025-01-24 06:00:00', '2025-01-24 07:00:00', '2025-01-24 08:00:00', 
#         '2025-01-24 09:00:00', '2025-01-24 10:00:00', '2025-01-24 11:00:00', 
#         '2025-01-24 12:00:00', '2025-01-24 13:00:00', '2025-01-24 14:00:00', 
#         '2025-01-24 15:00:00', '2025-01-24 16:00:00', '2025-01-24 17:00:00', 
#         '2025-01-24 18:00:00', '2025-01-24 19:00:00', '2025-01-24 20:00:00', 
#         '2025-01-24 21:00:00', '2025-01-24 22:00:00', '2025-01-24 23:00:00',
#         # Jan 25, 2025
#         '2025-01-25 00:00:00', '2025-01-25 01:00:00', '2025-01-25 02:00:00', 
#         '2025-01-25 03:00:00', '2025-01-25 04:00:00', '2025-01-25 05:00:00', 
#         '2025-01-25 06:00:00', '2025-01-25 07:00:00', '2025-01-25 08:00:00', 
#         '2025-01-25 09:00:00', '2025-01-25 10:00:00', '2025-01-25 11:00:00', 
#         '2025-01-25 12:00:00', '2025-01-25 13:00:00', '2025-01-25 14:00:00', 
#         '2025-01-25 15:00:00', '2025-01-25 16:00:00', '2025-01-25 17:00:00', 
#         '2025-01-25 18:00:00', '2025-01-25 19:00:00', '2025-01-25 20:00:00', 
#         '2025-01-25 21:00:00', '2025-01-25 22:00:00', '2025-01-25 23:00:00',
#         # Jan 26, 2025
#         '2025-01-26 00:00:00', '2025-01-26 01:00:00', '2025-01-26 02:00:00', 
#         '2025-01-26 03:00:00', '2025-01-26 04:00:00', '2025-01-26 05:00:00', 
#         '2025-01-26 06:00:00', '2025-01-26 07:00:00', '2025-01-26 08:00:00', 
#         '2025-01-26 09:00:00', '2025-01-26 10:00:00', '2025-01-26 11:00:00', 
#         '2025-01-26 12:00:00', '2025-01-26 13:00:00', '2025-01-26 14:00:00', 
#         '2025-01-26 15:00:00', '2025-01-26 16:00:00', '2025-01-26 17:00:00', 
#         '2025-01-26 18:00:00', '2025-01-26 19:00:00', '2025-01-26 20:00:00', 
#         '2025-01-26 21:00:00', '2025-01-26 22:00:00', '2025-01-26 23:00:00'
#     ],
#     'Actual Price [Euro/MWh]': [ 
#         # Jan 20, 2025
#         122.27, 119.44, 116.56, 114.41, 115.45, 127.54, 161.71, 276.48, 431.99, 
#         291.70, 236.29, 187.48, 176.00, 171.39, 191.85, 277.17, 203.12, 180.40, 
#         173.28, 135.57, 220.00, 170.00, 152.51, 137.98,
#         # Jan 21, 2025
#         127.52, 121.65, 116.67, 113.86, 113.54, 122.34, 142.20, 190.06, 248.32, 
#         211.68, 198.05, 161.50, 147.44, 142.83, 150.05, 173.43, 212.13, 301.15, 
#         251.93, 202.68, 174.84, 155.19, 138.60, 125.29,
#         # Jan 22, 2025
#         128.00, 125.06, 122.22, 121.66, 123.16, 128.32, 156.92, 208.25, 238.60, 
#         199.41, 179.06, 172.94, 165.00, 170.00, 179.92, 195.67, 199.04, 208.99, 
#         180.43, 167.08, 134.49, 127.78, 128.09, 114.68,
#         # Jan 23, 2025
#         113.70, 108.88, 105.58, 100.01, 97.59, 100.01, 106.41, 136.60, 159.03, 
#         155.42, 130.94, 116.10, 94.93, 88.56, 85.86, 89.80, 89.87, 106.75, 112.00, 
#         101.87, 86.36, 74.28, 75.68, 58.23,
#         # Jan 24, 2025
#         69.03, 58.16, 45.05, 40.60, 50.17, 73.20, 85.44, 109.16, 116.23, 96.34, 
#         88.90, 78.44, 76.48, 74.69, 74.51, 74.51, 75.25, 83.02, 88.59, 91.78, 
#         84.99, 80.79, 91.97, 86.29,
#         # Jan 25, 2025
#         59.04, 64.96, 63.55, 67.17, 76.77, 76.89, 79.32, 76.89, 88.99, 87.31, 
#         86.32, 88.15, 82.55, 81.98, 89.00, 111.07, 132.97, 145.85, 151.00, 147.53, 
#         137.60, 134.61, 132.26, 122.03,
#         # Jan 26, 2025
#         126.21, 115.79, 111.30, 106.85, 105.43, 107.86, 110.60, 117.62, 124.17, 
#         121.97, 105.51, 102.57, 96.73, 90.02, 92.52, 111.63, 125.92, 132.77, 118.94, 
#         90.32, 78.93, 68.26, 49.25, 23.89
#     ]
# })

# actual_data['StartDateTime'] = pd.to_datetime(actual_data['StartDateTime'])

# comparison_df = pd.merge(future_predictions_df, actual_data, on='StartDateTime', how='inner')

# if comparison_df.empty:
#     print("No matching actual data available for the forecast period. Skipping future evaluation metrics.")
# else:
#     mse_future = mean_squared_error(comparison_df['Actual Price [Euro/MWh]'], comparison_df['Predicted Price [Euro/MWh]'])
#     mae_future = mean_absolute_error(comparison_df['Actual Price [Euro/MWh]'], comparison_df['Predicted Price [Euro/MWh]'])
#     r2_future = r2_score(comparison_df['Actual Price [Euro/MWh]'], comparison_df['Predicted Price [Euro/MWh]'])
#     rmse_future = np.sqrt(mse_future)

#     print("Future Evaluation Metrics:")
#     print(f"Mean Squared Error: {mse_future:.4f}")
#     print(f"Mean Absolute Error: {mae_future:.4f}")
#     print(f"R-squared: {r2_future:.4f}")
#     print(f"Root Mean Squared Error: {rmse_future:.4f}")

#     # Plot predicted vs actual prices
#     plt.figure(figsize=(12, 6))
#     plt.plot(comparison_df['StartDateTime'], comparison_df['Predicted Price [Euro/MWh]'], label='Predicted', marker='o')
#     plt.plot(comparison_df['StartDateTime'], comparison_df['Actual Price [Euro/MWh]'], label='Actual', marker='x')
#     plt.title('Predicted vs Actual Hourly Prices')
#     plt.xlabel('StartDateTime')
#     plt.ylabel('Price [Euro/MWh]')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

#     # Plot residuals
#     comparison_df['Residuals'] = comparison_df['Actual Price [Euro/MWh]'] - comparison_df['Predicted Price [Euro/MWh]']
#     plt.figure(figsize=(12, 6))
#     plt.plot(comparison_df['StartDateTime'], comparison_df['Residuals'], marker='o', color='red')
#     plt.axhline(0, color='black', linestyle='--')
#     plt.title('Residuals (Actual - Predicted)')
#     plt.xlabel('StartDateTime')
#     plt.ylabel('Residuals [Euro/MWh]')
#     plt.grid(True)
#     plt.show()