import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Layer
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from prophet import Prophet
from typing import Tuple

# Custom Attention Layer
class AttentionLayer(Layer):
    def __init__(self, units: int, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        hidden_states = inputs
        score = self.V(K.tanh(self.W1(hidden_states) + self.W2(hidden_states)))
        attention_weights = K.softmax(score, axis=1)
        context_vector = attention_weights * hidden_states
        context_vector = K.sum(context_vector, axis=1)
        return context_vector

# 1. Enhanced Data Pipeline
def load_and_preprocess(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.rename(columns={
    'start date/time': 'datetime',
    'day of the week': 'dow',
    'day-price': 'price'
    }, inplace=True)
    
    # Temporal features
    df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True)
    df['dow'] = df['dow'].map({
        "Monday": 0, "Tuesday": 1, "Wednesday": 2,
        "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
    })
    
    # Numeric cleaning
    numeric_cols = ['price', 'total-consumption']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    
    return df.sort_values('datetime').dropna().reset_index(drop=True)

# 2. Temporal Feature Engineering
def create_temporal_features(df: pd.DataFrame, buffer_days: int = 7) -> pd.DataFrame:
    # Time-based features
    df['hour_sin'] = np.sin(2 * np.pi * df['datetime'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['datetime'].dt.hour / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
    
    # Lag features with buffer
    max_lag = 72  # 3 days
    for lag in [1, 6, 24, 48, 72]:
        df[f'price_lag_{lag}'] = df['price'].shift(lag)
    
    # Rolling features with buffer
    rolling_config = {
        'temperature_2m': [6, 12, 24],
        'total-consumption': [12, 24, 72],
        'wind_speed_100m (km/h)': [6, 12]
    }
    
    for col, windows in rolling_config.items():
        for window in windows:
            df[f'{col}_roll{window}_mean'] = df[col].rolling(window, min_periods=1).mean()
            df[f'{col}_roll{window}_std'] = df[col].rolling(window, min_periods=1).std()
    
    # Interaction features
    df['temp_load'] = df['temperature_2m'] * df['total-consumption']
    df['wind_temp'] = df['wind_speed_100m (km/h)'] / (df['temperature_2m'] + 1e-6)
    
    return df.iloc[max_lag:].reset_index(drop=True)

# 3. Temporal Dataset Split
def temporal_split(df: pd.DataFrame, 
                  test_start: str,
                  buffer_days: int = 7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    test_start = pd.to_datetime(test_start)
    buffer_start = test_start - pd.Timedelta(days=buffer_days)
    
    train_df = df[df['datetime'] < buffer_start]
    buffer_df = df[(df['datetime'] >= buffer_start) & (df['datetime'] < test_start)]
    test_df = df[df['datetime'] >= test_start]
    
    return pd.concat([train_df, buffer_df]), test_df

# 4. Enhanced LSTM Model
def build_temporal_model(input_shape: tuple) -> Sequential:
    model = Sequential([
        LSTM(256, return_sequences=True, input_shape=input_shape,
            kernel_regularizer=L1L2(1e-5, 1e-4)),
        Dropout(0.3),
        AttentionLayer(128),
        LSTM(128, kernel_regularizer=L1L2(1e-5, 1e-4)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    optimizer = Adam(learning_rate=0.0005, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model


def forecast_prophet(df: pd.DataFrame, 
                    target: str, 
                    start: str, 
                    end: str,
                    samples: int = 200) -> pd.DataFrame:
    model = Prophet(mcmc_samples=samples, uncertainty_samples=1000)
    model.add_seasonality(name='hourly', period=1, fourier_order=10)
    model.fit(df[['datetime', target]].rename(columns={'datetime': 'ds', target: 'y'}))
    
    future = model.make_future_dataframe(
        periods=int((pd.to_datetime(end) - pd.to_datetime(start)).total_seconds() // 3600),
        freq='H'
    )
    return model.predict(future)

def monte_carlo_predict(model: Sequential,
                       features: np.ndarray,
                       scaler: StandardScaler,
                       samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    predictions = []
    for _ in range(samples):
        # Add Gaussian noise to features
        noisy_features = features + np.random.normal(0, 0.1, features.shape)
        pred = model.predict(noisy_features, verbose=0)
        predictions.append(pred)
    
    predictions = np.array(predictions)
    return np.median(predictions, axis=0), np.std(predictions, axis=0)

def generate_future_features(train_df: pd.DataFrame,
                            start: str,
                            end: str) -> pd.DataFrame:
    """Generate future features for prediction period"""
    # 1. Create base datetime index
    future_dates = pd.date_range(start, end, freq='H')
    future_df = pd.DataFrame({'datetime': future_dates})
    
    # 2. Add temporal features
    future_df['dow'] = future_df['datetime'].dt.dayofweek
    future_df['hour_sin'] = np.sin(2 * np.pi * future_df['datetime'].dt.hour / 24)
    future_df['hour_cos'] = np.cos(2 * np.pi * future_df['datetime'].dt.hour / 24)
    future_df['dow_sin'] = np.sin(2 * np.pi * future_df['dow'] / 7)
    future_df['dow_cos'] = np.cos(2 * np.pi * future_df['dow'] / 7)
    
    # 3. Forecast external features
    for feature in ['temperature_2m', 'total-consumption', 'wind_speed_100m (km/h)']:
        fcst = forecast_prophet(train_df, feature, start, end)
        future_df[feature] = fcst['yhat'].values
    
    # 4. Create lag/roll features from training data
    last_prices = train_df['price'].values[-72:]
    for lag in [1, 6, 24, 48, 72]:
        future_df[f'price_lag_{lag}'] = np.concatenate([
            last_prices[-lag:], 
            np.full(len(future_df) - lag, np.nan)
        ])[:len(future_df)].ffill().bfill().values
    
    # 5. Calculate rolling features
    rolling_config = {
        'temperature_2m': [6, 12, 24],
        'total-consumption': [12, 24, 72],
        'wind_speed_100m (km/h)': [6, 12]
    }
    
    for col, windows in rolling_config.items():
        for window in windows:
            future_df[f'{col}_roll{window}_mean'] = (
                future_df[col].rolling(window, min_periods=1).mean()
            )
    
    # 6. Interaction features
    future_df['temp_load'] = future_df['temperature_2m'] * future_df['total-consumption']
    future_df['wind_temp'] = future_df['wind_speed_100m (km/h)'] / (future_df['temperature_2m'] + 1e-6)
    
    return future_df.dropna().reset_index(drop=True)

# Main Execution Flow
if __name__ == "__main__":
    # Configuration
    PREDICTION_START = '2025-01-20'  # Update this based on your data
    PREDICTION_END = '2025-01-26'
    FEATURES = [
        'temperature_2m', 'total-consumption', 'wind_speed_100m (km/h)',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'price_lag_1', 'price_lag_24', 'price_lag_48',
        'temperature_2m_roll24_mean', 'total-consumption_roll24_mean',
        'wind_temp', 'temp_load'
    ]
    TARGET = 'price'

    # 1. Load and validate data
    full_df = load_and_preprocess("data/merged-data.csv")
    print(f"Dataset date range: {full_df['datetime'].min()} to {full_df['datetime'].max()}")
    
    # 2. Temporal split validation
    last_known_date = full_df['datetime'].max()
    predict_future = pd.to_datetime(PREDICTION_START) > last_known_date
    
    if predict_future:
        print("\n[Forecasting Future Prices]")
        print("Generating future features...")
        
        # Use all data for training
        train_df = create_temporal_features(full_df)
        
        # Scale data
        scaler_x = StandardScaler()
        scaler_y = MinMaxScaler()
        X_train = scaler_x.fit_transform(train_df[FEATURES])
        y_train = scaler_y.fit_transform(train_df[[TARGET]])
        
        # Create sequences
        X_seq, y_seq = create_sequences(X_train, y_train)
        
        # Build and train model
        print("Training model...")
        model = build_temporal_model((X_seq.shape[1], X_seq.shape[2]))
        model.fit(
            X_seq, y_seq,
            epochs=200,
            batch_size=128,
            callbacks=[
                EarlyStopping(patience=20, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.2, patience=10)
            ],
            verbose=1
        )
        
        # Generate future features
        future_df = generate_future_features(train_df, PREDICTION_START, PREDICTION_END)
        X_future = scaler_x.transform(future_df[FEATURES])
        X_future_seq = np.array([X_future[i:i+24] for i in range(len(X_future) - 23)])
        
        # Predict with uncertainty
        print("Making predictions...")
        pred_median, pred_std = monte_carlo_predict(model, X_future_seq, scaler_x)
        pred_median = scaler_y.inverse_transform(pred_median)
        pred_std = scaler_y.inverse_transform(pred_std)
        
        # Create output
        predictions = pd.DataFrame({
            'datetime': pd.date_range(start=PREDICTION_START, end=PREDICTION_END, freq='H')[23:],
            'predicted_price': pred_median.flatten(),
            'uncertainty': pred_std.flatten()
        })
        
        # Save results
        predictions.to_csv('price_predictions.csv', index=False)
        print(f"Predictions saved for {PREDICTION_START} to {PREDICTION_END}")
        
        # Visualization
        plt.figure(figsize=(15, 6))
        plt.plot(predictions['datetime'], predictions['predicted_price'], label='Forecast')
        plt.fill_between(predictions['datetime'],
                        predictions['predicted_price'] - 1.96*predictions['uncertainty'],
                        predictions['predicted_price'] + 1.96*predictions['uncertainty'],
                        alpha=0.2, label='95% CI')
        plt.title(f"Energy Price Forecast {PREDICTION_START} to {PREDICTION_END}")
        plt.xlabel("Date")
        plt.ylabel("Price (EUR/MWh)")
        plt.legend()
        plt.grid()
        plt.show()

    else:
        print("\n[Validating on Historical Data]")
        # Temporal split
        train_df, test_df = temporal_split(full_df, PREDICTION_START)
        
        # Feature engineering
        train_df = create_temporal_features(train_df)
        test_df = create_temporal_features(test_df)
        
        # Scaling
        scaler_x = StandardScaler()
        scaler_y = MinMaxScaler()
        X_train = scaler_x.fit_transform(train_df[FEATURES])
        y_train = scaler_y.fit_transform(train_df[[TARGET]])
        X_test = scaler_x.transform(test_df[FEATURES])
        y_test = scaler_y.transform(test_df[[TARGET]])
        
        # Create sequences
        X_train_seq, y_train_seq = create_sequences(X_train, y_train)
        X_test_seq, y_test_seq = create_sequences(X_test, y_test)
        
        # Train model
        print("Training model...")
        model = build_temporal_model((X_train_seq.shape[1], X_train_seq.shape[2]))
        history = model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_test_seq, y_test_seq),
            epochs=200,
            batch_size=128,
            callbacks=[
                EarlyStopping(patience=20, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.2, patience=10)
            ],
            verbose=1
        )
        
        # Evaluate
        test_pred = model.predict(X_test_seq)
        test_pred = scaler_y.inverse_transform(test_pred)
        test_true = scaler_y.inverse_transform(y_test_seq)
        
        print("\nValidation Metrics:")
        print(f"MAE: {mean_absolute_error(test_true, test_pred):.2f}")
        print(f"RMSE: {np.sqrt(mean_squared_error(test_true, test_pred)):.2f}")
        print(f"R²: {r2_score(test_true, test_pred):.2f}")


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