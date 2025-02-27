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
import keras_tuner as kt  # Make sure you have keras-tuner installed

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
            matches = [col for col in df.columns if any(p == col or p in col for p in patterns)]
            if not matches:
                raise ValueError(f"Missing column matching patterns: {patterns}")
            final_columns[matches[0]] = target
        
        df = df.rename(columns=final_columns)[list(expected_columns.keys())]
        
        # Convert datetime column (assume dayfirst)
        df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True, errors='coerce')
        if df['datetime'].isnull().any():
            raise ValueError("Invalid datetime values – check format (DD/MM/YYYY HH:MM)")
        
        # Convert day-of-week to numeric
        dow_map = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
                   "friday": 4, "saturday": 5, "sunday": 6}
        df['dow'] = df['dow'].str.strip().str.lower().map(dow_map)
        if df['dow'].isnull().any():
            raise ValueError("Invalid day-of-week values found")
        
        # Convert numeric columns and drop rows with invalid values
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

# --- Hypermodel for Keras Tuner ---
def build_model(hp, input_shape):
    model = Sequential()
    lstm_units = hp.Int('lstm_units', min_value=64, max_value=256, step=64)
    model.add(LSTM(lstm_units, return_sequences=True, input_shape=input_shape,
                   kernel_regularizer=L1L2(1e-5, 1e-4)))
    dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
    model.add(Dropout(dropout_rate))
    att_units = hp.Int('att_units', min_value=32, max_value=128, step=32)
    model.add(AttentionLayer(att_units))
    dense_units = hp.Int('dense_units', min_value=16, max_value=64, step=16)
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(1))
    lr = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
    model.compile(optimizer=Adam(learning_rate=lr, clipnorm=1.0), loss='mse', metrics=['mae'])
    return model

# --- Forecasting from a given initial sequence ---
def forecast_from_seq(model, initial_seq, initial_datetime, forecast_horizon, initial_price_history, 
                      FEATURES, TARGET, scaler_x, scaler_y):
    current_seq = initial_seq.copy()  # shape: (window, num_features)
    current_dt = initial_datetime
    price_history = initial_price_history.copy()
    
    forecast_timestamps = []
    predictions = []
    
    # We'll use the global raw constants for exogenous features (set in main)
    global const_temp_raw, const_cons_raw, const_wind_raw, const_temp_roll24_raw, const_cons_roll24_raw
    
    for i in range(forecast_horizon):
        X_input = current_seq.reshape(1, current_seq.shape[0], current_seq.shape[1])
        pred_scaled = model.predict(X_input)
        pred_price = scaler_y.inverse_transform(pred_scaled)[0, 0]
        
        current_dt = current_dt + pd.Timedelta(hours=1)
        forecast_timestamps.append(current_dt)
        predictions.append(pred_price)
        
        price_history.append(pred_price)
        hour = current_dt.hour
        dow = current_dt.dayofweek
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        dow_sin = np.sin(2 * np.pi * dow / 7)
        dow_cos = np.cos(2 * np.pi * dow / 7)
        lag1 = price_history[-1]
        lag6 = price_history[-6] if len(price_history) >= 6 else price_history[0]
        lag24 = price_history[-24] if len(price_history) >= 24 else price_history[0]
        
        new_features = np.array([
            const_temp_raw,
            const_cons_raw,
            const_wind_raw,
            hour_sin,
            hour_cos,
            dow_sin,
            dow_cos,
            lag1,
            lag6,
            lag24,
            const_temp_roll24_raw,
            const_cons_roll24_raw,
            const_temp_raw * const_cons_raw,
            const_wind_raw / (const_temp_raw + 1e-6)
        ])
        new_features_scaled = scaler_x.transform(new_features.reshape(1, -1))[0]
        current_seq = np.vstack([current_seq[1:], new_features_scaled])
    
    forecast_df = pd.DataFrame({
        'datetime': forecast_timestamps,
        'predicted_price': predictions
    })
    return forecast_df

# --- Main execution ---
def main():
    try:
        DATA_PATH = "data/merged-data.csv"
        df = load_and_preprocess(DATA_PATH)
        df = create_temporal_features(df)
        
        # Define features in expected order
        FEATURES = [
            'temperature_2m', 'total_consumption', 'wind_speed_100m',
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
            'price_lag_1', 'price_lag_6', 'price_lag_24',
            'temperature_2m_roll24_mean', 'total_consumption_roll24_mean',
            'temp_load', 'wind_temp'
        ]
        TARGET = 'price'
        
        # Scale features and target using all historical data
        scaler_x = StandardScaler()
        scaler_y = MinMaxScaler()
        X = scaler_x.fit_transform(df[FEATURES])
        y = scaler_y.fit_transform(df[[TARGET]])
        
        window = 24
        X_seq, y_seq = create_sequences(X, y, window)
        print(f"Training on {X_seq.shape[0]} sequences of length {window}.")
        
        # --- Hyperparameter Tuning with Keras Tuner ---
        input_shape = (window, len(FEATURES))
        tuner = kt.RandomSearch(
            lambda hp: build_model(hp, input_shape),
            objective='val_loss',
            max_trials=5,
            directory='tuner_dir',
            project_name='electricity_price'
        )
        tuner.search(X_seq, y_seq, epochs=50, validation_split=0.2,
                     callbacks=[EarlyStopping(patience=5, restore_best_weights=True)], verbose=1)
        best_model = tuner.get_best_models(num_models=1)[0]
        print("Best hyperparameters:")
        print(tuner.get_best_hyperparameters(num_trials=1)[0].values)
        
        # --- Train best model further if desired ---
        history = best_model.fit(
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
        train_pred = best_model.predict(X_seq)
        train_pred_inv = scaler_y.inverse_transform(train_pred)
        train_true_inv = scaler_y.inverse_transform(y_seq)
        print("\nTraining Metrics:")
        print(f"MAE: {mean_absolute_error(train_true_inv, train_pred_inv):.2f}")
        print(f"RMSE: {np.sqrt(mean_squared_error(train_true_inv, train_pred_inv)):.2f}")
        print(f"R²: {r2_score(train_true_inv, train_pred_inv):.2f}")
        
        # ---------- Overall Future Forecast (from training end) ----------
        last_train_dt = df['datetime'].max()  # e.g., 2025-01-01 23:00:00
        initial_seq = scaler_x.transform(df[FEATURES].iloc[-window:].values)
        initial_price_history = df[TARGET].iloc[-window:].tolist()
        # Set global raw exogenous features from last row:
        global const_temp_raw, const_cons_raw, const_wind_raw, const_temp_roll24_raw, const_cons_roll24_raw
        last_row = df.iloc[-1]
        const_temp_raw = last_row['temperature_2m']
        const_cons_raw = last_row['total_consumption']
        const_wind_raw = last_row['wind_speed_100m']
        const_temp_roll24_raw = last_row['temperature_2m_roll24_mean']
        const_cons_roll24_raw = last_row['total_consumption_roll24_mean']
        
        forecast_horizon = 168  # next week
        overall_forecast = forecast_from_seq(best_model, initial_seq, last_train_dt, forecast_horizon, 
                                             initial_price_history, FEATURES, TARGET, scaler_x, scaler_y)
        # Plot overall forecast (optional)
        plt.figure(figsize=(12,6))
        plt.plot(overall_forecast['datetime'], overall_forecast['predicted_price'], label='Overall Forecast')
        plt.xlabel('Datetime')
        plt.ylabel('Electricity Price')
        plt.title('Overall Future Electricity Price Forecast')
        plt.legend()
        plt.show()
        
        # ---------- Custom Forecast for Week 06-01-2025 to 13-01-2025 ----------
        custom_start = pd.to_datetime("2025-01-06 00:00")
        custom_horizon = 168  # 7 days
        gap_hours = int((custom_start - last_train_dt).total_seconds() // 3600)
        total_custom_horizon = gap_hours + custom_horizon
        
        custom_forecast_full = forecast_from_seq(best_model, initial_seq, last_train_dt, total_custom_horizon, 
                                                 initial_price_history, FEATURES, TARGET, scaler_x, scaler_y)
        custom_forecast = custom_forecast_full.iloc[gap_hours: gap_hours + custom_horizon].reset_index(drop=True)
        
        # Build DataFrame with actual custom prices (provided manually)
        actual_prices = [
            27.52, 19.26, 11.35, 9.20, 10.00, 14.81, 23.82, 31.72, 41.37, 36.36, 34.69, 32.83,
            31.28, 28.60, 25.61, 26.32, 26.87, 31.66, 31.58, 28.74, 26.59, 13.82, 26.94, 12.99,
            19.07, 8.71, 8.90, 5.01, 5.13, 5.80, 48.86, 76.83, 85.08, 84.24, 75.25, 62.80,
            62.44, 63.90, 72.56, 78.11, 79.98, 96.03, 101.11, 86.21, 78.01, 72.45, 72.45, 50.04,
            71.05, 68.01, 63.34, 57.01, 66.29, 72.07, 82.70, 100.73, 128.22, 108.18, 94.65, 100.01,
            89.99, 97.20, 110.19, 127.80, 135.82, 155.46, 149.23, 146.48, 136.86, 127.86, 115.92, 103.59,
            101.44, 100.00, 98.77, 95.22, 98.28, 102.65, 133.54, 148.80, 164.88, 156.15, 147.64, 140.21,
            128.18, 121.15, 123.95, 127.53, 123.75, 130.91, 134.75, 125.44, 119.23, 104.99, 101.10, 88.19,
            84.79, 80.23, 71.29, 69.05, 69.89, 83.90, 99.09, 123.92, 139.47, 136.92, 123.57, 113.59,
            107.43, 105.01, 110.01, 128.69, 134.87, 142.56, 144.12, 141.05, 134.82, 122.51, 119.77, 111.71,
            106.64, 98.99, 95.71, 89.45, 88.35, 88.40, 88.13, 94.68, 107.65, 103.14, 101.13, 99.56,
            96.74, 93.15, 94.90, 104.89, 112.12, 119.80, 122.34, 114.85, 111.19, 104.80, 103.07, 105.69,
            83.58, 83.79, 86.75, 84.20, 85.00, 83.75, 85.90, 101.75, 114.67, 120.07, 117.77, 105.61,
            102.99, 99.61, 104.74, 118.70, 132.92, 141.00, 146.66, 143.66, 134.86, 127.49, 121.20, 114.90
        ]
        custom_dates = pd.date_range(start=custom_start, periods=custom_horizon, freq='H')
        actual_df = pd.DataFrame({
            'datetime': custom_dates,
            'actual_price': actual_prices
        })
        
        mae_custom = mean_absolute_error(actual_df['actual_price'], custom_forecast['predicted_price'])
        rmse_custom = np.sqrt(mean_squared_error(actual_df['actual_price'], custom_forecast['predicted_price']))
        r2_custom = r2_score(actual_df['actual_price'], custom_forecast['predicted_price'])
        
        print("\nCustom Forecast Evaluation Metrics (06-01-2025 to 13-01-2025):")
        print(f"MAE: {mae_custom:.2f}")
        print(f"RMSE: {rmse_custom:.2f}")
        print(f"R²: {r2_custom:.2f}")
        
        # Plot the comparison graph
        plt.figure(figsize=(14,7))
        plt.plot(custom_forecast['datetime'], custom_forecast['predicted_price'], label='Predicted Price', marker='o')
        plt.plot(actual_df['datetime'], actual_df['actual_price'], label='Actual Price', marker='x')
        plt.xlabel('Datetime')
        plt.ylabel('Electricity Price')
        plt.title('Predicted vs Actual Electricity Price (06-01-2025 to 13-01-2025)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
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
