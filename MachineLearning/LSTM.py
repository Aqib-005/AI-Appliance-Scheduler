import os
import pandas as pd
import numpy as np
import pywt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import keras_tuner as kt

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Layer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

# ----------------------------------------------------------------
#  Custom Attention Layer (with serialization decorator)
# ----------------------------------------------------------------
@tf.keras.utils.register_keras_serializable()
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

# ----------------------------------------------------------------
#  Wavelet Denoising Function
# ----------------------------------------------------------------
def wavelet_denoise(series, wavelet='db4', level=1):
    arr = np.array(series, dtype=np.float64)
    coeffs = pywt.wavedec(arr, wavelet, mode='periodization', level=level)
    for i in range(1, len(coeffs)):
        coeffs[i] = np.zeros_like(coeffs[i])
    filtered = pywt.waverec(coeffs, wavelet, mode='periodization')
    return filtered[:len(arr)]

# ----------------------------------------------------------------
#  Utility Functions
# ----------------------------------------------------------------
def clean_column_name(col: str) -> str:
    return (col.strip().lower()
            .replace(' ', '_')
            .replace('-', '_')
            .replace('/', '_')
            .replace('(', '')
            .replace(')', ''))

def load_and_preprocess(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print("Original columns detected:", df.columns.tolist())
    df.columns = [clean_column_name(col) for col in df.columns]
    print("Standardized columns:", df.columns.tolist())
    
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
    
    df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True, errors='coerce')
    if df['datetime'].isnull().any():
        raise ValueError("Invalid datetime values – check format (DD/MM/YYYY HH:MM)")
    
    dow_map = {"monday": 0, "tuesday": 1, "wednesday": 2,
               "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}
    df['dow'] = df['dow'].str.strip().str.lower().map(dow_map)
    if df['dow'].isnull().any():
        raise ValueError("Invalid day-of-week values found")
    
    numeric_cols = ['temperature_2m', 'total_consumption', 'wind_speed_100m']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    df = df.dropna(subset=numeric_cols)
    
    df = df.sort_values('datetime').reset_index(drop=True)
    print("\nData validation successful!")
    print("Final columns:", df.columns.tolist())
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Total records: {len(df)}")
    return df

def wavelet_transform_dataframe(df: pd.DataFrame, wavelet='db4', level=1) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].notna().any():
            arr = df[col].ffill()  # forward-fill missing values
            filtered = wavelet_denoise(arr, wavelet=wavelet, level=level)
            df[col] = filtered
    return df

def create_cyclical_feature(vals, max_val):
    vals_sin = np.sin(2 * np.pi * vals / max_val)
    vals_cos = np.cos(2 * np.pi * vals / max_val)
    return vals_sin, vals_cos

def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['hour_sin'] = np.sin(2 * np.pi * df['datetime'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['datetime'].dt.hour / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
    df['month_sin'], df['month_cos'] = create_cyclical_feature(df['datetime'].dt.month, 12)
    df['day_of_year_sin'], df['day_of_year_cos'] = create_cyclical_feature(df['datetime'].dt.dayofyear, 365)
    for lag in [1, 6, 24, 48, 72]:
        df[f'price_lag_{lag}'] = df['price'].shift(lag)
    for col, windows in {'temperature_2m': [24, 72], 'total_consumption': [24, 72]}.items():
        for w in windows:
            df[f'{col}_roll{w}_mean'] = df[col].rolling(w, min_periods=1).mean()
    df['temp_load'] = df['temperature_2m'] * df['total_consumption']
    df['wind_temp'] = df['wind_speed_100m'] / (df['temperature_2m'] + 1e-6)
    df = df.dropna().reset_index(drop=True)
    return df

def create_sequences(X, y, window=24):
    X_seq, y_seq = [], []
    for i in range(len(X) - window):
        X_seq.append(X[i:i+window])
        y_seq.append(y[i+window])
    return np.array(X_seq), np.array(y_seq)

def create_sequences_with_timestamps(data_array, timestamps, window=24):
    X_seq = []
    end_timestamps = []
    for i in range(len(data_array) - window):
        X_seq.append(data_array[i:i+window])
        end_timestamps.append(timestamps[i+window])
    return np.array(X_seq), end_timestamps

# ----------------------------------------------------------------
#  Build Model with Two LSTM Layers and Attention
# ----------------------------------------------------------------
def build_model(hp, input_shape):
    model = Sequential()
    model.add(tf.keras.Input(shape=input_shape))
    
    lstm_units = hp.Int('lstm_units', min_value=64, max_value=256, step=64)
    model.add(LSTM(lstm_units, return_sequences=True, kernel_regularizer=L1L2(1e-5, 1e-4)))
    
    dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
    model.add(Dropout(dropout_rate))
    
    second_lstm_units = hp.Int('second_lstm_units', min_value=32, max_value=128, step=32)
    model.add(LSTM(second_lstm_units, return_sequences=True, kernel_regularizer=L1L2(1e-5, 1e-4)))
    model.add(Dropout(dropout_rate))
    
    att_units = hp.Int('att_units', min_value=32, max_value=128, step=32)
    model.add(AttentionLayer(att_units))
    
    dense_units = hp.Int('dense_units', min_value=16, max_value=64, step=16)
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(1))
    
    lr = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
    model.compile(optimizer=Adam(learning_rate=lr, clipnorm=1.0),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])
    return model

# ----------------------------------------------------------------
#  Calculate MAPE
# ----------------------------------------------------------------
def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero_idx = y_true != 0
    return np.mean(np.abs((y_true[nonzero_idx] - y_pred[nonzero_idx]) / y_true[nonzero_idx])) * 100

# ----------------------------------------------------------------
#  Iterative Forecast Function
# ----------------------------------------------------------------
def iterative_forecast(model, df, start_time, end_time, scaler_x, scaler_y, FEATURES, window=24):
    hist_df = df[df['datetime'] < start_time].tail(window).copy()
    if len(hist_df) < window:
        raise ValueError("Not enough historical data for iterative forecast.")
    
    forecast_times = pd.date_range(start=start_time, end=end_time, freq='H')[:-1]
    predictions = []
    forecast_df = hist_df.copy().reset_index(drop=True)
    predicted_indices = {}
    
    for t in forecast_times:
        if t in df['datetime'].values:
            future_row = df[df['datetime'] == t].iloc[0]
        else:
            future_row = forecast_df.iloc[-1].copy()
            future_row['datetime'] = t
        
        for idx, pred_val in predicted_indices.items():
            forecast_df.loc[idx, 'price'] = pred_val
        
        sub_df = forecast_df.iloc[-window:]
        X_sub = scaler_x.transform(sub_df[FEATURES].values)
        X_sub = X_sub.reshape(1, window, len(FEATURES))
        pred_scaled = model.predict(X_sub)
        pred_price = scaler_y.inverse_transform(pred_scaled)[0, 0]
        
        predictions.append((t, pred_price))
        predicted_indices[len(forecast_df)] = pred_price
        
        new_row = future_row.copy()
        new_row['price'] = pred_price
        # Replace deprecated append with pd.concat:
        forecast_df = pd.concat([forecast_df, new_row.to_frame().T], ignore_index=True)
    
    forecast_results = pd.DataFrame(predictions, columns=['datetime', 'predicted_price'])
    return forecast_results

# ----------------------------------------------------------------
#  Main Execution
# ----------------------------------------------------------------
def main():
    try:
        DATA_PATH = "data/merged-data.csv"
        df = load_and_preprocess(DATA_PATH)
        df = wavelet_transform_dataframe(df, wavelet='db4', level=1)
        df = create_temporal_features(df)
        
        FEATURES = [
            'temperature_2m', 'total_consumption', 'wind_speed_100m',
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
            'month_sin', 'month_cos',
            'day_of_year_sin', 'day_of_year_cos',
            'price_lag_1', 'price_lag_6', 'price_lag_24', 'price_lag_48', 'price_lag_72',
            'temperature_2m_roll24_mean', 'temperature_2m_roll72_mean',
            'total_consumption_roll24_mean', 'total_consumption_roll72_mean',
            'temp_load', 'wind_temp'
        ]
        TARGET = 'price'
        
        train_df = df[df[TARGET].notna()].copy()
        scaler_x = StandardScaler()
        scaler_y = MinMaxScaler()
        X_train = scaler_x.fit_transform(train_df[FEATURES])
        y_train = scaler_y.fit_transform(train_df[[TARGET]])
        
        window = 24
        X_seq, y_seq = create_sequences(X_train, y_train, window)
        print(f"Training on {X_seq.shape[0]} sequences of length {window}.")
        
        input_shape = (window, len(FEATURES))
        model_path = "best_model_v2.h5"
        
        if os.path.exists(model_path):
            best_model = load_model(model_path, custom_objects={'AttentionLayer': AttentionLayer, 'mse': tf.keras.losses.MeanSquaredError})
            print("Loaded saved model.")
        else:
            tuner = kt.BayesianOptimization(
                lambda hp: build_model(hp, input_shape),
                objective='val_loss',
                max_trials=5,
                directory='tuner_dir_v2',
                project_name='electricity_price_improved'
            )
            tuner.search(X_seq, y_seq, epochs=30, validation_split=0.2,
                         callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
                         verbose=1)
            best_model = tuner.get_best_models(num_models=1)[0]
            best_model.fit(X_seq, y_seq, epochs=50, batch_size=64,
                           callbacks=[EarlyStopping(patience=10, restore_best_weights=True),
                                      ReduceLROnPlateau(factor=0.2, patience=3)],
                           verbose=1)
            best_model.save(model_path)
            print("Saved best model to disk.")
        
        train_pred = best_model.predict(X_seq)
        train_pred_inv = scaler_y.inverse_transform(train_pred)
        train_true_inv = scaler_y.inverse_transform(y_seq)
        
        mae_train = mean_absolute_error(train_true_inv, train_pred_inv)
        rmse_train = np.sqrt(mean_squared_error(train_true_inv, train_pred_inv))
        r2_train = r2_score(train_true_inv, train_pred_inv)
        mape_train = calculate_mape(train_true_inv, train_pred_inv)
        print("\nTraining Metrics:")
        print(f"MAE: {mae_train:.2f}")
        print(f"RMSE: {rmse_train:.2f}")
        print(f"R²: {r2_train:.2f}")
        print(f"MAPE: {mape_train:.2f}%")
        
        custom_start = pd.to_datetime("2025-01-06 00:00")
        custom_end = pd.to_datetime("2025-01-13 00:00")
        
        hist_df = df[df['datetime'] < custom_start]
        if len(hist_df) < window:
            raise ValueError("Not enough historical data for iterative forecast.")
        
        forecast_df = iterative_forecast(
            model=best_model,
            df=df,
            start_time=custom_start,
            end_time=custom_end,
            scaler_x=scaler_x,
            scaler_y=scaler_y,
            FEATURES=FEATURES,
            window=window
        )
        
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
        custom_dates = pd.date_range(start="2025-01-06 00:00", end="2025-01-13 00:00", freq='H')[:-1]
        if len(custom_dates) != len(actual_prices):
            raise ValueError("The number of hours in the custom period does not match the length of actual_prices.")
        
        actual_df = pd.DataFrame({'datetime': custom_dates, 'actual_price': actual_prices})
        compare_df = pd.merge(forecast_df, actual_df, on='datetime')
        
        mae_custom = mean_absolute_error(compare_df['actual_price'], compare_df['predicted_price'])
        rmse_custom = np.sqrt(mean_squared_error(compare_df['actual_price'], compare_df['predicted_price']))
        r2_custom = r2_score(compare_df['actual_price'], compare_df['predicted_price'])
        mape_custom = calculate_mape(compare_df['actual_price'], compare_df['predicted_price'])
        
        print("\nCustom Forecast Evaluation Metrics (06-01-2025 to 13-01-2025):")
        print(f"MAE: {mae_custom:.2f}")
        print(f"RMSE: {rmse_custom:.2f}")
        print(f"R²: {r2_custom:.2f}")
        print(f"MAPE: {mape_custom:.2f}%")
        
        plt.figure(figsize=(14,7))
        plt.plot(compare_df['datetime'], compare_df['predicted_price'], label='Predicted Price', marker='o')
        plt.plot(compare_df['datetime'], compare_df['actual_price'], label='Actual Price', marker='x')
        plt.xlabel('Datetime')
        plt.ylabel('Electricity Price')
        plt.title('Iterative Predicted vs Actual Electricity Price (06-01-2025 to 13-01-2025)')
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
