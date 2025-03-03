import os
import numpy as np
import pandas as pd
import pywt
import tensorflow as tf
import matplotlib.pyplot as plt
import keras_tuner as kt

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Layer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

# -----------------------------
# Custom Attention Layer (with serialization)
# -----------------------------
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

# -----------------------------
# Wavelet Denoising (Exogenous Only)
# -----------------------------
def wavelet_denoise_exog(series, wavelet='db4', level=1):
    arr = np.array(series, dtype=np.float64)
    coeffs = pywt.wavedec(arr, wavelet, mode='periodization', level=level)
    for i in range(1, len(coeffs)):
        coeffs[i] = np.zeros_like(coeffs[i])
    filtered = pywt.waverec(coeffs, wavelet, mode='periodization')
    return filtered[:len(arr)]

# -----------------------------
# Data Loading & Preprocessing
# -----------------------------
def load_and_preprocess(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print("Original columns detected:", df.columns.tolist())
    
    # Clean column names:
    # Strip spaces, lower-case, replace spaces, hyphens, and slashes with underscores,
    # and remove parentheses.
    df.columns = (df.columns.str.strip()
                         .str.lower()
                         .str.replace(' ', '_')
                         .str.replace('-', '_')
                         .str.replace('/', '_')
                         .str.replace('[()]', '', regex=True))
    print("Standardized columns:", df.columns.tolist())
    
    # Rename columns to match our code expectations:
    rename_dict = {
        'start_date_time': 'datetime',
        'day_price': 'price',
        'grid_load': 'total_consumption',
        'wind_speed_100m_kmh': 'wind_speed_100m'
    }
    df.rename(columns=rename_dict, inplace=True)
    
    if 'datetime' not in df.columns:
        raise ValueError("No 'datetime' column found. Rename your time column to 'datetime'.")
    df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Convert numeric columns (remove commas)
    numeric_cols = ['temperature_2m', 'total_consumption', 'wind_speed_100m']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    
    print("\nData loaded and preprocessed.")
    print("Columns:", df.columns.tolist())
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Total records: {len(df)}")
    return df

# -----------------------------
# Feature Transformation
# -----------------------------
def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Apply wavelet denoising only for exogenous columns (do NOT denoise 'price')
    exog_cols = ['temperature_2m', 'wind_speed_100m', 'total_consumption']
    for col in exog_cols:
        if col in df.columns:
            df[col] = df[col].ffill()  # forward-fill missing values
            df[col] = wavelet_denoise_exog(df[col])
    
    # Create cyclical time features
    df['hour'] = df['datetime'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    df['dow'] = df['datetime'].dt.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
    
    df['month'] = df['datetime'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Day-of-year features
    df['day_of_year'] = df['datetime'].dt.dayofyear
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # Create price lag features
    if 'price' not in df.columns:
        raise ValueError("No 'price' column found. Ensure your dataset has a 'price' column.")
    for lag in [1, 6, 24, 48, 72]:
        df[f'price_lag_{lag}'] = df['price'].shift(lag)
    
    # Create rolling means for exogenous features
    for col, windows in {'temperature_2m': [24, 72], 'total_consumption': [24, 72]}.items():
        if col in df.columns:
            for w in windows:
                df[f'{col}_roll{w}_mean'] = df[col].rolling(w, min_periods=1).mean()
    
    # Interaction features
    if 'temperature_2m' in df.columns and 'total_consumption' in df.columns:
        df['temp_load'] = df['temperature_2m'] * df['total_consumption']
    if 'wind_speed_100m' in df.columns and 'temperature_2m' in df.columns:
        df['wind_temp'] = df['wind_speed_100m'] / (df['temperature_2m'] + 1e-6)
    
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# -----------------------------
# Create Sequences for LSTM
# -----------------------------
def create_sequences(X, y, window=24):
    X_seq, y_seq = [], []
    for i in range(len(X) - window):
        X_seq.append(X[i:i+window])
        y_seq.append(y[i+window])
    return np.array(X_seq), np.array(y_seq)

# -----------------------------
# Build the LSTM + Attention Model
# -----------------------------
@tf.keras.utils.register_keras_serializable()
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

# -----------------------------
# MAPE Calculation
# -----------------------------
def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero_idx = y_true != 0
    return np.mean(np.abs((y_true[nonzero_idx] - y_pred[nonzero_idx]) / y_true[nonzero_idx])) * 100

# -----------------------------
# Iterative Forecast Function
# -----------------------------
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
        forecast_df = pd.concat([forecast_df, new_row.to_frame().T], ignore_index=True)
    
    forecast_results = pd.DataFrame(predictions, columns=['datetime', 'predicted_price'])
    return forecast_results

# -----------------------------
# Main Execution
# -----------------------------
def main():
    try:
        DATA_PATH = "data/merged-data.csv"
        df = load_and_preprocess(DATA_PATH)
        df = transform_features(df)
        
        # Define features and target; target is 'price'
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
        
        # Restrict dataset to 01/01/2023 - 01/01/2025
        df = df[(df['datetime'] >= pd.to_datetime("01/01/2023 00:00:00")) &
                (df['datetime'] <= pd.to_datetime("01/01/2025 23:00:00"))]
        
        train_df = df[df[TARGET].notna()].copy()
        scaler_x = StandardScaler()
        scaler_y = RobustScaler()
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
                max_trials=10,
                directory='tuner_dir_v2',
                project_name='electricity_price_improved'
            )
            tuner.search(X_seq, y_seq, epochs=40, validation_split=0.2,
                         callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
                         verbose=1)
            best_model = tuner.get_best_models(num_models=1)[0]
            best_model.fit(X_seq, y_seq, epochs=60, batch_size=64,
                           callbacks=[EarlyStopping(patience=10, restore_best_weights=True),
                                      ReduceLROnPlateau(factor=0.2, patience=3)],
                           verbose=1)
            best_model.save(model_path)
            print("Saved best model to disk.")
        
        train_pred = best_model.predict(X_seq)
        train_pred_inv = scaler_y.inverse_transform(train_pred)
        train_true_inv = scaler_y.inverse_transform(y_seq.reshape(-1, 1))
        
        mae_train = mean_absolute_error(train_true_inv, train_pred_inv)
        rmse_train = np.sqrt(mean_squared_error(train_true_inv, train_pred_inv))
        r2_train = r2_score(train_true_inv, train_pred_inv)
        mape_train = calculate_mape(train_true_inv, train_pred_inv)
        print("\nTraining Metrics:")
        print(f"MAE: {mae_train:.2f}")
        print(f"RMSE: {rmse_train:.2f}")
        print(f"R²: {r2_train:.2f}")
        print(f"MAPE: {mape_train:.2f}%")
        
        # Set custom forecast period within dataset range
        custom_start = pd.to_datetime("2025-01-01 00:00")
        custom_end = pd.to_datetime("2025-01-01 23:00")
        max_date = df['datetime'].max()
        if custom_end > max_date:
            print(f"Custom end {custom_end} exceeds dataset max date {max_date}, adjusting custom_end.")
            custom_end = max_date
        
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
            31.28, 28.60, 25.61, 26.32, 26.87, 31.66, 31.58, 28.74, 26.59, 13.82, 26.94, 12.99
        ]
        custom_dates = pd.date_range(start=custom_start, end=custom_end, freq='H')[:-1]
        if len(custom_dates) != len(actual_prices):
            raise ValueError("The number of hours in the custom period does not match the length of actual_prices.")
        
        actual_df = pd.DataFrame({'datetime': custom_dates, 'actual_price': actual_prices})
        compare_df = pd.merge(forecast_df, actual_df, on='datetime')
        
        mae_custom = mean_absolute_error(compare_df['actual_price'], compare_df['predicted_price'])
        rmse_custom = np.sqrt(mean_squared_error(compare_df['actual_price'], compare_df['predicted_price']))
        r2_custom = r2_score(compare_df['actual_price'], compare_df['predicted_price'])
        mape_custom = calculate_mape(compare_df['actual_price'], compare_df['predicted_price'])
        
        print("\nCustom Forecast Evaluation Metrics:")
        print(f"MAE: {mae_custom:.2f}")
        print(f"RMSE: {rmse_custom:.2f}")
        print(f"R²: {r2_custom:.2f}")
        print(f"MAPE: {mape_custom:.2f}%")
        
        plt.figure(figsize=(14,7))
        plt.plot(compare_df['datetime'], compare_df['predicted_price'], label='Predicted Price', marker='o')
        plt.plot(compare_df['datetime'], compare_df['actual_price'], label='Actual Price', marker='x')
        plt.xlabel('Datetime')
        plt.ylabel('Electricity Price')
        plt.title('Iterative Predicted vs Actual Electricity Price')
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
