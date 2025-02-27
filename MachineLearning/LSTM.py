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
import keras_tuner as kt  # Ensure keras-tuner is installed

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

# --- Utility Functions ---
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
        
        # Map expected columns
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
        
        # Convert datetime (assume dayfirst)
        df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True, errors='coerce')
        if df['datetime'].isnull().any():
            raise ValueError("Invalid datetime values – check format (DD/MM/YYYY HH:MM)")
        
        # Convert dow to numeric
        dow_map = {"monday": 0, "tuesday": 1, "wednesday": 2, 
                   "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}
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
    # Cyclical features
    df['hour_sin'] = np.sin(2 * np.pi * df['datetime'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['datetime'].dt.hour / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
    # Lag features for price
    for lag in [1, 6, 24, 48]:
        df[f'price_lag_{lag}'] = df['price'].shift(lag)
    # Rolling mean (24-hour) for temperature and consumption
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

def create_sequences_with_timestamps(data_array, timestamps, window=24):
    """
    Creates sequences from data_array (already scaled) and returns:
      - X_seq: np.array of shape (num_sequences, window, num_features)
      - end_timestamps: list of timestamps corresponding to the end of each sequence.
    """
    X_seq = []
    end_timestamps = []
    for i in range(len(data_array) - window):
        X_seq.append(data_array[i:i+window])
        end_timestamps.append(timestamps[i+window])
    return np.array(X_seq), end_timestamps

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

# --- Main Execution ---
def main():
    try:
        DATA_PATH = "data/merged-data.csv"
        df = load_and_preprocess(DATA_PATH)
        df = create_temporal_features(df)
        
        # Define feature order and target
        FEATURES = [
            'temperature_2m', 'total_consumption', 'wind_speed_100m',
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
            'price_lag_1', 'price_lag_6', 'price_lag_24',
            'temperature_2m_roll24_mean', 'total_consumption_roll24_mean',
            'temp_load', 'wind_temp'
        ]
        TARGET = 'price'
        
        # Scale features and target on all available (historical) data
        scaler_x = StandardScaler()
        scaler_y = MinMaxScaler()
        X = scaler_x.fit_transform(df[FEATURES])
        y = scaler_y.fit_transform(df[[TARGET]])
        
        window = 24
        X_seq, y_seq = create_sequences(X, y, window)
        print(f"Training on {X_seq.shape[0]} sequences of length {window}.")
        
        # --- Hyperparameter Tuning ---
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
        
        # --- Train the Best Model Further ---
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
        
        # ---------- Custom Forecast Using Future Data from Dataset ----------
        # Assume your dataset now includes accurate forecasted features beyond the historical period.
        # Define custom forecast period:
        custom_start = pd.to_datetime("2025-01-06 00:00")
        custom_end = pd.to_datetime("2025-01-13 00:00")
        custom_df = df[(df['datetime'] >= custom_start) & (df['datetime'] < custom_end)].reset_index(drop=True)
        if custom_df.empty:
            raise ValueError("No future data found for the custom forecast period in the dataset.")
        
        # For sequence creation, we need the preceding window from historical data:
        history_window = df[df['datetime'] < custom_start].tail(window)
        combined_df = pd.concat([history_window, custom_df]).reset_index(drop=True)
        combined_features = scaler_x.transform(combined_df[FEATURES])
        combined_timestamps = combined_df['datetime'].tolist()
        X_combined_seq, seq_timestamps = create_sequences_with_timestamps(combined_features, combined_timestamps, window)
        
        # Since combined_df length = window + len(custom_df), the last len(custom_df) sequences correspond to the forecast period.
        custom_pred_seq = X_combined_seq[-len(custom_df):]
        custom_pred = best_model.predict(custom_pred_seq)
        custom_pred_inv = scaler_y.inverse_transform(custom_pred).flatten()
        
        # Build DataFrame for custom forecast predictions
        # The corresponding timestamps are the last len(custom_df) timestamps from seq_timestamps.
        pred_timestamps = seq_timestamps[-len(custom_df):]
        custom_forecast = pd.DataFrame({
            'datetime': pred_timestamps,
            'predicted_price': custom_pred_inv
        })
        
        # Build DataFrame with actual prices from the dataset for the custom period
        actual_df = custom_df[['datetime', TARGET]].rename(columns={TARGET: 'actual_price'})
        
        # Evaluate custom forecast
        mae_custom = mean_absolute_error(actual_df['actual_price'], custom_forecast['predicted_price'])
        rmse_custom = np.sqrt(mean_squared_error(actual_df['actual_price'], custom_forecast['predicted_price']))
        r2_custom = r2_score(actual_df['actual_price'], custom_forecast['predicted_price'])
        
        print("\nCustom Forecast Evaluation Metrics (06-01-2025 to 13-01-2025):")
        print(f"MAE: {mae_custom:.2f}")
        print(f"RMSE: {rmse_custom:.2f}")
        print(f"R²: {r2_custom:.2f}")
        
        # Plot predicted vs. actual prices
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
