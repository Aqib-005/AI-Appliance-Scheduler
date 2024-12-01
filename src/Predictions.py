import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam

# Load the data
data = pd.read_csv("data/merged-data.csv")

# Data cleaning and preprocessing
data['Price Germany/Luxembourg [Euro/MWh]'] = data['Price Germany/Luxembourg [Euro/MWh]'].replace({',': ''}, regex=True).astype(float)
data['Total (grid consumption) [MWh]'] = data['Total (grid consumption) [MWh]'].replace({',': ''}, regex=True).astype(float)
data['Start date/time'] = pd.to_datetime(data['Start date/time'], dayfirst=True)
data['Year'] = data['Start date/time'].dt.year
data['Month'] = data['Start date/time'].dt.month
data['Day'] = data['Start date/time'].dt.day
data['Hour'] = data['Start date/time'].dt.hour
data['Lag_Price'] = data['Price Germany/Luxembourg [Euro/MWh]'].shift(1)
data = data.dropna()

# Feature selection
features = [
    'temperature_2m (°C)', 
    'wind_speed_100m (km/h)', 
    'Total (grid consumption) [MWh]', 
    'Day', 
    'Hour', 
    'Lag_Price'
]
target = 'Price Germany/Luxembourg [Euro/MWh]'

# Split data
train_data = data[data['Start date/time'] < '2023-09-30']
test_data = data[data['Start date/time'] >= '2023-09-30']

# Initialize lists for storing results
rolling_predictions = []
actual_values = []
predictions_storage = []  # For saving predictions with timestamps
window_size = 30  # Rolling window size

# Rolling prediction loop
for start_idx in range(0, len(train_data) - window_size, window_size):
    end_idx = start_idx + window_size
    rolling_train_data = train_data.iloc[start_idx:end_idx]
    if rolling_train_data.empty:
        continue

    X_rolling_train = rolling_train_data[features]
    y_rolling_train = rolling_train_data[target]

    scaler = StandardScaler()
    X_rolling_train_scaled = scaler.fit_transform(X_rolling_train)
    y_train_mean, y_train_std = y_rolling_train.mean(), y_rolling_train.std()
    y_rolling_train_scaled = (y_rolling_train - y_train_mean) / y_train_std

    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf_random = RandomizedSearchCV(
        RandomForestRegressor(random_state=42),
        param_distributions=rf_param_grid,
        n_iter=10,
        cv=3,
        random_state=42,
        n_jobs=-1
    )
    rf_random.fit(X_rolling_train, y_rolling_train)
    best_rf = rf_random.best_estimator_

    rf_train_pred = best_rf.predict(X_rolling_train)
    rf_train_residuals = y_rolling_train - rf_train_pred
    rf_train_residuals_reshaped = rf_train_residuals.values.reshape(-1, 1, 1)

    X_train_lstm = np.concatenate([X_rolling_train_scaled.reshape((-1, 1, X_rolling_train_scaled.shape[1])), rf_train_residuals_reshaped], axis=2)

    lstm_model = Sequential()
    lstm_model.add(LSTM(64, activation='relu', return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
    lstm_model.add(LSTM(32, activation='relu'))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    lstm_model.fit(X_train_lstm, rf_train_residuals, epochs=10, batch_size=32, verbose=2, validation_split=0.2)

    test_start_date = train_data.iloc[end_idx]['Start date/time']
    test_end_date = test_start_date + pd.DateOffset(days=window_size)
    rolling_test_data = test_data[(test_data['Start date/time'] >= test_start_date) & (test_data['Start date/time'] < test_end_date)]
    if rolling_test_data.empty:
        continue

    X_rolling_test = rolling_test_data[features]
    y_rolling_test = rolling_test_data[target]
    X_rolling_test_scaled = scaler.transform(X_rolling_test)
    y_rolling_test_scaled = (y_rolling_test - y_train_mean) / y_train_std

    rf_test_pred = best_rf.predict(X_rolling_test)
    rf_test_residuals = y_rolling_test - rf_test_pred
    rf_test_residuals_reshaped = rf_test_residuals.values.reshape(-1, 1, 1)

    X_test_lstm = np.concatenate([X_rolling_test_scaled.reshape((-1, 1, X_rolling_test_scaled.shape[1])), rf_test_residuals_reshaped], axis=2)
    lstm_residual_predictions = lstm_model.predict(X_test_lstm)
    final_hybrid_predictions = rf_test_pred + lstm_residual_predictions.flatten()

    rolling_predictions.extend(final_hybrid_predictions)
    actual_values.extend(y_rolling_test.values)

    # Save predictions with corresponding timestamps
    predictions_storage.extend(zip(rolling_test_data['Start date/time'], final_hybrid_predictions))

# Save predictions to a CSV file
predictions_df = pd.DataFrame(predictions_storage, columns=['Timestamp', 'Predicted Price'])
predictions_df.to_csv("predictions.csv", index=False)

# Evaluate and plot
if rolling_predictions and actual_values:
    mse = mean_squared_error(actual_values, rolling_predictions)
    mae = mean_absolute_error(actual_values, rolling_predictions)
    r2 = r2_score(actual_values, rolling_predictions)
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    plt.figure(figsize=(10, 6))
    plt.plot(actual_values, label="True Values", color='blue')
    plt.plot(rolling_predictions, label="Predictions", color='orange')
    plt.legend()
    plt.show()
else:
    print("No predictions were generated.")
