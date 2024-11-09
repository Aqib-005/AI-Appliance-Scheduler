import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

# Load data
data = pd.read_csv("data/merged-data.csv").dropna()

# Preprocess data and add weather data columns if available
data['Price Germany/Luxembourg [Euro/MWh]'] = data['Price Germany/Luxembourg [Euro/MWh]'].replace({',': ''}, regex=True).astype(float)
data['Total (grid consumption) [MWh]'] = data['Total (grid consumption) [MWh]'].replace({',': ''}, regex=True).astype(float)
data['Start date/time'] = pd.to_datetime(data['Start date/time'], dayfirst=True)
data['Year'] = data['Start date/time'].dt.year
data['Month'] = data['Start date/time'].dt.month
data['Day'] = data['Start date/time'].dt.day
data['Hour'] = data['Start date/time'].dt.hour

# Encode categorical features
label_encoder = LabelEncoder()
data['Day of the Week'] = label_encoder.fit_transform(data['Start date/time'].dt.day_name())

# Define features and target
target = 'Price Germany/Luxembourg [Euro/MWh]'
features = [
    'temperature_2m (°C)', 'relative_humidity_2m (%)', 'precipitation (mm)',
    'rain (mm)', 'snowfall (cm)', 'weather_code (wmo code)', 'wind_speed_100m (km/h)',
    'Total (grid consumption) [MWh]', 'Day of the Week', 'Year', 'Month', 'Day', 'Hour'
]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, shuffle=False)

# Preprocess features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Scale target for LSTM model
y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))

# Reshape for LSTM
X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# LSTM Model
lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
    Dropout(0.2),
    LSTM(30),
    Dropout(0.2),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train_lstm, y_train_scaled, epochs=20, batch_size=16, validation_split=0.2)

# LSTM Predictions
y_pred_lstm_scaled = lstm_model.predict(X_test_lstm)
y_pred_lstm = y_scaler.inverse_transform(y_pred_lstm_scaled).flatten()

# XGBoost Model with Hyperparameter Tuning
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
param_grid = {'max_depth': [6, 8], 'learning_rate': [0.01, 0.1], 'n_estimators': [100, 200]}
grid_search = GridSearchCV(xgb_model, param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)
best_xgb_model = grid_search.best_estimator_

# XGBoost Predictions
y_pred_xgb = best_xgb_model.predict(X_test_scaled)

# Ensemble Prediction (Averaging)
y_pred_ensemble = (y_pred_lstm + y_pred_xgb) / 2

# Evaluate
mae_ensemble = mean_absolute_error(y_test, y_pred_ensemble)
mse_ensemble = mean_squared_error(y_test, y_pred_ensemble)
rmse_ensemble = np.sqrt(mse_ensemble)
r2_ensemble = r2_score(y_test, y_pred_ensemble)

print("Ensemble Model Evaluation:")
print("Mean Absolute Error (MAE):", mae_ensemble)
print("Mean Squared Error (MSE):", mse_ensemble)
print("Root Mean Squared Error (RMSE):", rmse_ensemble)
print("R-squared:", r2_ensemble)
