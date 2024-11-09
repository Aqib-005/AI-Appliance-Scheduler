import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import os

# Load dataset and preprocess as before
data = pd.read_csv("data/merged-data.csv").dropna()

# Process datetime and target column as before
data['Price Germany/Luxembourg [Euro/MWh]'] = data['Price Germany/Luxembourg [Euro/MWh]'].replace({',': ''}, regex=True).astype(float)
data['Total (grid consumption) [MWh]'] = data['Total (grid consumption) [MWh]'].replace({',': ''}, regex=True).astype(float)
data['Start date/time'] = pd.to_datetime(data['Start date/time'], dayfirst=True)
data['Year'] = data['Start date/time'].dt.year
data['Month'] = data['Start date/time'].dt.month
data['Day'] = data['Start date/time'].dt.day
data['Hour'] = data['Start date/time'].dt.hour

# Define features and target
target = 'Price Germany/Luxembourg [Euro/MWh]'
features = [
    'temperature_2m (°C)', 'relative_humidity_2m (%)', 'precipitation (mm)',
    'rain (mm)', 'snowfall (cm)', 'weather_code (wmo code)', 'wind_speed_100m (km/h)',
    'Total (grid consumption) [MWh]', 'Day of the Week', 'Year', 'Month', 'Day', 'Hour'
]

X = data[features]
y = data[target]

# impute missing target values 
data[target] = data[target].fillna(data[target].mean())

# Split data by date for training (2022-2023) and testing (2023-2024)
train_data = data[data['Start date/time'] < '2023-09-30']
test_data = data[data['Start date/time'] >= '2023-09-30']
X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]

# Preprocess features and scale target
numeric_features = [...]  # numeric columns
categorical_features = [...]  # categorical columns

# Verify columns
for col in numeric_features + categorical_features:
    if col not in X_train.columns:
        print(f"Column '{col}' is missing from X_train.")

# Update ColumnTransformer and ensure columns exist
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), [col for col in numeric_features if col in X_train.columns]),
        ('cat', OneHotEncoder(), [col for col in categorical_features if col in X_train.columns])
    ]
)

# Apply the preprocessor
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

# Scale target
y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))

# Reshape for LSTM
X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Define LSTM model
model = Sequential([
    Input(shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
    LSTM(50, return_sequences=True),
    Dropout(0.2),
    LSTM(30),
    Dropout(0.2),
    Dense(1)
])

# Compile model
optimizer = Adam(learning_rate=0.0001, clipvalue=1.0)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Train model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train_lstm, y_train_scaled, epochs=50, batch_size=16, validation_split=0.2, callbacks=[early_stopping])

# Evaluate model
y_pred_scaled = model.predict(X_test_lstm)
y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()
y_test_unscaled = y_scaler.inverse_transform(y_test_scaled).flatten()

mae = mean_absolute_error(y_test_unscaled, y_pred)
mse = mean_squared_error(y_test_unscaled, y_pred)
rmse = np.sqrt(mse)

print("LSTM Model Evaluation:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
