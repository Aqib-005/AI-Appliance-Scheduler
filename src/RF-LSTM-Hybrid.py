import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import xgboost as xgb

# Load and preprocess data
data = pd.read_csv("data/merged-data.csv").dropna()

# Convert columns to appropriate types
data['Price Germany/Luxembourg [Euro/MWh]'] = data['Price Germany/Luxembourg [Euro/MWh]'].replace({',': ''}, regex=True).astype(float)
data['Total (grid consumption) [MWh]'] = data['Total (grid consumption) [MWh]'].replace({',': ''}, regex=True).astype(float)
data['Start date/time'] = pd.to_datetime(data['Start date/time'], dayfirst=True)
data['Year'] = data['Start date/time'].dt.year
data['Month'] = data['Start date/time'].dt.month
data['Day'] = data['Start date/time'].dt.day
data['Hour'] = data['Start date/time'].dt.hour

# Define features and target, including weather data
target = 'Price Germany/Luxembourg [Euro/MWh]'
features = [
    'temperature_2m (°C)', 'relative_humidity_2m (%)', 'precipitation (mm)',
    'rain (mm)', 'snowfall (cm)', 'weather_code (wmo code)', 'wind_speed_100m (km/h)',
    'Total (grid consumption) [MWh]', 'Day of the Week', 'Year', 'Month', 'Day', 'Hour'
]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, shuffle=False)

# Preprocess features
numeric_features = [col for col in features if data[col].dtype in ['float64', 'int64']]
categorical_features = [col for col in features if col not in numeric_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

# Scale target
y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))

# Convert sparse matrix to dense matrix
X_train_scaled_dense = X_train_scaled.toarray() if hasattr(X_train_scaled, 'toarray') else X_train_scaled
X_test_scaled_dense = X_test_scaled.toarray() if hasattr(X_test_scaled, 'toarray') else X_test_scaled

# Reshape for LSTM
X_train_lstm = X_train_scaled_dense.reshape((X_train_scaled_dense.shape[0], 1, X_train_scaled_dense.shape[1]))
X_test_lstm = X_test_scaled_dense.reshape((X_test_scaled_dense.shape[0], 1, X_test_scaled_dense.shape[1]))

# Step 1: Train the LSTM model
lstm_model = Sequential([
    Input(shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
    LSTM(50, return_sequences=True),
    Dropout(0.2),
    LSTM(30, return_sequences=False),
    Dropout(0.2),
    Dense(10, activation='relu')  # Output intermediate features
])

optimizer = Adam(learning_rate=0.0001, clipvalue=1.0)
lstm_model.compile(optimizer=optimizer, loss='mean_squared_error')

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lstm_model.fit(X_train_lstm, y_train_scaled, epochs=50, batch_size=16, validation_split=0.2, callbacks=[early_stopping])

# Extract LSTM intermediate features
intermediate_features_train = lstm_model.predict(X_train_lstm)
intermediate_features_test = lstm_model.predict(X_test_lstm)

# Ensure compatibility for concatenation
if intermediate_features_train.ndim == 1:
    intermediate_features_train = intermediate_features_train.reshape(-1, 1)
if intermediate_features_test.ndim == 1:
    intermediate_features_test = intermediate_features_test.reshape(-1, 1)

# Concatenate original features and intermediate LSTM features
X_train_xgb = np.concatenate([X_train_scaled_dense, intermediate_features_train], axis=1)
X_test_xgb = np.concatenate([X_test_scaled_dense, intermediate_features_test], axis=1)

# Step 2: Train XGBoost on the combined features
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)

# Define hyperparameters for GridSearchCV
param_grid = {
    'max_depth': [6, 8, 10],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'n_estimators': [100, 200]
}

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1, scoring='neg_mean_squared_error')
grid_search.fit(X_train_xgb, y_train)

# Get best model from grid search
best_xgb_model = grid_search.best_estimator_

# Step 3: Evaluate the model
y_pred_xgb = best_xgb_model.predict(X_test_xgb)
y_pred_xgb_unscaled = y_scaler.inverse_transform(y_pred_xgb.reshape(-1, 1)).flatten()
y_test_unscaled = y_scaler.inverse_transform(y_test_scaled).flatten()

mae_xgb = mean_absolute_error(y_test_unscaled, y_pred_xgb_unscaled)
mse_xgb = mean_squared_error(y_test_unscaled, y_pred_xgb_unscaled)
rmse_xgb = np.sqrt(mse_xgb)
r2_xgb = r2_score(y_test_unscaled, y_pred_xgb_unscaled)

# Output results
print("LSTM-XGBoost Hybrid Model Evaluation:")
print("Mean Absolute Error (MAE):", mae_xgb)
print("Mean Squared Error (MSE):", mse_xgb)
print("Root Mean Squared Error (RMSE):", rmse_xgb)
print("R-squared:", r2_xgb)
