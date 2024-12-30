import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Load the data
data = pd.read_csv("data/merged-data.csv")

# Data cleaning and preprocessing
data['Price Germany/Luxembourg [Euro/MWh]'] = data['Price Germany/Luxembourg [Euro/MWh]'].replace({',': ''}, regex=True).astype(float)
data['Total (grid consumption) [MWh]'] = data['Total (grid consumption) [MWh]'].replace({',': ''}, regex=True).astype(float)

# Convert Start date/time to datetime and extract useful time features
data['Start date/time'] = pd.to_datetime(data['Start date/time'], dayfirst=True)
data['Year'] = data['Start date/time'].dt.year
data['Month'] = data['Start date/time'].dt.month
data['Day'] = data['Start date/time'].dt.day
data['Hour'] = data['Start date/time'].dt.hour

# Create lagged features for target variable
data['Lag_Price'] = data['Price Germany/Luxembourg [Euro/MWh]'].shift(1)

# Drop rows with missing values after lagging
data = data.dropna()

# Feature selection based on correlation and engineering
features = [
    'temperature_2m (°C)', 
    'wind_speed_100m (km/h)', 
    'Total (grid consumption) [MWh]', 
    'Day', 
    'Hour', 
    'Lag_Price'
]
target = 'Price Germany/Luxembourg [Euro/MWh]'

# Subset the data
data_subset = data[features + [target]]

# Split data chronologically for train-test
train_data = data[data['Start date/time'] < '2023-09-30']
test_data = data[data['Start date/time'] >= '2023-09-30']

X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Normalize the target
y_train_mean, y_train_std = y_train.mean(), y_train.std()
y_train_scaled = (y_train - y_train_mean) / y_train_std

# Step 1: Hyperparameter tuning for Random Forest
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_random = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_distributions=rf_param_grid,
    n_iter=20,
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

rf_random.fit(X_train, y_train)
best_rf = rf_random.best_estimator_
print(f"Best RF Parameters: {rf_random.best_params_}")

# Predict with the best RF model
rf_train_pred = best_rf.predict(X_train)
rf_test_pred = best_rf.predict(X_test)

# Calculate residuals for Random Forest predictions
rf_train_residuals = y_train - rf_train_pred
rf_test_residuals = y_test - rf_test_pred

# Reshape residuals to 3D for LSTM
rf_train_residuals_reshaped = rf_train_residuals.values.reshape(-1, 1, 1)
rf_test_residuals_reshaped = rf_test_residuals.values.reshape(-1, 1, 1)

# Prepare input data for LSTM by appending residuals
X_train_lstm = np.concatenate([X_train_scaled.reshape((-1, 1, X_train_scaled.shape[1])), rf_train_residuals_reshaped], axis=2)
X_test_lstm = np.concatenate([X_test_scaled.reshape((-1, 1, X_test_scaled.shape[1])), rf_test_residuals_reshaped], axis=2)

# Adjust the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(units=64, activation='relu', return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
lstm_model.add(Dropout(0.2))  # Add dropout for regularization
lstm_model.add(LSTM(units=32, activation='relu'))
lstm_model.add(Dropout(0.2))  # Add dropout for regularization
lstm_model.add(Dense(units=1))

# Compile the model
lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Add early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the LSTM model on residuals
history = lstm_model.fit(
    X_train_lstm, 
    rf_train_residuals, 
    epochs=50,  # Increase epochs since early stopping will handle overfitting
    batch_size=32, 
    verbose=2, 
    validation_split=0.2, 
    callbacks=[early_stopping]
)

# Predict residuals with LSTM
lstm_residual_predictions = lstm_model.predict(X_test_lstm)

# Combine RF predictions with LSTM residuals for final prediction
final_hybrid_predictions = rf_test_pred + lstm_residual_predictions.flatten()

# Evaluate the hybrid model
mse = mean_squared_error(y_test, final_hybrid_predictions)
mae = mean_absolute_error(y_test, final_hybrid_predictions)
r2 = r2_score(y_test, final_hybrid_predictions)

print("Hybrid Model Evaluation:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R-squared: {r2:.4f}")

# Visualize performance
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="True Values", color='blue')
plt.plot(final_hybrid_predictions, label="Hybrid Predictions", color='orange')
plt.legend()
plt.title("Hybrid Model Predictions vs True Values")
plt.xlabel("Test Samples")
plt.ylabel("Price Germany/Luxembourg [Euro/MWh]")
plt.show()

# Residual analysis
residuals = y_test - final_hybrid_predictions
plt.figure(figsize=(10, 6))
plt.scatter(final_hybrid_predictions, residuals, color='blue')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.title('Residual Plot for Hybrid Model')
plt.show()

# Feature importance from Random Forest
importances = best_rf.feature_importances_
feature_names = features
plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances, color='skyblue')
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.show()

# Save Random Forest model
import joblib
joblib.dump(best_rf, 'random_forest_model.pkl')

# Save LSTM model
lstm_model.save('lstm_model.h5')

# Save the scaler (for feature normalization)
joblib.dump(scaler, 'scaler.pkl')