import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

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

# Train Random Forest model on the entire training set
X_train = train_data[features]
y_train = train_data[target]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

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
rf_random.fit(X_train_scaled, y_train)
best_rf = rf_random.best_estimator_

# Generate future timestamps for the desired prediction range
future_dates = pd.date_range(start="2024-12-01 00:00", end="2024-12-31 23:00", freq="H")  # 1 month
future_df = pd.DataFrame({"Start date/time": future_dates})
future_df['Day'] = future_df['Start date/time'].dt.day
future_df['Hour'] = future_df['Start date/time'].dt.hour
future_df['Month'] = future_df['Start date/time'].dt.month

# Placeholder for predictions
future_predictions = []
rolling_window = 24  # Predict 1 day at a time

# Iteratively forecast day-ahead prices
for i in range(0, len(future_df), rolling_window):
    future_window = future_df.iloc[i:i + rolling_window].copy()

    # Generate lagged price feature for the current window
    if not future_predictions:
        lag_price = train_data.iloc[-1][target]  # Last known price
    else:
        lag_price = future_predictions[-1]

    future_window['Lag_Price'] = lag_price
    future_window['temperature_2m (°C)'] = np.random.normal(loc=15, scale=5, size=len(future_window))  # Placeholder
    future_window['wind_speed_100m (km/h)'] = np.random.normal(loc=10, scale=2, size=len(future_window))  # Placeholder
    future_window['Total (grid consumption) [MWh]'] = np.random.normal(loc=50000, scale=1000, size=len(future_window))  # Placeholder

    # Prepare features
    X_future = scaler.transform(future_window[features])
    future_pred = best_rf.predict(X_future)
    future_predictions.extend(future_pred)

# Save future predictions
future_df['Predicted Price'] = future_predictions
future_df.to_csv("future_predictions_2024.csv", index=False)

# Plot the future predictions
plt.figure(figsize=(12, 6))
plt.plot(future_df['Start date/time'], future_df['Predicted Price'], label="Future Predictions", color='orange')
plt.title("Electricity Price Predictions for December 2024")
plt.xlabel("Timestamp")
plt.ylabel("Price (Euro/MWh)")
plt.legend()
plt.grid()
plt.show()
