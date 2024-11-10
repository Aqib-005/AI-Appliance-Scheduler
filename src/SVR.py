import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv("data/merged-data.csv")

# Data cleaning and preprocessing
data['Price Germany/Luxembourg [Euro/MWh]'] = data['Price Germany/Luxembourg [Euro/MWh]'].replace({',': ''}, regex=True).astype(float)
data['Total (grid consumption) [MWh]'] = data['Total (grid consumption) [MWh]'].replace({',': ''}, regex=True).astype(float)

# Convert Start date/time to datetime and extract Year, Month, Day, Hour
data['Start date/time'] = pd.to_datetime(data['Start date/time'], dayfirst=True)
data['Year'] = data['Start date/time'].dt.year
data['Month'] = data['Start date/time'].dt.month
data['Day'] = data['Start date/time'].dt.day
data['Hour'] = data['Start date/time'].dt.hour

# Feature selection based on correlation analysis
features = [
    'temperature_2m (°C)', 'relative_humidity_2m (%)', 'wind_speed_100m (km/h)', 
    'Total (grid consumption) [MWh]', 'Month', 'Hour'
]
target = 'Price Germany/Luxembourg [Euro/MWh]'

# Subset the data to include only the features and target
data_subset = data[features + [target]]

# Remove rows with missing values
data_subset = data_subset.dropna()

# Split data into features (X) and target (y)
X = data_subset[features]
y = data_subset[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svr = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
svr.fit(X_train_scaled, y_train)

y_pred = svr.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation results
print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R-squared: {r2:.4f}")

# Plot the true vs predicted prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2, label='Ideal Prediction')
plt.title('SVR: Predicted vs Actual Prices')
plt.xlabel('True Price')
plt.ylabel('Predicted Price')
plt.legend()
plt.show()
