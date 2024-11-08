import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
data = pd.read_csv("data/merged-data.csv") 

# Remove commas in numeric columns and convert them to float
data['Price Germany/Luxembourg [Euro/MWh]'] = data['Price Germany/Luxembourg [Euro/MWh]'].replace({',': ''}, regex=True).astype(float)
data['Total (grid consumption) [MWh]'] = data['Total (grid consumption) [MWh]'].replace({',': ''}, regex=True).astype(float)

# Specify dayfirst=True to indicate that the date format is DD/MM/YYYY
data['Start date/time'] = pd.to_datetime(data['Start date/time'], dayfirst=True)
data['Year'] = data['Start date/time'].dt.year
data['Month'] = data['Start date/time'].dt.month
data['Day'] = data['Start date/time'].dt.day
data['Hour'] = data['Start date/time'].dt.hour

# Define features and target variable
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

# Define numerical and categorical columns for preprocessing
numeric_features = [
    'temperature_2m (°C)', 'relative_humidity_2m (%)', 'precipitation (mm)',
    'rain (mm)', 'snowfall (cm)', 'wind_speed_100m (km/h)', 'Total (grid consumption) [MWh]',
    'weather_code (wmo code)', 'Year', 'Month', 'Day', 'Hour'
]
categorical_features = ['Day of the Week']

# Preprocessing: scaling numeric data and encoding categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

# Pipeline with preprocessing and Random Forest regressor
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Model Evaluation:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
