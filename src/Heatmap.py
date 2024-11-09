import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load and preprocess data
data = pd.read_csv("data/merged-data.csv") 

data['Price Germany/Luxembourg [Euro/MWh]'] = data['Price Germany/Luxembourg [Euro/MWh]'].replace({',': ''}, regex=True).astype(float)
data['Total (grid consumption) [MWh]'] = data['Total (grid consumption) [MWh]'].replace({',': ''}, regex=True).astype(float)

data['Start date/time'] = pd.to_datetime(data['Start date/time'], dayfirst=True)
data['Year'] = data['Start date/time'].dt.year
data['Month'] = data['Start date/time'].dt.month
data['Day'] = data['Start date/time'].dt.day
data['Hour'] = data['Start date/time'].dt.hour

# Encode 'Day of the Week' column
label_encoder = LabelEncoder()
data['Day of the Week'] = label_encoder.fit_transform(data['Start date/time'].dt.day_name())

target = 'Price Germany/Luxembourg [Euro/MWh]'
features = [
    'temperature_2m (°C)', 'relative_humidity_2m (%)', 'precipitation (mm)',
    'rain (mm)', 'snowfall (cm)', 'weather_code (wmo code)', 'wind_speed_100m (km/h)',
    'Total (grid consumption) [MWh]', 'Day of the Week', 'Year', 'Month', 'Day', 'Hour'
]

data_subset = data[features + [target]]

# Calculate correlation matrix
corr_matrix = data_subset.corr(method='pearson')

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()
