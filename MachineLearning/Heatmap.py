import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load and preprocess data with correct encoding
try:
    data = pd.read_csv("data/merged-data.csv", encoding='utf-8-sig')  # Changed encoding
except UnicodeDecodeError:
    data = pd.read_csv("data/merged-data.csv", encoding='latin-1')

# Clean column names by stripping whitespace and normalizing
data.columns = data.columns.str.strip().str.lower()

# Clean and convert numerical columns
numeric_columns = ['day-price', 'total-grid', 'total-consumption']
for col in numeric_columns:
    data[col] = data[col].astype(str).str.replace(',', '').astype(float)

# Convert datetime and extract features
data['start date/time'] = pd.to_datetime(data['start date/time'], dayfirst=True)
data['year'] = data['start date/time'].dt.year
data['month'] = data['start date/time'].dt.month
data['day'] = data['start date/time'].dt.day
data['hour'] = data['start date/time'].dt.hour

# Encode categorical features
label_encoder = LabelEncoder()
data['day of the week'] = label_encoder.fit_transform(data['day of the week'])

# Define target and features using cleaned column names
target = 'day-price'
features = [
    'temperature_2m (ã¸c)',  # Original column name from your data
    'precipitation (mm)',
    'rain (mm)',
    'snowfall (cm)',
    'weather_code (wmo code)',
    'wind_speed_100m (km/h)',
    'total-consumption',
    'total-grid',
    'year',
    'month',
    'day',
    'hour',
    'day of the week'
]

# Create analysis dataframe with friendly names
data_subset = data[features + [target]].rename(columns={
    'temperature_2m (ã¸c)': 'Temperature (°C)',
    'day-price': 'Price (€/MWh)',
    'total-consumption': 'Consumption (MWh)',
    'total-grid': 'Generation (MWh)',
    'weather_code (wmo code)': 'Weather Code',
    'wind_speed_100m (km/h)': 'Wind Speed'
})

# Calculate correlation matrix
corr_matrix = data_subset.corr(method='pearson')

# Plot the heatmap
plt.figure(figsize=(18, 16))
sns.heatmap(corr_matrix, 
            annot=True, 
            fmt='.2f', 
            cmap='coolwarm', 
            linewidths=0.5,
            mask=np.triu(np.ones_like(corr_matrix, dtype=bool)),
            cbar_kws={'label': 'Correlation Strength'})

plt.title("Electricity Price Correlation Matrix (2023-2025)\n", fontsize=20, pad=25)
plt.xticks(rotation=55, ha='right', fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()

# Save the visualization
plt.savefig('price_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()