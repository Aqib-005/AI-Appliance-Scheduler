import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load and preprocess data
try:
    data = pd.read_csv("data/merged-data.csv", encoding='utf-8-sig')
except UnicodeDecodeError:
    try:
        data = pd.read_csv("data/merged-data.csv", encoding='latin-1')
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        exit(1)

# Print actual columns for debugging
print("Actual columns in DataFrame:", data.columns.tolist())

# Clean up column names (remove trailing spaces)
data.columns = data.columns.str.strip()

# List of columns that should be numeric
numeric_columns = [
    'temperature_2m',
    'precipitation (mm)',
    'rain (mm)',
    'snowfall (cm)',
    'weather_code (wmo code)',
    'wind_speed_100m (km/h)',
    'grid_load',
    'day_price',
    'Year',
    'Month',
    'Day',
    'Hour'
]

# Convert columns to numeric, coercing errors (e.g., "#NUM!" becomes NaN)
for col in numeric_columns:
    if col in data.columns:
        # Remove commas and convert to numeric
        data[col] = pd.to_numeric(data[col].astype(str).str.replace(',', ''), errors='coerce')
    else:
        print(f"Warning: Column {col} not found in DataFrame")

# Define required columns (make sure names match cleaned column names)
required_columns = [
    'temperature_2m',
    'precipitation (mm)',
    'rain (mm)',
    'snowfall (cm)',
    'weather_code (wmo code)',
    'wind_speed_100m (km/h)',
    'grid_load',
    'Year',
    'Month',
    'Day',
    'Hour',
    'day of the week',
    'day_price'
]

# Check for missing columns
missing = [col for col in required_columns if col not in data.columns]
if missing:
    print(f"Critical columns missing: {missing}")
    exit(1)

# Encode categorical feature 'day of the week'
label_encoder = LabelEncoder()
data['day of the week'] = label_encoder.fit_transform(data['day of the week'])

# Create analysis dataframe and rename columns for clarity
data_subset = data[required_columns].rename(columns={
    'temperature_2m': 'Temperature (°C)',
    'day_price': 'Price (€/MWh)',
    'grid_load': 'Consumption (MWh)',
    'weather_code (wmo code)': 'Weather Code',
    'wind_speed_100m (km/h)': 'Wind Speed (km/h)'
})

# Calculate correlation matrix
corr_matrix = data_subset.corr(method='pearson')

# Plot heatmap
plt.figure(figsize=(18, 16))
sns.heatmap(corr_matrix, 
            annot=True, 
            fmt='.2f', 
            cmap='coolwarm', 
            linewidths=0.5,
            cbar_kws={'label': 'Correlation Strength'})

plt.title("Electricity Price Correlation Matrix\n", fontsize=20, pad=25)
plt.xticks(rotation=55, ha='right', fontsize=14)
plt.yticks(fontsize=14) 
plt.tight_layout()
plt.savefig('price_correlation_heatmap_full.png', dpi=300, bbox_inches='tight')
plt.show()
