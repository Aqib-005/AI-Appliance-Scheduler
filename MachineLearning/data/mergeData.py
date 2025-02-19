import pandas as pd

# Load data files
weather_data = pd.read_csv('weather-data.csv')
price_data = pd.read_csv('price-data.csv')
consumption_data = pd.read_csv('consumption-data.csv')
generation_data = pd.read_csv('generation-data.csv')  # New dataset

# Remove unnecessary columns from all datasets
columns_to_drop = ['End date/time', 'Unnamed: 0', 'unnamed']  # Add any extra columns to remove
price_data = price_data.drop(columns=columns_to_drop, errors='ignore')
consumption_data = consumption_data.drop(columns=columns_to_drop, errors='ignore')
generation_data = generation_data.drop(columns=columns_to_drop, errors='ignore')

# Convert datetime columns to datetime format
def process_datetime(df, col_name):
    df[col_name] = pd.to_datetime(
        df[col_name], 
        format='%b %d, %Y %I:%M %p',
        errors='coerce'
    )
    return df

# Process datetime for all datasets
weather_data['time'] = pd.to_datetime(weather_data['time'])
price_data = process_datetime(price_data, 'Start date/time')
consumption_data = process_datetime(consumption_data, 'Start date/time')
generation_data = process_datetime(generation_data, 'Start date/time')

# Set updated date range
start_date = '2023-01-01 00:00'
end_date = '2025-01-01 23:00'

# Filter datasets by the new date range
def filter_by_date(df, date_col):
    return df[
        (df[date_col] >= start_date) & 
        (df[date_col] <= end_date)
    ]

weather_data = filter_by_date(weather_data, 'time')
price_data = filter_by_date(price_data, 'Start date/time')
consumption_data = filter_by_date(consumption_data, 'Start date/time')
generation_data = filter_by_date(generation_data, 'Start date/time')

# Remove duplicates in all datasets
for df in [price_data, consumption_data, generation_data]:
    df.drop_duplicates(subset='Start date/time', inplace=True, ignore_index=True)

# Set index to time for all datasets
weather_data.set_index('time', inplace=True)
price_data.set_index('Start date/time', inplace=True)
consumption_data.set_index('Start date/time', inplace=True)
generation_data.set_index('Start date/time', inplace=True)

# Create complete hourly date range
date_range = pd.date_range(start=start_date, end=end_date, freq='H')

# Reindex all datasets to ensure continuity
weather_data = weather_data.reindex(date_range)
price_data = price_data.reindex(date_range)
consumption_data = consumption_data.reindex(date_range)
generation_data = generation_data.reindex(date_range)

# Prepare for merging
dfs_to_merge = [
    weather_data.reset_index().rename(columns={'index': 'Start date/time'}),
    price_data.reset_index(),
    consumption_data.reset_index(),
    generation_data.reset_index()
]

# Merge all datasets sequentially
merged_data = dfs_to_merge[0]
for df in dfs_to_merge[1:]:
    merged_data = pd.merge(
        merged_data, 
        df, 
        on='Start date/time', 
        how='outer',
        suffixes=('', '_DROP')
    )
    # Remove any duplicate columns
    merged_data = merged_data[[c for c in merged_data.columns if '_DROP' not in c]]

# Add day of week feature
merged_data['Day of the Week'] = merged_data['Start date/time'].dt.day_name()

# Final cleanup: Sort and forward fill missing weather data
merged_data.sort_values('Start date/time', inplace=True)
merged_data.ffill(inplace=True)  # Handle missing values for weather data

# Save merged data
merged_data.to_csv('merged-data.csv', index=False)
print("Merged data saved to merged-data-2023-2025.csv")