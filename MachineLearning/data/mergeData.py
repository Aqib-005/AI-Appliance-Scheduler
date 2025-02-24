import pandas as pd
import numpy as np

# ------------------------------
# Helper functions
# ------------------------------
def clean_and_find_datetime(df, possible_names, default_name):
    """
    Cleans column names and finds the datetime column from a list of possible names.
    If found, renames it to the default name.
    """
    df.columns = df.columns.str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)
    possible_names = [name.strip().lower() for name in possible_names]
    default_name = default_name.strip().lower()
    if default_name in df.columns:
        return df
    for col in possible_names:
        if col in df.columns:
            df = df.rename(columns={col: default_name})
            return df
    raise KeyError(f"Datetime column '{default_name}' not found. Existing columns: {list(df.columns)}")

def process_datetime(df):
    """
    Converts the 'start date/time' column to datetime using a known format.
    """
    if 'start date/time' not in df.columns:
        raise KeyError(f"Column 'start date/time' not found. Available columns: {list(df.columns)}")
    df['start date/time'] = pd.to_datetime(df['start date/time'], format='%b %d, %Y %I:%M %p', errors='coerce')
    return df

def filter_by_date(df, date_col, start_date, end_date):
    """
    Filters the dataset to include only rows within the specified date range.
    """
    if date_col not in df.columns:
        raise KeyError(f"Column '{date_col}' not found. Available columns: {list(df.columns)}")
    return df[(df[date_col] >= pd.to_datetime(start_date)) & (df[date_col] <= pd.to_datetime(end_date))]

# ------------------------------
# Load Historical Data
# ------------------------------
weather_data = pd.read_csv('weather-data.csv', encoding='latin-1')
price_data = pd.read_csv('price-data.csv', encoding='latin-1')
consumption_data = pd.read_csv('consumption-data.csv', encoding='latin-1')
# generation_data will be dropped

# Process datetime columns in price and consumption data using our helper
datetime_cols = {
    'price_data': ['start date', 'start date/time', 'datetime', 'date/time'],
    'consumption_data': ['start date', 'start date/time', 'datetime', 'date/time']
}
try:
    price_data = clean_and_find_datetime(price_data, datetime_cols['price_data'], 'start date/time')
    consumption_data = clean_and_find_datetime(consumption_data, datetime_cols['consumption_data'], 'start date/time')
except KeyError as e:
    print(f"Error: {e}")
    exit(1)

# For weather, assume the datetime column is named "time"
weather_data['time'] = pd.to_datetime(weather_data['time'])
    
# Remove unnecessary columns
cols_to_keep = {
    'price_data': ['start date/time', 'germany/luxembourg [?/mwh]'],
    'consumption_data': ['start date/time', 'total (grid load) [mwh]']
}
price_data = price_data[cols_to_keep['price_data']]
consumption_data = consumption_data[cols_to_keep['consumption_data']]

# Process datetime for historical datasets
price_data = process_datetime(price_data)
consumption_data = process_datetime(consumption_data)
# Weather data: already converted to datetime in 'time'

# ------------------------------
# Set Historical Date Range
# ------------------------------
hist_start = '2023-01-01 00:00'
hist_end = '2025-01-01 23:00'
weather_data = weather_data[(weather_data['time'] >= pd.to_datetime(hist_start)) &
                              (weather_data['time'] <= pd.to_datetime(hist_end))]
price_data = filter_by_date(price_data, 'start date/time', hist_start, hist_end)
consumption_data = filter_by_date(consumption_data, 'start date/time', hist_start, hist_end)

# Remove duplicates based on datetime
for df in [price_data, consumption_data]:
    df.drop_duplicates(subset='start date/time', inplace=True, ignore_index=True)

# Set index to datetime for reindexing later
weather_data.set_index('time', inplace=True)
price_data.set_index('start date/time', inplace=True)
consumption_data.set_index('start date/time', inplace=True)

# ------------------------------
# Load Future Data
# ------------------------------
# Future Weather Data
future_weather = pd.read_csv('future_weather_data.csv', encoding='latin-1')
future_weather['time'] = pd.to_datetime(future_weather['time'])
future_weather.set_index('time', inplace=True)
# Future Consumption Data
future_consumption = pd.read_csv('Future_consumption.csv', encoding='latin-1')
# Rename columns: assume "Start date" is the timestamp, "Total (grid load) [MWh]" is consumption
future_consumption.rename(columns={"Start date": "start date/time",
                                   "Total (grid load) [MWh]": "total (grid load)"}, inplace=True)
# Convert using the format in the file (e.g., "Jan 1, 2023 12:00 AM")
future_consumption['start date/time'] = pd.to_datetime(future_consumption['start date/time'],
                                                       format='%b %d, %Y %I:%M %p', errors='coerce')
future_consumption.set_index('start date/time', inplace=True)
# Filter future consumption to only include dates from 2025-01-01 to, say, 2025-02-28 23:00
fut_cons_start = '2025-01-01 00:00'
fut_cons_end = '2025-02-28 23:00'
future_consumption = future_consumption.loc[fut_cons_start:fut_cons_end]

# ------------------------------
# Define Overall Date Range for Merging
# ------------------------------
# We'll merge historical and future data.
# For weather and consumption, use historical up to hist_end and then future for dates > hist_end.
overall_start = pd.to_datetime(hist_start)
overall_end = pd.to_datetime('2025-02-28 23:00')
date_range = pd.date_range(start=overall_start, end=overall_end, freq='h')

# ------------------------------
# Combine Historical and Future Data for Weather and Consumption
# ------------------------------
# For weather, concatenate historical and future, then remove duplicates giving priority to future data
weather_combined = pd.concat([weather_data, future_weather])
weather_combined = weather_combined[~weather_combined.index.duplicated(keep='last')]
weather_combined = weather_combined.reindex(date_range)

# For consumption, do the same
consumption_combined = pd.concat([consumption_data, future_consumption])
consumption_combined = consumption_combined[~consumption_combined.index.duplicated(keep='last')]
consumption_combined = consumption_combined.reindex(date_range)

# For price, we only have historical data. Future rows will be NaN.
price_combined = price_data.reindex(date_range)

# ------------------------------
# Prepare for Merging
# ------------------------------
# For merging, reset index and rename index to 'start date/time'
weather_combined = weather_combined.reset_index().rename(columns={'index': 'start date/time'})
price_combined = price_combined.reset_index().rename(columns={'index': 'start date/time'})
consumption_combined = consumption_combined.reset_index().rename(columns={'index': 'start date/time'})

# We no longer use generation_data, so skip it.
dfs_to_merge = [weather_combined, price_combined, consumption_combined]

# Merge datasets one-by-one using outer join.
merged_data = dfs_to_merge[0]
for df in dfs_to_merge[1:]:
    if 'start date/time' not in df.columns:
        print(f"Error: 'start date/time' column missing. Available columns: {list(df.columns)}")
        exit(1)
    merged_data = pd.merge(merged_data, df, on='start date/time', how='outer', suffixes=('', '_DROP'))
    merged_data = merged_data[merged_data.columns[~merged_data.columns.str.endswith('_DROP')]]

# ------------------------------
# Add Time-Based Features
# ------------------------------
merged_data['Year'] = pd.to_datetime(merged_data['start date/time']).dt.year
merged_data['Month'] = pd.to_datetime(merged_data['start date/time']).dt.month
merged_data['Day'] = pd.to_datetime(merged_data['start date/time']).dt.day
merged_data['Hour'] = pd.to_datetime(merged_data['start date/time']).dt.hour
merged_data['day of the week'] = pd.to_datetime(merged_data['start date/time']).dt.day_name()

# ------------------------------
# Cleanup and Forward Fill
# ------------------------------
merged_data.sort_values('start date/time', inplace=True)
merged_data.ffill(inplace=True)

# ------------------------------
# Save the Merged Data
# ------------------------------
merged_data.to_csv('merged-data.csv', index=False)
print("Merged data saved to merged-data.csv")
