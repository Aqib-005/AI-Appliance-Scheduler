import pandas as pd
import numpy as np


def clean_and_find_datetime(df, possible_names, default_name):
    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)
    df.columns = df.columns.str.replace('Ã¸', 'ø')
    possible_names = [name.strip().lower() for name in possible_names]
    default_name = default_name.strip().lower()
    for col in possible_names:
        if col in df.columns:
            df = df.rename(columns={col: default_name})
            return df
    raise KeyError(f"Datetime column '{default_name}' not found. Existing columns: {list(df.columns)}")

def process_datetime(df, datetime_col, fmt=None):
    if datetime_col not in df.columns:
        raise KeyError(f"Column '{datetime_col}' not found. Available columns: {list(df.columns)}")
    if fmt:
        df[datetime_col] = pd.to_datetime(df[datetime_col], format=fmt, errors='coerce')
    else:
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
    return df

def filter_by_date(df, date_col, start_date, end_date):
    if date_col not in df.columns:
        raise KeyError(f"Column '{date_col}' not found. Available columns: {list(df.columns)}")
    return df[(df[date_col] >= pd.to_datetime(start_date)) & (df[date_col] <= pd.to_datetime(end_date))]

def normalize_weather_columns(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)
    df.columns = df.columns.str.replace('Ã¸', 'ø')   
    return df

# Define historical date range
hist_start = '2022-02-09 00:00'
hist_end   = '2025-02-09 23:00'

price_data = pd.read_csv('price-data.csv', encoding='latin-1')
price_data = clean_and_find_datetime(price_data, 
                                       possible_names=['start date', 'start date/time', 'datetime', 'date/time'], 
                                       default_name='start date/time')
if 'germany/luxembourg [?/mwh]' not in price_data.columns:
    raise KeyError("Expected column 'germany/luxembourg [?/mwh]' not found in price-data.csv")
price_data = price_data[['start date/time', 'germany/luxembourg [?/mwh]']]

weather_data = pd.read_csv('weather-data.csv', encoding='latin-1')
weather_data = normalize_weather_columns(weather_data)
weather_data = weather_data.rename(columns={'time': 'start date/time'})

desired_weather_cols = [
    'start date/time', 
    'temperature_2m (øc)',   
    'relative_humidity_2m (%)', 
    'precipitation (mm)', 
    'rain (mm)', 
    'snowfall (cm)', 
    'weather_code (wmo code)', 
    'wind_speed_100m (km/h)'
]

existing_weather_cols = [col for col in desired_weather_cols if col in weather_data.columns]
if len(existing_weather_cols) < len(desired_weather_cols):
    missing = set(desired_weather_cols) - set(existing_weather_cols)
    print(f"Warning: The following weather columns are missing and will be skipped: {missing}")
weather_data = weather_data[existing_weather_cols]

consumption_data = pd.read_csv('consumption-data.csv', encoding='latin-1')
consumption_data = clean_and_find_datetime(consumption_data, 
                                             possible_names=['start date', 'start date/time', 'datetime', 'date/time'], 
                                             default_name='start date/time')
if 'grid load [mwh]' not in consumption_data.columns:
    raise KeyError("Expected column 'grid load [mwh]' not found in consumption-data.csv")
consumption_data = consumption_data[['start date/time', 'grid load [mwh]']]

generation_data = pd.read_csv('generation-data.csv', encoding='latin-1')
generation_data = clean_and_find_datetime(generation_data, 
                                            possible_names=['start date', 'start date/time', 'datetime', 'date/time'], 
                                            default_name='start date/time')
if 'total_generation' not in generation_data.columns:
    raise KeyError("Expected column 'total_generation' not found in generation-data.csv")
generation_data = generation_data[['start date/time', 'total_generation']]

price_data = process_datetime(price_data, 'start date/time')
weather_data = process_datetime(weather_data, 'start date/time')
consumption_data = process_datetime(consumption_data, 'start date/time')
generation_data = process_datetime(generation_data, 'start date/time')

price_data = filter_by_date(price_data, 'start date/time', hist_start, hist_end)
weather_data = filter_by_date(weather_data, 'start date/time', hist_start, hist_end)
consumption_data = filter_by_date(consumption_data, 'start date/time', hist_start, hist_end)
generation_data = filter_by_date(generation_data, 'start date/time', hist_start, hist_end)

for df in [price_data, consumption_data, generation_data]:
    df.drop_duplicates(subset='start date/time', inplace=True, ignore_index=True)

price_data.set_index('start date/time', inplace=True)
weather_data.set_index('start date/time', inplace=True)
consumption_data.set_index('start date/time', inplace=True)
generation_data.set_index('start date/time', inplace=True)

hist_date_range = pd.date_range(start=hist_start, end=hist_end, freq='h')
price_data = price_data.reindex(hist_date_range)
weather_data = weather_data.reindex(hist_date_range)
consumption_data = consumption_data.reindex(hist_date_range)
generation_data = generation_data.reindex(hist_date_range)

price_data = price_data.reset_index().rename(columns={'index': 'start date/time'})
weather_data = weather_data.reset_index().rename(columns={'index': 'start date/time'})
consumption_data = consumption_data.reset_index().rename(columns={'index': 'start date/time'})
generation_data = generation_data.reset_index().rename(columns={'index': 'start date/time'})

dfs_hist = [weather_data, price_data, consumption_data, generation_data]
merged_data = dfs_hist[0]
for df in dfs_hist[1:]:
    merged_data = pd.merge(merged_data, df, on='start date/time', how='outer', suffixes=('', '_DROP'))
    merged_data = merged_data[merged_data.columns[~merged_data.columns.str.endswith('_DROP')]]

merged_data['Year'] = pd.to_datetime(merged_data['start date/time']).dt.year
merged_data['Month'] = pd.to_datetime(merged_data['start date/time']).dt.month
merged_data['Day'] = pd.to_datetime(merged_data['start date/time']).dt.day
merged_data['Hour'] = pd.to_datetime(merged_data['start date/time']).dt.hour
merged_data['day of the week'] = pd.to_datetime(merged_data['start date/time']).dt.day_name()

hist_final_cols = [
    'start date/time',
    'temperature_2m (øc)',
    'relative_humidity_2m (%)',
    'precipitation (mm)',
    'rain (mm)',
    'snowfall (cm)',
    'weather_code (wmo code)',
    'wind_speed_100m (km/h)',
    'germany/luxembourg [?/mwh]',
    'grid load [mwh]',
    'total_generation',
    'Year',
    'Month',
    'Day',
    'Hour',
    'day of the week'
]
merged_data = merged_data[hist_final_cols]

hist_rename_mapping = {
    'start date/time': 'Start Date/Time',
    'temperature_2m (øc)': 'Temperature (°C)',
    'relative_humidity_2m (%)': 'Relative Humidity (%)',
    'precipitation (mm)': 'Precipitation (mm)',
    'rain (mm)': 'Rain (mm)',
    'snowfall (cm)': 'Snowfall (cm)',
    'weather_code (wmo code)': 'Weather Code',
    'wind_speed_100m (km/h)': 'Wind Speed (km/h)',
    'germany/luxembourg [?/mwh]': 'Price (€/MWh)',
    'grid load [mwh]': 'Grid Load (MWh)',
    'total_generation': 'Total Generation (MWh)',
    'day of the week': 'Day of Week'
}
merged_data.rename(columns=hist_rename_mapping, inplace=True)

merged_data.sort_values('Start Date/Time', inplace=True)
merged_data.ffill(inplace=True)
merged_data.to_csv('merged-data.csv', index=False)
print("Historical merged data saved to merged-data.csv")

# future date range
future_start = '2025-02-10 00:00'
future_end   = '2025-03-10 23:00'

future_weather = pd.read_csv('future-weather.csv', encoding='latin-1')
future_weather = normalize_weather_columns(future_weather)
future_weather = future_weather.rename(columns={'time': 'start date/time'})
future_weather_cols = [
    'start date/time', 
    'temperature_2m (øc)', 
    'relative_humidity_2m (%)', 
    'precipitation (mm)', 
    'rain (mm)', 
    'snowfall (cm)', 
    'weather_code (wmo code)', 
    'wind_speed_100m (km/h)'
]
existing_future_weather_cols = [col for col in future_weather_cols if col in future_weather.columns]
if len(existing_future_weather_cols) < len(future_weather_cols):
    missing = set(future_weather_cols) - set(existing_future_weather_cols)
    print(f"Warning: Future weather missing columns: {missing}")
future_weather = future_weather[existing_future_weather_cols]

future_consumption = pd.read_csv('future-consumption.csv', encoding='latin-1')
future_consumption = clean_and_find_datetime(future_consumption, 
                                               possible_names=['start date', 'start date/time', 'datetime', 'date/time'], 
                                               default_name='start date/time')
if 'grid load [mwh]' not in future_consumption.columns:
    raise KeyError("Expected column 'grid load [mwh]' not found in future-consumption.csv")
future_consumption = future_consumption[['start date/time', 'grid load [mwh]']]

future_generation = pd.read_csv('future-generation.csv', encoding='latin-1')
future_generation = clean_and_find_datetime(future_generation, 
                                              possible_names=['start date', 'start date/time', 'datetime', 'date/time'], 
                                              default_name='start date/time')
future_generation.columns = future_generation.columns.str.lower().str.replace('-', '_')
if 'total_generation' not in future_generation.columns:
    if 'total generation' in future_generation.columns:
        future_generation = future_generation.rename(columns={'total generation': 'total_generation'})
    else:
        raise KeyError("Expected column 'total_generation' not found in future-generation.csv")
future_generation = future_generation[['start date/time', 'total_generation']]

future_weather = process_datetime(future_weather, 'start date/time')
future_consumption = process_datetime(future_consumption, 'start date/time')
future_generation = process_datetime(future_generation, 'start date/time')

future_weather = filter_by_date(future_weather, 'start date/time', future_start, future_end)
future_consumption = filter_by_date(future_consumption, 'start date/time', future_start, future_end)
future_generation = filter_by_date(future_generation, 'start date/time', future_start, future_end)

for df in [future_consumption, future_generation]:
    df.drop_duplicates(subset='start date/time', inplace=True, ignore_index=True)

future_weather.set_index('start date/time', inplace=True)
future_consumption.set_index('start date/time', inplace=True)
future_generation.set_index('start date/time', inplace=True)

future_date_range = pd.date_range(start=future_start, end=future_end, freq='h')
future_weather = future_weather.reindex(future_date_range)
future_consumption = future_consumption.reindex(future_date_range)
future_generation = future_generation.reindex(future_date_range)

future_weather = future_weather.reset_index().rename(columns={'index': 'start date/time'})
future_consumption = future_consumption.reset_index().rename(columns={'index': 'start date/time'})
future_generation = future_generation.reset_index().rename(columns={'index': 'start date/time'})

dfs_future = [future_weather, future_consumption, future_generation]
merged_future = dfs_future[0]
for df in dfs_future[1:]:
    merged_future = pd.merge(merged_future, df, on='start date/time', how='outer', suffixes=('', '_DROP'))
    merged_future = merged_future[merged_future.columns[~merged_future.columns.str.endswith('_DROP')]]

merged_future['Year'] = pd.to_datetime(merged_future['start date/time']).dt.year
merged_future['Month'] = pd.to_datetime(merged_future['start date/time']).dt.month
merged_future['Day'] = pd.to_datetime(merged_future['start date/time']).dt.day
merged_future['Hour'] = pd.to_datetime(merged_future['start date/time']).dt.hour
merged_future['day of the week'] = pd.to_datetime(merged_future['start date/time']).dt.day_name()

future_final_cols = [
    'start date/time',
    'temperature_2m (øc)',
    'relative_humidity_2m (%)',
    'precipitation (mm)',
    'rain (mm)',
    'snowfall (cm)',
    'weather_code (wmo code)',
    'wind_speed_100m (km/h)',
    'grid load [mwh]',
    'total_generation',
    'Year',
    'Month',
    'Day',
    'Hour',
    'day of the week'
]
merged_future = merged_future[future_final_cols]

future_rename_mapping = {
    'start date/time': 'Start Date/Time',
    'temperature_2m (øc)': 'Temperature (°C)',
    'relative_humidity_2m (%)': 'Relative Humidity (%)',
    'precipitation (mm)': 'Precipitation (mm)',
    'rain (mm)': 'Rain (mm)',
    'snowfall (cm)': 'Snowfall (cm)',
    'weather_code (wmo code)': 'Weather Code',
    'wind_speed_100m (km/h)': 'Wind Speed (km/h)',
    'grid load [mwh]': 'Grid Load (MWh)',
    'total_generation': 'Total Generation (MWh)',
    'day of the week': 'Day of Week'
}
merged_future.rename(columns=future_rename_mapping, inplace=True)

merged_future.sort_values('Start Date/Time', inplace=True)
merged_future.ffill(inplace=True)
merged_future.to_csv('future-data.csv', index=False)
print("Future merged data saved to future-data.csv")
