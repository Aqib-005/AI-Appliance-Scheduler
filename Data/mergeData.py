import pandas as pd

# Load data files
weather_data = pd.read_csv('weather-data.csv')
price_data = pd.read_csv('price-data.csv')
consumption_data = pd.read_csv('consumption-data.csv')

# Remove unnecessary columns
price_data = price_data.drop(columns=['End date/time'])
consumption_data = consumption_data.drop(columns=['End date/time'])

# Convert datetime columns to datetime format
weather_data['time'] = pd.to_datetime(weather_data['time'])
price_data['Start date/time'] = pd.to_datetime(price_data['Start date/time'], format='%b %d, %Y %I:%M %p')
consumption_data['Start date/time'] = pd.to_datetime(consumption_data['Start date/time'], format='%b %d, %Y %I:%M %p')

# Add day of the week 
consumption_data['Day of the Week'] = consumption_data['Start date/time'].dt.day_name()

# Set date range
start_date = '2022-09-30 00:00'
end_date = '2024-09-30 23:00'

# Filter datasets by the date range
weather_data = weather_data[(weather_data['time'] >= start_date) & (weather_data['time'] <= end_date)]
price_data = price_data[(price_data['Start date/time'] >= start_date) & (price_data['Start date/time'] <= end_date)]
consumption_data = consumption_data[(consumption_data['Start date/time'] >= start_date) & (consumption_data['Start date/time'] <= end_date)]

# Remove any duplicates
price_data = price_data.drop_duplicates(subset='Start date/time')
consumption_data = consumption_data.drop_duplicates(subset='Start date/time')

# Set index to time for all datasets
weather_data.set_index('time', inplace=True)
price_data.set_index('Start date/time', inplace=True)
consumption_data.set_index('Start date/time', inplace=True)

# Create a complete date range and reindex all datasets to ensure continuity
date_range = pd.date_range(start=start_date, end=end_date, freq='H')
weather_data = weather_data.reindex(date_range)
price_data = price_data.reindex(date_range)
consumption_data = consumption_data.reindex(date_range)

# Reset index and rename for merging
weather_data = weather_data.reset_index().rename(columns={'index': 'Start date/time'})
price_data = price_data.reset_index().rename(columns={'index': 'Start date/time'})
consumption_data = consumption_data.reset_index().rename(columns={'index': 'Start date/time'})

# Merge datasets on time
merged_data = pd.merge(weather_data, price_data, on='Start date/time', how='outer')
merged_data = pd.merge(merged_data, consumption_data, on='Start date/time', how='outer')

# Save the merged data to a CSV file
merged_data.to_csv('merged-data.csv', index=False)
print("Merged data has been saved to merged-data.csv")

