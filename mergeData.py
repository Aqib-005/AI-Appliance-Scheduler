import pandas as pd

# Load data files
weather_data = pd.read_csv('weather-data.csv')
price_data = pd.read_csv('price-data.csv')

#remove unwanted columns
price_data = price_data.drop(columns=['ISO3 Code', 'Datetime (UTC)'])

# Convert datetime columns
weather_data['time'] = pd.to_datetime(weather_data['time'])
price_data['Datetime (Local)'] = pd.to_datetime(price_data['Datetime (Local)'], format='%d/%m/%Y %H:%M', dayfirst=True)

# Assign range
start_date = '2022-09-30 00:00'
end_date = '2024-09-30 23:00'

weather_data = weather_data[(weather_data['time'] >= start_date) & (weather_data['time'] <= end_date)]
price_data = price_data[(price_data['Datetime (Local)'] >= start_date) & (price_data['Datetime (Local)'] <= end_date)]

# Remove duplicate timestamps from price_data
price_data = price_data.drop_duplicates(subset='Datetime (Local)')

# Set index to time for both
weather_data.set_index('time', inplace=True)
price_data.set_index('Datetime (Local)', inplace=True)

# Create a complete date range and reindex
date_range = pd.date_range(start=start_date, end=end_date, freq='H')
weather_data = weather_data.reindex(date_range)
price_data = price_data.reindex(date_range)

# Reset index and rename for merge
weather_data = weather_data.reset_index().rename(columns={'index': 'Datetime (Local)'})
price_data = price_data.reset_index().rename(columns={'index': 'Datetime (Local)'})

# Merge datasets on time
merged_data = pd.merge(weather_data, price_data, on='Datetime (Local)', how='outer')

# Save the merged data
merged_data.to_csv('merged-data.csv', index=False)
print("Merged data has been saved to merged_data.csv")


# merged_data = pd.merge(filtered_weather, filtered_price, left_on='time', right_on='Datetime (Local)')
# merged_data.drop(columns=['Datetime (Local)'], inplace=True)
# merged_data.to_csv('merged_data.csv', index=False)





