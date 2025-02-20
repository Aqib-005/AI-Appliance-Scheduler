import pandas as pd

def clean_and_find_datetime(df, possible_names, default_name):
    """
    Cleans column names and finds the datetime column from a list of possible names.
    If found, renames it to the default name.
    """
    # Clean column names by stripping whitespace and normalizing
    df.columns = df.columns.str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)
    
    # Normalize possible names and default name
    possible_names = [name.strip().lower() for name in possible_names]
    default_name = default_name.strip().lower()
    
    # Check if default column exists
    if default_name in df.columns:
        return df
    
    # Look for possible alternative columns
    for col in possible_names:
        if col in df.columns:
            df = df.rename(columns={col: default_name})
            return df
    
    # If no column found, raise an error with available columns
    raise KeyError(f"Datetime column '{default_name}' not found. Existing columns: {list(df.columns)}")

# Load data files
weather_data = pd.read_csv('weather-data.csv', encoding='latin-1')
price_data = pd.read_csv('price-data.csv', encoding='latin-1')
consumption_data = pd.read_csv('consumption-data.csv', encoding='latin-1')
generation_data = pd.read_csv('generation-data.csv', encoding='latin-1')

# Define datetime column names to search for each dataset
datetime_columns = {
    'price_data': ['start date', 'start date/time', 'datetime', 'date/time'],
    'consumption_data': ['start date', 'start date/time', 'datetime', 'date/time'],
    'generation_data': ['start date', 'start date/time', 'datetime', 'date/time']
}

# Process datetime columns
try:
    price_data = clean_and_find_datetime(price_data, datetime_columns['price_data'], 'start date/time')
    consumption_data = clean_and_find_datetime(consumption_data, datetime_columns['consumption_data'], 'start date/time')
    generation_data = clean_and_find_datetime(generation_data, datetime_columns['generation_data'], 'start date/time')
except KeyError as e:
    print(f"Error: {e}")
    exit(1)

# Remove unnecessary columns (keep only relevant columns)
columns_to_keep = {
    'price_data': ['start date/time', 'germany/luxembourg [?/mwh]'],
    'consumption_data': ['start date/time', 'total (grid load) [mwh]'],
    'generation_data': ['start date/time', 'total']
}

price_data = price_data[columns_to_keep['price_data']]
consumption_data = consumption_data[columns_to_keep['consumption_data']]
generation_data = generation_data[columns_to_keep['generation_data']]

# Process datetime for all datasets
def process_datetime(df):
    """
    Converts the 'start date/time' column to datetime format.
    """
    if 'start date/time' not in df.columns:
        raise KeyError(f"Column 'start date/time' not found in dataset. Available columns: {list(df.columns)}")
    
    df['start date/time'] = pd.to_datetime(
        df['start date/time'],
        format='%b %d, %Y %I:%M %p',
        errors='coerce'
    )
    return df

weather_data['time'] = pd.to_datetime(weather_data['time'])
price_data = process_datetime(price_data)
consumption_data = process_datetime(consumption_data)
generation_data = process_datetime(generation_data)

# Set updated date range
start_date = '2023-01-01 00:00'
end_date = '2025-01-01 23:00'

# Filter datasets by the new date range
def filter_by_date(df, date_col):
    """
    Filters the dataset to include only rows within the specified date range.
    """
    if date_col not in df.columns:
        raise KeyError(f"Column '{date_col}' not found in dataset. Available columns: {list(df.columns)}")
    
    return df[
        (df[date_col] >= pd.to_datetime(start_date)) & 
        (df[date_col] <= pd.to_datetime(end_date))
    ]

try:
    weather_data = filter_by_date(weather_data, 'time')
    price_data = filter_by_date(price_data, 'start date/time')
    consumption_data = filter_by_date(consumption_data, 'start date/time')
    generation_data = filter_by_date(generation_data, 'start date/time')
except KeyError as e:
    print(f"Error: {e}")
    exit(1)

# Remove duplicates
for df in [price_data, consumption_data, generation_data]:
    if 'start date/time' in df.columns:
        df.drop_duplicates(subset='start date/time', inplace=True, ignore_index=True)

# Set index to datetime
weather_data.set_index('time', inplace=True)
price_data.set_index('start date/time', inplace=True)
consumption_data.set_index('start date/time', inplace=True)
generation_data.set_index('start date/time', inplace=True)

# Create complete hourly date range
date_range = pd.date_range(start=start_date, end=end_date, freq='h')  # Use 'h' instead of 'H'

# Reindex all datasets
weather_data = weather_data.reindex(date_range)
price_data = price_data.reindex(date_range)
consumption_data = consumption_data.reindex(date_range)
generation_data = generation_data.reindex(date_range)

# Prepare for merging
dfs_to_merge = [
    weather_data.reset_index().rename(columns={'index': 'start date/time'}),
    price_data.reset_index().rename(columns={'index': 'start date/time'}),
    consumption_data.reset_index().rename(columns={'index': 'start date/time'}),
    generation_data.reset_index().rename(columns={'index': 'start date/time'})
]

# Merge datasets
merged_data = dfs_to_merge[0]
for df in dfs_to_merge[1:]:
    if 'start date/time' not in df.columns:
        print(f"Error: 'start date/time' column missing in dataset. Available columns: {list(df.columns)}")
        exit(1)
    
    merged_data = pd.merge(
        merged_data, 
        df, 
        on='start date/time', 
        how='outer',
        suffixes=('', '_DROP')
    )
    # Remove columns with _DROP suffix
    merged_data = merged_data[merged_data.columns[~merged_data.columns.str.endswith('_DROP')]]

# Add new features: Year, Month, Day, Hour
merged_data['Year'] = merged_data['start date/time'].dt.year
merged_data['Month'] = merged_data['start date/time'].dt.month
merged_data['Day'] = merged_data['start date/time'].dt.day
merged_data['Hour'] = merged_data['start date/time'].dt.hour

# Add day of week
merged_data['day of the week'] = merged_data['start date/time'].dt.day_name()

# Cleanup and forward fill
merged_data.sort_values('start date/time', inplace=True)
merged_data.ffill(inplace=True)

# Save merged data
merged_data.to_csv('merged-data.csv', index=False)
print("Merged data saved to merged-data.csv")