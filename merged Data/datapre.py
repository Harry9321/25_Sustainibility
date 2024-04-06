import pandas as pd

# Assuming 'df' is your DataFrame with hourly granular data and 'timestamp_column' is the column containing timestamps
# Convert the timestamp column to datetime if it's not already in datetime format
df['DateTime'] = pd.to_datetime(df['DateTime'])

# Set the timestamp column as the index
df.set_index('DateTime', inplace=True)

# Resample the data to daily frequency and aggregate using sum, mean, or any other appropriate method
# Here we use mean aggregation as an example
df_daily = df.resample('D').mean()  # 'D' represents daily frequency

# Reset index to have 'timestamp_column' as a regular column again
df_daily.reset_index(inplace=True)

# Optionally, you can rename the timestamp_column to something more descriptive like 'date' if needed
# df_daily.rename(columns={'timestamp_column': 'date'}, inplace=True)

# Display the daily data
print(df_daily)
df_daily.to_csv("predicted_power_generation_2024.csv")

print("Predicted power generation for 2024 saved to predicted_power_generation_2024.csv file.")

