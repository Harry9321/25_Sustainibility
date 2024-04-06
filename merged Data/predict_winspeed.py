import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

# Load dataset
dataset = pd.read_csv('C:/Users/hariom/OneDrive/Desktop/Hakthon/merged Data/total_merge.csv', parse_dates=['DateTime'], index_col='DateTime')

# Drop any missing values
dataset.dropna(inplace=True)

# Select relevant features for modeling
features = ['Air_temperature', 'Power_generated_by_system', 'Pressure', 'Wind_speed']
data = dataset[features]

# Define the number of timesteps (hours) to consider for LSTM
timesteps = 24

# Create sequences of data
X, y = [], []
for i in range(len(data) - timesteps):
    X.append(data.values[i:i + timesteps])
    y.append(data.values[i + timesteps][3])  # speed is the 4th column

# Convert lists to arrays
X, y = np.array(X), np.array(y)

# Reshape data for LSTM input: [samples, timesteps, features]
X = X.reshape(X.shape[0], X.shape[1], len(features))

# Split the data into training and testing sets (80% train, 20% test)
split_index = int(0.8 * len(X))
X_train, X_test, y_train, y_test = X[:split_index], X[split_index:], y[:split_index], y[split_index:]

# Define the LSTM model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(units=50),
    Dense(units=1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=256, verbose=1)

# Make predictions
predictions = model.predict(X_test)

# Ensure non-negative predictions
predictions = np.maximum(predictions, 0)

# Calculate root mean squared error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("Root Mean Squared Error (RMSE):", rmse)

# Display the predicted air temperature for the first three months of 2024
print("Predicted Wind Speed for 2024:")
print(predictions)

# Convert predictions to DataFrame with dates as index
predicted_data = pd.DataFrame(predictions, index=dataset.index[timesteps:timesteps + len(predictions)], columns=["Predicted Wind Speed"])

# Set the index to start from January 1, 2024
start_date = pd.Timestamp('2024-01-01')
predicted_data.index = pd.date_range(start=start_date, periods=len(predictions), freq='H')

# Write the DataFrame to a CSV file
predicted_data.to_csv("predicted_wind_speed_2024.csv")

print("Predicted air temperature for 2024 saved to predicted_wind_speed_2024.csv file.")
