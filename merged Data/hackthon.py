import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Load the historical data
data = pd.read_csv('total_merge.csv')

# Print the column names to find the exact column name for 'Air temperature | (Â°C)'
print(data.columns)

# Define the sequence length (e.g., past 24 hours to predict the next hour)
sequence_length = 24

# Initialize lists to store input (X) and output (y) sequences
X, y = [], []

# Specify the column to predict
target_column = 'Air_temperature'  # Update this with the correct column name

# Create sequences of past values and future values for the target column
for i in range(len(data) - sequence_length):
    X.append(data[target_column][i:i+sequence_length].values.flatten())
    y.append(data[target_column][i+sequence_length])

# Convert lists to arrays
X = np.array(X)
y = np.array(y)

# Reshape X to match LSTM input shape (samples, time steps, features)
X = X.reshape(X.shape[0], sequence_length, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X.reshape(X.shape[0], -1))
y_scaled = scaler.fit_transform(y.reshape(-1, 1))

# Reshape X_scaled back to match LSTM input shape
X_scaled = X_scaled.reshape(X_scaled.shape[0], sequence_length, 1)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model (you may need to split the data into train and test sets)
model.fit(X_scaled, y_scaled, epochs=2, batch_size=32)

# Make predictions for the first three months of 2024
future_dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='H')
future_X = data[target_column][-sequence_length:].values.reshape(1, sequence_length, 1)
future_X_scaled = scaler.transform(future_X.reshape(future_X.shape[0], -1))
future_predictions_scaled = model.predict(future_X_scaled)
future_predictions = scaler.inverse_transform(future_predictions_scaled)

# Create a DataFrame for the predictions
future_predictions_df = pd.DataFrame({'DateTime': future_dates, target_column: future_predictions.flatten()})

# Save the predictions to a CSV file
future_predictions_df.to_csv('air_temperature_predictions.csv', index=False)