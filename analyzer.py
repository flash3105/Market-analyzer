import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# Fetch historical data for Gold futures from Yahoo Finance
ticker = "GC=F"
data = yf.download(ticker, start="2010-01-01", end="2023-01-01") 
data.reset_index(inplace=True)

# Load data from CSV file
#data = pd.read_csv('XAUUSD_historical_data.csv', parse_dates=['Date'], index_col='Date')
#data.reset_index(inplace=True)

# Prepare the data
data = data[['Date', 'Close']]
data.set_index('Date', inplace=True)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Prepare the dataset
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 100
X, Y = create_dataset(scaled_data, time_step)

# TimeSeriesSplit
k = 5
tscv = TimeSeriesSplit(n_splits=k)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=50, num_layers=2, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(50, 25)
        self.tanh = nn.Tanh()  # Tanh activation function
        self.fc2 = nn.Linear(25, 1)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm1(input_seq)
        lstm_out = lstm_out[:, -1, :]  # Get the output of the last time step
        x = self.fc1(lstm_out)
        x = self.tanh(x)  # Apply Tanh activation
        predictions = self.fc2(x)
        return predictions

# Convert to tensors
X = torch.tensor(X, dtype=torch.float32).view(-1, time_step, 1)
Y = torch.tensor(Y, dtype=torch.float32).view(-1, 1)

# Train the model on the entire dataset
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
model.train()
for epoch in range(100):  # Increase the number of epochs as needed
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, Y)
    loss.backward()
    optimizer.step()

# Predict future values
model.eval()
with torch.no_grad():
    # Prepare the input sequence
    input_seq = torch.tensor(scaled_data[-time_step:], dtype=torch.float32).view(1, time_step, 1)
    future_predictions = []
    for _ in range(365):  # Number of days to predict (for next year)
        prediction = model(input_seq)
        future_predictions.append(prediction.item())
        # Update the input sequence
        input_seq = torch.cat((input_seq[:, 1:, :], prediction.view(1, 1, 1)), dim=1)

# Inverse transform the predictions
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Create a date range for future predictions
last_date = data.index[-1]
future_dates = pd.date_range(start=last_date, periods=365 + 1, freq='D')[1:]  # Remove the start date itself

# Enable interactive mode
plt.ion()

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(data.index, scaler.inverse_transform(scaled_data), label='Actual Price')
plt.plot(future_dates, future_predictions, label='Future Predictions', linestyle='dashed')
plt.title('Gold Futures Closing Price Prediction')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.show()

# Keep the plot open
plt.ioff()
plt.show()
