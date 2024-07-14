# Gold Futures Closing Price Prediction

This project predicts the future closing prices of Gold futures (XAUUSD) using an LSTM model. The model is trained on historical price data and makes predictions for the next 365 days.

## Requirements
1. Clone the repository or download the project files.

2. Navigate to the project directory:

cd path/to/project/directory



3. Create and activate a virtual environment (optional but recommended):

- **On Windows:**

  ```
  python -m venv venv
  venv\Scripts\activate
  ```

- **On macOS and Linux:**

  ```
  python3 -m venv venv
  source venv/bin/activate
  ```
To run this project, you need the following packages installed:

- numpy
- pandas
- yfinance
- scikit-learn
- matplotlib
- torch

You can install these packages using pip:

pip install numpy pandas yfinance scikit-learn matplotlib torch


## Data

The historical data for Gold futures will be downloaded from yahoo , when you run the script


The `Date` column should contain the date of each record, and the `Close` column should contain the closing prices of Gold futures.

## Running the Code

1. Ensure you have all the required packages installed.
2. Run the script using Python:
python analyzer.py


The script will train an LSTM model on the historical data and make predictions for the next 365 days. The results will be plotted, showing the actual prices and the predicted future prices.

## About

This project demonstrates the use of an LSTM model for time series forecasting. The LSTM model is implemented using PyTorch and trained on historical Gold futures data.
The model predicts future closing prices based on past prices.

This is still an on-going project.


