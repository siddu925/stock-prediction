import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
model_path = r'C:\Users\Siddesh G M\OneDrive\Desktop\new1\Stock Predictions Model.keras'
timesteps = 10  
input_features = 4

model = Sequential()
model.add(LSTM(50, input_shape=(timesteps, input_features))) 
model.add(Dense(1))  
model.compile(optimizer='adam', loss='mse')
# Load the pre-trained model
model = load_model(model_path)
try:
    model = load_model(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading the model: {e}")

# Streamlit header
st.header('Stock Market Predictor')

# User input for stock symbol
stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2018-01-01'
end = '2024-07-12'

# Fetch stock data
data = yf.download(stock, start, end)

# Display stock data
st.subheader('Stock Data')
st.write(data)

# Example empty data (replace with your actual data loading logic)
data_test = np.array([])  

# Check if data_test is not empty before scaling
if data_test.shape[0] > 0:
    scaler = MinMaxScaler()
    data_test_scaled = scaler.fit_transform(data_test)
else:
    print("Error: Data array is empty.")


# Splitting the data into train and test sets
data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])

# Scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scaled = scaler.fit_transform(data_test)

# Plot Price vs MA50
st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(data.Close, 'g', label='Close Price')
plt.legend()
st.pyplot(fig1)

# Plot Price vs MA50 vs MA100
st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(ma_100_days, 'b', label='MA100')
plt.plot(data.Close, 'g', label='Close Price')
plt.legend()
st.pyplot(fig2)

# Plot Price vs MA100 vs MA200
st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8, 6))
plt.plot(ma_100_days, 'r', label='MA100')
plt.plot(ma_200_days, 'b', label='MA200')
plt.plot(data.Close, 'g', label='Close Price')
plt.legend()
st.pyplot(fig3)

# Prepare data for prediction
x = []
y = []

for i in range(100, data_test_scaled.shape[0]):
    x.append(data_test_scaled[i-100:i])
    y.append(data_test_scaled[i, 0])

x, y = np.array(x), np.array(y)

# Predicting the stock prices
predicted_prices = model.predict(x)

# Inverse scaling to get the original prices
scale = 1 / scaler.scale_[0]
predicted_prices = predicted_prices * scale
y = y * scale

# Plot Original Price vs Predicted Price
st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8, 6))
plt.plot(y, 'g', label='Original Price')
plt.plot(predicted_prices, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)
