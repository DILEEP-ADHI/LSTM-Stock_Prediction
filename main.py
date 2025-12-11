from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

data=pd.read_csv("MicrosoftStock.csv")
'''print(data.head())
print(data.info())
print(data.describe())'''

#Data Visualization
#open and close prices of time
'''plt.figure(figsize=(12,6))
plt.plot(data['date'],data['open'],label="Open",color="blue")
plt.plot(data['date'],data['close'],label="Close",color="red")
plt.title("Open-Close Price over time")
plt.legend()
#plt.show()

#check for outliers
#Trading Volume
plt.figure(figsize=(12,6))
plt.plot(data['date'],data['volume'],label="Volume",color="orange")
plt.title("Stock volume over time")
#plt.show()

#drop non-numeric columns
num_data=data.drop(['date','Name'],axis=1)
num_data.head()
#check correlation
plt.figure(figsize=(12,6))
sns.heatmap(num_data.corr(),annot=True,cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
#plt.show()'''

#Convert date to datetime
data['date']=pd.to_datetime(data['date'])
prediction=data.loc[
    (data['date']>datetime(2013,1,1))&
    (data['date']<datetime(2018,1,1))
]

'''plt.figure(figsize=(12,6))
plt.plot(data['date'],data['close'],color="blue")
plt.xlabel("Date")
plt.ylabel("Close")
plt.title("Price over time")
plt.show()'''

#Prepare for LSTM model
stock_close = data.filter(["close"])
dataset = stock_close.values #convert to numpy

training_data_len=int(np.ceil(len(dataset)*0.8))

# Split the data into training and test sets before scaling
train_data_unscaled = dataset[:training_data_len]
test_data_unscaled = dataset[training_data_len:]
y_test_actual = test_data_unscaled.flatten() # Actual unscaled test values

#Data Pre-processing
# Preprocessing Stages: Fit scaler *only* on training data for LSTM
scaler = StandardScaler()
scaled_train_data = scaler.fit_transform(train_data_unscaled)
scaled_test_data = scaler.transform(test_data_unscaled)

window_size = 60
X_train, y_train = [], []

for i in range(window_size, len(scaled_train_data)):
    X_train.append(scaled_train_data[i-window_size:i, 0])
    y_train.append(scaled_train_data[i,0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#SARIMAX MODEL
# Import for SARIMAX model
import pmdarima as pm

exog_features = ['open', 'high', 'low', 'volume']
exogenous_data = data[exog_features].values

# Split the exogenous data based on the training data length
exog_train = exogenous_data[:training_data_len]
exog_test = exogenous_data[training_data_len:]

sarimax_model = pm.auto_arima(
    y=train_data_unscaled,
    exogenous=exog_train, # Multiple columns now included
    start_p=1, start_q=1,
    max_p=3, max_q=3,
    m=5,
    d=None, D=None,
    seasonal=True,
    start_P=0, start_Q=0,
    trace=False,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True
)

print(sarimax_model.summary())

#make predictions based on the test set
n_periods = len(test_data_unscaled)
predictions_sarimax = sarimax_model.predict(
    n_periods=n_periods,
    exogenous=exog_test # Must provide the future values for the regressors
)

#Performance Metrics Calculation (SARIMAX)
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse
from math import sqrt

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

metrics_sarimax = {
    'MAE': mae(y_test_actual, predictions_sarimax),
    'RMSE': sqrt(mse(y_test_actual, predictions_sarimax)),
    'MAPE': calculate_mape(y_test_actual, predictions_sarimax)
}
print(metrics_sarimax)

#LSTM Model
#Build the model
model = keras.models.Sequential()
#1st layer
model.add(keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1],1)))
#2nd layer
model.add(keras.layers.LSTM(64, return_sequences=False))
#Dense layer
model.add(keras.layers.Dense(128, activation="relu"))
#Drop half inputs to avoid overfitting
model.add(keras.layers.Dropout(0.5))
#final layer
model.add(keras.layers.Dense(1))

model.summary()

model.compile(optimizer="adam", loss="mae", metrics=[keras.metrics.RootMeanSquaredError()])
training = model.fit(X_train, y_train, epochs=20, batch_size=32)

full_scaled_data_for_testing = np.concatenate((scaled_train_data[-window_size:], scaled_test_data), axis=0)
X_test = []

for i in range(window_size, len(full_scaled_data_for_testing)):
    X_test.append(full_scaled_data_for_testing[i-window_size:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1 ))

predictions_scaled_lstm = model.predict(X_test)
predictions_lstm = scaler.inverse_transform(predictions_scaled_lstm).flatten()

#Metrics for LSTM
metrics_LSTM = {
    'MAE': mae(y_test_actual, predictions_lstm),
    'RMSE': sqrt(mse(y_test_actual, predictions_lstm)),
    'MAPE': calculate_mape(y_test_actual, predictions_lstm)
}
print(metrics_LSTM)

#Plotting LSTM vs SARIMAX model
train = data[:training_data_len].copy()
test = data[training_data_len:].copy()

test['LSTM Predictions'] = predictions_lstm
test['SARIMAX Predictions'] = predictions_sarimax


plt.figure(figsize=(14, 8))
plt.plot(train['date'], train['close'], label="Train (Actual)", color='blue')
plt.plot(test['date'], test['close'], label="Test (Actual)", color='orange')
plt.plot(test['date'], test['LSTM Predictions'], label="LSTM Predictions", color='red')
plt.plot(test['date'], test['SARIMAX Predictions'], label="SARIMAX Predictions", color='green', linestyle='--')

plt.title("Stock Price Predictions: LSTM vs SARI MAX Model (Multi-Feature)")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.grid(True)
plt.show()


