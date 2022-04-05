import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data

from keras.models import  load_model
import streamlit as st








start = '2000-01-01'
end = '2022-01-01'


st.title('Stock Trend Prediction')


user_input = st.text_input('Enter Stock Ticker','MSFT')
df = data.DataReader('MSFT','yahoo',start,end)

# describe the data
st.subheader('Date from 2000 - 2022')
st.write(df.describe())

#visualisation

st.subheader('Closing price vs Time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing price vs Time chart with ma100 & ma200')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)



data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

print(data_training.shape)
print(data_testing.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)


#load my model
model = load_model('keras_model.h5')

#testing
past_100days = data_training.tail(100)
final_df = past_100days.append(data_testing,ignore_index = True)

input_data = scaler.fit_transform(final_df)
x_test = []
y_test = []

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test,y_test = np.array(x_test),np.array(y_test)

y_predicted = model.predict(x_test)

scaler = scaler.scale_
scaler_factor = 1/scaler[0]
y_predicted = y_predicted * scaler_factor
y_test = y_test * scaler_factor

#final graph

st.subheader('prediction vs original')
fig2 = plt.figure(figsize = (12,6))
plt.plot(y_test,'b',label = 'Original price')
plt.plot(y_predicted, 'r', label = 'predicted price')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
st.pyplot(fig2)


