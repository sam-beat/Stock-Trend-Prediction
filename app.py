import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
from keras.layers import Dense, LSTM
from keras.models import Sequential
from keras.layers import Dropout
import time
start = '2010-01-01'
end = '2019-12-31'

x_train = []
y_train = []


st.title('Stock Trend Prediction')


user_input = st.text_input('Enter Stock ticker', '')

if user_input == '':
  st.warning('Please Enter the Stock Ticker to predict the values')

if user_input != '': 
 df = data.DataReader(user_input,'yahoo', start, end)
 st.subheader('Data from 2010 - 2019')
 st.write(df.describe())



 st.subheader('Closing Price Vs Time Chart')
 fig = plt.figure(figsize = (12,6))
 plt.plot(df.Close)
 st.pyplot(fig)


 st.subheader('Closing Price Vs Time Chart with 100 Day Moving Average')
 ma100 = df.Close.rolling(100).mean()
 fig = plt.figure(figsize = (12,6))
 plt.plot(ma100)
 plt.plot(df.Close)
 st.pyplot(fig)


 st.subheader('Closing Price Vs Time Chart with 100D and 200D Moving Average')
 ma100 = df.Close.rolling(100).mean()
 ma200 = df.Close.rolling(200).mean()
 fig = plt.figure(figsize = (12,6))
 plt.plot(ma100,'g')
 plt.plot(ma200,'r')
 plt.plot(df.Close, 'b')
 st.pyplot(fig)


 data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
 data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

 print(data_training.shape)
 print(data_testing.shape)

 from sklearn.preprocessing import MinMaxScaler
 scaler = MinMaxScaler(feature_range = (0,1))
 data_training_array = scaler.fit_transform(data_training)



    
 with st.spinner(text = 'Please wait !!!! While the model is being trained'):
      for i in range(100,data_training_array.shape[0]):
         x_train.append(data_training_array[i-100: i])
         y_train.append(data_training_array[i,0])

      x_train, y_train = np.array(x_train), np.array(y_train)
  
 
      model= Sequential()

      model.add(LSTM(units = 50, activation = 'relu', return_sequences = True, 
      input_shape = (x_train.shape[1],1)))
      model.add(Dropout(0.2))


      model.add(LSTM(units = 50, activation = 'relu', return_sequences = True))
      model.add(Dropout(0.3))
          
      model.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
      model.add(Dropout(0.4))
          
          
      model.add(LSTM(units = 50, activation = 'relu'))
      model.add(Dropout(0.5))
          
      model.add(Dense(units = 1))



      model.compile(optimizer = 'adam', loss = 'mean_squared_error')
      model.fit(x_train,y_train, epochs = 50)
      time.sleep(5)
 st.success('Done')
 #Testing Part
 past_100_days = data_training.tail(100)
 final_df = past_100_days.append(data_testing,ignore_index = True)
 input_data = scaler.fit_transform(final_df)

 x_test = []
 y_test = []
 for i in range (100, input_data.shape[0]):
     x_test.append(input_data[i-100: i])
     y_test.append(input_data[i,0])

 x_test, y_test = np.array(x_test), np.array(y_test)

  # Model Prediction

 y_predicted = model.predict(x_test)
 scaler = scaler.scale_

 scale_factor = 1/scaler[0]
 y_predicted = y_predicted * scale_factor
 y_test = y_test * scale_factor

 # Final Graph
 st.subheader('Predictions Vs Original')
 predicted_vs_original_fig = plt.figure(figsize = (12,6 ))
 plt.plot(y_test,'b', label = 'Original Price')
 plt.plot(y_predicted, 'r', label = 'Predicted Price')
 plt.xlabel('Time')
 plt.ylabel('Price')
 plt.legend()
 st.pyplot(predicted_vs_original_fig)