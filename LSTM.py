#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import json

def get_crypto_price(symbol, interval,n_interval, start_date, end_date):
 api_key = 'f1d337a829dd41a0a391dd2fc8e2d232'
 api_url = f'https://api.twelvedata.com/time_series?symbol={symbol}&exchange=binance&interval={interval}&start_date={start_date}&end_date={end_date}&apikey={api_key}'
 raw = requests.get(api_url).json()
 df = pd.DataFrame(raw['values']).set_index('datetime')
 df = df.iloc[::-1]
 return df

start_date = '2020-10-22'
end_date = '2023-02-04'
dataset = get_crypto_price('BTC/USD', '4h', 5000, start_date, end_date)
dataset

dataset.info()

dataset=dataset.iloc[:,:].astype(float)

dataset.info()

dataset.hist(figsize=(22,8))

dataset.plot(subplots=True,figsize=(12,12))

dataset=pd.DataFrame(dataset)

TRAIN_SPLIT=int(len(dataset)*0.8)

TRAIN_SPLIT

dataset=np.array(dataset)

def multivariate_data(dataset,target,start_index,end_index,history_size,
                      target_size,step,single_step=False):
    data=[]
    labels=[]

    start_index=start_index+history_size
    if end_index is None:
       end_index=len(dataset)-target_size

    for i in range(start_index, end_index):
       indices=range(i-history_size,i,step)
       data.append(dataset[indices])

       if single_step:
           labels.append(target[i+target_size])
       else:
           labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)

past_history=16
STEP=1
n_step=3

x_train_multi, y_train_multi=multivariate_data(dataset,dataset[:,3],0,TRAIN_SPLIT,past_history,n_step,STEP,single_step=False)

x_test_multi, y_test_multi=multivariate_data(dataset,dataset[:,3],TRAIN_SPLIT,None,past_history,n_step,STEP,single_step=False)

print(x_train_multi.shape)
print(x_test_multi.shape)

from sklearn.preprocessing import MinMaxScaler

scalers={}
for i in range(x_train_multi.shape[1]):
      scalers [i]=MinMaxScaler(feature_range=(0,1))
      x_train_multi[:,i,:]=scalers[i].fit_transform(x_train_multi[:,i,:])
for i in range(x_test_multi.shape[1]):
      x_test_multi[:,i,:]=scalers[i].transform(x_test_multi[:,i,:])

x_train_multi.shape, x_test_multi.shape

scaler_y=MinMaxScaler(feature_range=(0,1))
y_train_multi=scaler_y.fit_transform(y_train_multi)
y_test_multi=scaler_y.transform(y_test_multi)

y_test_multi.shape, y_train_multi.shape

from keras.models import Sequential
from keras.layers import LSTM,Dense, Conv1D ,MaxPooling1D,Flatten,Dropout

LSTM_model=Sequential()
LSTM_model.add(LSTM(50,input_shape=x_train_multi.shape[-2:],activation='relu',return_sequences=True))
LSTM_model.add(LSTM(50,activation='relu', return_sequences=True))
LSTM_model.add(LSTM(50,activation='relu'))
LSTM_model.add(Dense(n_step, activation='selu'))
LSTM_model.compile(optimizer='adam',loss='mse')

history_LSTM=LSTM_model.fit(x_train_multi, y_train_multi, epochs=100, batch_size=16, validation_data=(x_test_multi, y_test_multi))

plt.figure(figsize=(8,6))
plt.plot(history_LSTM.history['loss'])
plt.plot(history_LSTM.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

y_train_N_LSTM=LSTM_model.predict(x_train_multi)

y_train_N_LSTM.shape

from sklearn.metrics import mean_squared_error

mse_LSTM=mean_squared_error(y_train_multi, y_train_N_LSTM)

mse_LSTM

y_train_N_LSTM_inv=scaler_y.inverse_transform(y_train_N_LSTM)
y_train_multi_inv=scaler_y.inverse_transform(y_train_multi)

fig=plt.figure(figsize=(8,6))
plt.plot(y_train_multi_inv[:,2] , color='b',label='Market' )
plt.plot(y_train_N_LSTM_inv[:,2] , color='r',label='Trained Data' )
plt.legend()

predict_LSTM=LSTM_model.predict(x_test_multi)

predict_LSTM.shape

mse_LSTM=mean_squared_error(y_test_multi, predict_LSTM)

mse_LSTM

predict_LSTM_inv=scaler_y.inverse_transform(predict_LSTM)
y_test_multi_inv=scaler_y.inverse_transform(y_test_multi)

fig=plt.figure(figsize=(8,6))
plt.plot(y_test_multi_inv[:,2] , color='b',label='Real' )
plt.plot(predict_LSTM_inv[:,2] , color='r',label='predicted' )
plt.legend()











# In[ ]:




