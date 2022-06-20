import csv
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard
from Pre_Data import Stock_Price_LSTM_Data_Precesing
from r2 import r2


df=pd.read_csv('./data/600519.XSHG.csv')

# Checkpoint在每次训练中查找最优模型
filepath='./mt_basic_models/{val_mape:.2f}-{val_mae:.2f}-{val_r2:.2f}-{val_loss:.2f}-{epoch:02d}'
checkpoint=ModelCheckpoint(
    filepath=filepath,
    save_weights_only=False,
    monitor='val_mape',
    mode='min',
    save_best_only=True)
tensorboard_callback = TensorBoard(log_dir='./mt_logs', profile_batch=(10,20))
#print(filepath)
X,y,X_lately=Stock_Price_LSTM_Data_Precesing(df)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,shuffle=False,test_size=0.1)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Dropout

model=Sequential()

model.add(LSTM(10,input_shape=X.shape[1:],activation='relu',return_sequences=True))
model.add(Dropout(0.1))

model.add(LSTM(80,activation='relu',return_sequences=True))
model.add(Dropout(0.1))

model.add(LSTM(100,activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(10,activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(1))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mean_squared_error',
        metrics=['mape','mae',r2])

history=model.fit(X_train,y_train,batch_size=32,epochs=100,validation_data=(X_test,y_test),callbacks=[checkpoint,tensorboard_callback])

loss = history.history['loss']
val_loss = history.history['val_loss']
mape = history.history['mape']
val_mape = history.history['val_mape']
mae=history.history['mae']
val_mae=history.history['val_mae']
r2=history.history['r2']
val_r2=history.history['val_r2']

import matplotlib.pyplot as plt
    
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号 #有中文出现的情况，需要u'内容'

plt.figure()
plt.subplot(2,2,1)
plt.title('MSE')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')

plt.subplot(2,2,2)
plt.title('MAPE')
plt.plot(history.history['mape'], label='train')
plt.plot(history.history['val_mape'], label='test')

plt.subplot(2,2,3)
plt.title('MAE')
plt.plot(history.history['mae'], label='train')
plt.plot(history.history['val_mae'], label='test')

plt.subplot(2,2,4)
plt.title('R2')
plt.plot(history.history['r2'], label='train')
plt.plot(history.history['val_r2'], label='test')
plt.show()