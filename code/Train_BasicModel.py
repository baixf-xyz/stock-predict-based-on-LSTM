from Pre_Data import Stock_Price_LSTM_Data_Precesing
import pandas as pd
from keras.models import load_model

dfmt=pd.read_csv('./data/600519.XSHG.csv')
X,y,X_lately=Stock_Price_LSTM_Data_Precesing(dfmt)

# 构建LSTM神经网络模型

# 数据集划分
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,shuffle=False,test_size=0.1)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Dropout
from r2 import r2

model=Sequential()

# 构造三层神经网络
model.add(LSTM(10,input_shape=X.shape[1:],activation='relu',return_sequences=True,return_state=False))
# 防止过拟合 删除 0.1的神经元
model.add(Dropout(0.1))

# return_sequences = True返回整个序列,每一个time step都会输出，比如stack两层LSTM时候要这么设置。
# return_sequences = False只返回输出序列的最后一个time step的输出
model.add(LSTM(80,activation='relu',return_sequences=True,return_state=False))
model.add(Dropout(0.1))

model.add(LSTM(100,activation='relu'))
model.add(Dropout(0.1))

#构建全连接层
model.add(Dense(10,activation='relu'))
model.add(Dropout(0.5))

#输出层
model.add(Dense(1))

# 编译：优化器、损失函数、评价函数
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='mean_squared_error',
            metrics=['mape','mae',r2])

# 训练模型(炼丹)
history=model.fit(X_train,y_train,batch_size=32,epochs=20,validation_data=(X_test,y_test),validation_freq=1)

model.summary()


model.save('./basic_model/mt_model')

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

model.evaluate(X_test,y_test)

pre=model.predict(X_test)
# print(pre)

import matplotlib.pyplot as plt

df_time=dfmt.index[-len(y_test):]
# print(df_time)

from matplotlib.pyplot import MultipleLocator
#从pyplot导入MultipleLocator类，这个类用于设置刻度间隔
x_major_locator=MultipleLocator(30)
#把x轴的刻度间隔设置为30，并存在变量里
y_major_locator=MultipleLocator(1)
#把y轴的刻度间隔设置为1，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为30的倍数

plt.plot(df_time,y_test,color='red',label='price')
plt.plot(df_time,pre,color='green',label='predict')
plt.show()