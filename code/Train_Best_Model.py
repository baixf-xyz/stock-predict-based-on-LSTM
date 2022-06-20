import csv
import pandas as pd
import time
import matplotlib.pyplot as plt
from Pre_Data import Stock_Price_LSTM_Data_Precesing
from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard
from r2 import r2

df=pd.read_csv('./data/002049.XSHE.csv')

# 设定参数值 炼丹最优解
units=[128]
p=[0.3]
lstm_layers=[7]
dense_layers=[4]
batch_size=[128]
learning_rate=[0.001]

for the_units in units:
    for the_p in p:
        for the_lstm_layers in lstm_layers:
            for the_dense_layers in dense_layers:
                for the_batch_size in batch_size:
                    for the_learning_rate in learning_rate:
                        # Checkpoint在每次训练中查找最优模型
                        checkpoint_filepath='./best_models/zggw_v1/{val_mape:.2f}-{epoch:02d}'
                        print(checkpoint_filepath)
                        checkpoint=ModelCheckpoint(
                            filepath=checkpoint_filepath,
                            save_weights_only=False,
                            monitor='val_mape',
                            mode='min',
                            #显示详细信息
                            verbose=1,
                            save_best_only=True)

                        #tensorboard
                        log_file=f'./logs/zggw_v1/lr_{the_learning_rate}_{int(time.time())}'
                        #print(log_file)
                        tensorboard = TensorBoard(log_file, profile_batch=(10,20))

                        X,y,X_lately=Stock_Price_LSTM_Data_Precesing(df)
                        
                        from sklearn.model_selection import train_test_split
                        X_train, X_test, y_train, y_test = train_test_split(X,y,shuffle=False,test_size=0.1)
                        
                        import tensorflow as tf
                        from tensorflow.keras.models import Sequential
                        from tensorflow.keras.layers import LSTM,Dense,Dropout
                        model=Sequential()
                        model.add(LSTM(the_units,input_shape=X.shape[1:],activation='relu',return_sequences=True))
                        model.add(Dropout(the_p))
                        
                        for i in range(the_lstm_layers):
                            model.add(LSTM(the_units,activation='relu',return_sequences=True))
                            model.add(Dropout(the_p))

                        model.add(LSTM(the_units,activation='relu'))
                        model.add(Dropout(the_p))

                        for i in range(the_dense_layers):
                            model.add(Dense(the_units,activation='relu'))
                            model.add(Dropout(the_p))

                        model.add(Dense(1))

                        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=the_learning_rate),
                              loss='mean_squared_error',
                              metrics=['mape','mae',r2])
                        #history=model.fit(X_train,y_train,batch_size=32,epochs=50,validation_data=(X_test,y_test),callbacks=[checkpoint,tensorboard_callback])
                        history=model.fit(X_train,y_train,batch_size=the_batch_size,epochs=300,validation_split=0.1,callbacks=[checkpoint,tensorboard])
                    
                        # 查看评价指标的变化趋势
                        pd.DataFrame(history.history).plot(figsize=(8, 5))
                        plt.grid(True)
                        plt.xlabel('epoch')
                        plt.show()

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