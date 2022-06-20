# from Pre_Data import Stock_Price_LSTM_Data_Precesing
# import pandas as pd

# dfmt=pd.read_csv('./data/600519.XSHG.csv')
# X,y,X_lately=Stock_Price_LSTM_Data_Precesing(dfmt)


# from tensorflow.keras.models import load_model
# model = load_model('./basic_model/mt_model.h5')

# from tensorflow.keras.models import load_model
# from r2 import r2

# best_model=load_model('./basic_model/mt_model',custom_objects={'r2': r2})

# best_model.summary()

import csv
import pandas as pd
import tensorflow as tf
import kerastuner as kt
from Pre_Data import Stock_Price_LSTM_Data_Precesing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard
from r2 import r2

def model_builder(hp):

    # Tune the number of units in the LSTM Dropout and Dense layer
    # Choose an optimal value between 32-512
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32,default=32)
    hp_drop = hp.Float('p', min_value=0.1, max_value=0.5, step=0.1,default=0.1)
    hp_number_of_lstm = hp.Int('number_of_lstm', min_value = 1, max_value = 10, step = 1,default=1)
    hp_number_of_dense = hp.Int('number_of_dense', min_value = 1, max_value = 10, step = 1,default=1)
    hp_batch_size = hp.Int('batch_size', min_value = 32, max_value = 256, step = 32,default=32)

    dfmt=pd.read_csv('./data/600519.XSHG.csv')
    X,y,X_lately=Stock_Price_LSTM_Data_Precesing(dfmt)
    X_train, X_test, y_train, y_test = train_test_split(X,y,shuffle=False,test_size=0.1)

    model=Sequential()

    model.add(LSTM(units=hp_units,input_shape=X.shape[1:],activation='relu',return_sequences=True))
    model.add(Dropout(hp_drop))

    for i in range(hp_number_of_lstm):
        model.add(LSTM(units=hp_units,activation='relu',return_sequences=True))
        model.add(Dropout(hp_drop))

    model.add(LSTM(units=hp_units,activation='relu'))
    model.add(Dropout(hp_drop))

    for i in range(hp_number_of_dense):
        model.add(Dense(units=hp_units,activation='relu'))
        model.add(Dropout(hp_drop))

    model.add(Dense(1))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='mean_squared_error',
                  metrics=['mape','mae',r2])
    return model

#包装一个随机搜索器
tuner = kt.tuners.RandomSearch(
    model_builder,
    objective='val_mape',
    max_trials=1,
    executions_per_trial=1,
    directory='./test/my_dir',
    project_name='bestparam')

dfmt=pd.read_csv('./data/600519.XSHG.csv')

X,y,X_lately=Stock_Price_LSTM_Data_Precesing(dfmt)

X_train, X_test, y_train, y_test = train_test_split(X,y,shuffle=False,test_size=0.1)

#搜索最佳的超参数配置
random_search=tuner.search(X_train,y_train,batch_size=32,epochs=50,validation_data=(X_test,y_test))

# # #检索最佳模型
# # models = tuner.get_best_models(num_models=2)
# best_model = tuner.get_best_models(1)[0]
# # best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
# # #打印结果摘要
# tuner.results_summary()