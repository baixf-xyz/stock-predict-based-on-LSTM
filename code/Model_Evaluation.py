import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from Pre_Data import Stock_Price_LSTM_Data_Precesing
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from r2 import r2
from matplotlib.pyplot import MultipleLocator
#从pyplot导入MultipleLocator类，这个类用于设置刻度间隔

dfmt=pd.read_csv('./data/600519.XSHG.csv')

X,y,X_lately=Stock_Price_LSTM_Data_Precesing(dfmt)

X_train, X_test, y_train, y_test = train_test_split(X,y,shuffle=False,test_size=0.1)

# best_model=load_model('./basic_model/mt_model',custom_objects={'r2': r2})

best_model=load_model('./mt_basic_models/6.97-135.48--2.83-30032.57-99',custom_objects={'r2': r2})

best_model.summary()

# val_loss,val_mape=best_model.evaluate(X_test,y_test)


pre_y=best_model.predict(X_test)
#print(pre_y)
#print(y_test)

# 计算准确率
pre_y=np.reshape(pre_y,len(pre_y))
print((1-np.fabs((pre_y-y_test)/y_test).mean())*100)

plot_model(best_model,
           'mtmodel.png',
           show_shapes=True,
           show_dtype=True,
           show_layer_names=True,
           dpi=200,
           layer_range=None,
           rankdir='LR',
           #show_layer_activations=True,
           expand_nested=False)

pre=best_model.predict(X_test)
#print(pre)
df1=pd.read_csv('./data/600519.XSHG.csv')
df_time=df1['Unnamed: 0'][-len(y_test):]

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