def Stock_Price_LSTM_Data_Precesing(dfmt,memory_days=30,pre_days=10):
    import csv
    import pandas as pd

    # dfmt=pd.read_csv("./data/000538.XSHE.csv")

    # 打印DataFrame数据查看数据是否有缺失，以及每列数据的类型
    # print(dfmt.info())

    # dfmt.tail(20)

    # 查看含有nan的行
    # dfmt[dfmt.iloc[:,:-1].isna().any(axis=1)]

    # 删除空值项
    # dfmt.dropna(inplace=True)
    #print(dfmt.info())
    # 数据按时间正序排序
    # dfmt.sort_index(inplace=True)
    #test matplot
    #import matplotlib.pyplot as plt 
    #dfmt['Close'].plot()
    #plt.show()

    # 增加预测 label列
    dfmt['label']=dfmt['close'].shift(-pre_days)
    #print(dfmt)

    # 数据标准化
    from sklearn.preprocessing import StandardScaler
    # 生成一个数据标准化的对象
    scaler = StandardScaler()
    sca_X = scaler.fit_transform(dfmt.iloc[:,1:-1])
    #print(sca_X)

    #构建特征向量

    # 引入队列
    from collections import deque
    # 固定大小的队列，新的元素加入并且这个队列已满的时候，最老的元素会自动被移除掉
    deq=deque(maxlen=memory_days)

    X=[]

    for i in sca_X:
        deq.append(list(i))
        if len(deq)==memory_days:
            X.append(list(deq))
            
    X_lately=X[-pre_days:]
    X=X[:-pre_days]
    #print(len(X))
    #print(len(X_lately))

    y=dfmt['label'].values[memory_days-1:-pre_days]
    #print(len(y))
    #print(y)

    #查看特征向量
    import numpy as np
    X=np.array(X)
    y=np.array(y)
    print(X.shape)
    print(y.shape)
        
    return X,y,X_lately