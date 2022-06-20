## 基于LSTM神经网络的股票价格趋势预测的研究与实现

### 所需环境

Anaconda python3.6

### python库

jqdatasdk
matplotlib
sklearn
pydot
graphviz
tensorboard
keras-tuner

### 目录文件说明

├─code 
│ ├─Basic_Model_Update.py :训练基本模型（需调用预处理文件Pre_Data.py）
│ ├─Get_Data.py ：聚宽平台获取数据代码（账号已过期）
│ ├─Model_Evaluation.py :基本模型评估（计算准确率、打印模型网络结构、绘制预测图)
│ ├─Model_Evaluation_best.py ：最优化模型评估函数（计算准确率、打印模型网络结构、绘制预测图)
│ ├─Model_Evaluation_Update.py ：优化模型评估函数（计算准确率、打印模型网络结构、绘制预测图)
│ ├─plot.py ：可视化查看爬取数据基本信息
│ ├─Pre_Data.py ：数据预处理封装函数（删除空值、构造输入输出数据特征、数据标准化操作）
│ ├─r2.py ：深度学习评价函数中计算r^2的函数
│ ├─requirment.txt ：需要安装的python库
│ ├─Select_Best_Model.py ：使用keras-tuner进行超参数优化寻找最有超参数
│ ├─test.py ：调试代码文件
│ ├─Train_BasicModel.py ：训练基本模型（已包含预处理代码）
│ ├─Train_Best_Model.py ：带入最优超参数进行模型的重新训练
│ └─__pycache__ 
│   ├─Pre_Data.cpython-36.pyc 
│   └─r2.cpython-36.pyc 
├─data 
│ ├─000002.XSHE.csv ：万科 A 股票数据信息
│ ├─000538.XSHE.csv ：云南白药股票数据信息
│ ├─002049.XSHE.csv ：紫光国微股票数据信息
│ └─600519.XSHG.csv ：贵州茅台股票数据信息