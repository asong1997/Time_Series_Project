from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt # Visualization
from datetime import datetime, date
from fbprophet import Prophet
import seaborn as sns # Visualization
from colorama import Fore
import pandas as pd
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import tensorflow as tf
from tensorflow.keras import Sequential,layers,utils,losses
from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard

from scipy.stats import pearsonr
from scipy.spatial.distance import cdist

import warnings # Supress warnings
warnings.filterwarnings('ignore')

#准备数据
dataset = pd.read_csv("MA.csv")

#将第一列设置为时间格式
dataset['ds'] = pd.to_datetime(dataset['ds'], format = '%Y/%m/%d')

# # 特征
# X=dataset.drop(columns=[vari],axis=1)
# y=dataset[vari]
# print(X.shape)
# print(y.shape)
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15,shuffle=False,random_state=666)


#设置85%的数据作为训练集，15%作为测试集
train_size = int(0.85* len(dataset))
# test_size= int(0.15* len(multivariate_df))

train = dataset.iloc[:train_size, :]
# test = multivariate_df.iloc[:test_size, :]
x_train, y_train = pd.DataFrame(dataset.iloc[:train_size, [0,2,3,4]]), pd.DataFrame(dataset.iloc[:train_size, 1])
x_valid, y_valid = pd.DataFrame(dataset.iloc[train_size:, [0,2,3,4]]), pd.DataFrame(dataset.iloc[train_size:, 1])


# 训练模型
model = Prophet()

model.add_regressor('pdcp_sduoctul')
model.add_regressor('DtchPrbAssnMeanUl_Rate')
model.add_regressor('DtchPrbAssnMeandl_Rate')

# # 模型拟合
model.fit(train)
#拟合模型
# model = Prophet(changepoint_prior_scale=0.05)
# model.fit(dataset)
# model = Prophet(interval_width=0.95).fit(dataset)
# 在测试集上预测
y_pred = model.predict(x_valid)
# future = model.make_future_dataframe(periods=5)
# future.tail()

forecast = model.predict(train)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

model.plot(forecast)
plt.show()
# 计算MAE以及RMSE
score_mae = mean_absolute_error(y_valid, y_pred['yhat'])
score_rmse = math.sqrt(mean_squared_error(y_valid, y_pred['yhat']))
score_mape=math.sqrt(mean_absolute_error(y_valid,y_pred['yhat']))
score_smape=math.sqrt(mean_absolute_error(y_valid,y_pred['yhat']))
print(Fore.GREEN + 'RMSE: {}'.format(score_rmse))
print(Fore.GREEN + 'MAE: {}'.format(score_mae))
print(Fore.GREEN + 'MAPE: {}'.format(score_mape))
print(Fore.GREEN + 'SMAPE: {}'.format(score_mape))
# 画出预测图象
f, ax = plt.subplots(1)
f.set_figheight(10)
f.set_figwidth(15)

# model.plot(train, ax=ax)
model.plot(y_pred, ax=ax)
sns.lineplot(x=x_valid['ds'], y=y_valid['y'], ax=ax, color='orange', label='Ground truth') #navajowhite

ax.set_title(f'Prediction \n MAE: {score_mae:.2f}, RMSE: {score_rmse:.2f}', fontsize=25)
ax.set_xlabel(xlabel='Date', fontsize=25)
ax.set_ylabel(ylabel='Depth to Groundwater', fontsize=25)

plt.show()
# 预测整个数据集
# future = model.make_future_dataframe(periods=7)
# future.tail()
# forecast = model.predict(future)
# forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
# model.plot(forecast)
# plt.show()

def creat_dataset(x,y,seq_len):
    feature=[]
    target=[]
    for i in range(0,len(x)-seq_len,1):
        data=x.iloc[i:i+seq_len] #序列数据
        label=y.iloc[i+seq_len] #标签数据
        feature.append(data)
        target.append(label)
    return np.array(feature),np.array(target)
train_dataset,train_labels=creat_dataset(x_train,y_train,5)
test_dataset,test_labels=creat_dataset(x_valid,y_valid,5)
test_preds=model.predict(test_dataset.astype(np.float32),verbose=1)
# 欧氏距离
def eculidDisSim(x,y):
    E=np.sqrt(sum(pow(a-b,2) for a,b in zip(x,y)))
    return 1/(1+E)

# 曼哈顿距离
def manhattanDisSim(x,y):
    M=sum(abs(a-b) for a,b in zip(x,y))
    return 1/(1+M)

# 余弦相似度
def cosSim(x,y):
    tmp=np.sum(x*y)
    non=np.linalg.norm(x)*np.linalg.norm(y)
    return np.round(tmp/float(non),9)

# 皮尔逊相关系数
def pearsonrSim(x,y):
    return pearsonr(x,y)[0]

# 弗雷歇距离
def eucl_dist(x,y):
    dist = np.linalg.norm(x-y)
    return dist
def _c(ca,i,j,P,Q):
    if ca[i,j] > -1:
        return ca[i,j]
    elif i == 0 and j == 0:
        ca[i,j] = eucl_dist(P[0],Q[0])
    elif i > 0 and j == 0:
        ca[i,j] = max(_c(ca,i-1,0,P,Q),eucl_dist(P[i],Q[0]))
    elif i == 0 and j > 0:
        ca[i,j] = max(_c(ca,0,j-1,P,Q),eucl_dist(P[0],Q[j]))
    elif i > 0 and j > 0:
        ca[i,j] = max(min(_c(ca,i-1,j,P,Q),_c(ca,i-1,j-1,P,Q),_c(ca,i,j-1,P,Q)),eucl_dist(P[i],Q[j]))
    else:
        ca[i,j] = float("inf")
    return ca[i,j]
def discret_frechet(P,Q):
    ca = np.ones((len(P),len(Q)))
    ca = np.multiply(ca,-1)
    D=_c(ca,len(P)-1,len(Q)-1,P,Q)
    return 1/(1+D)

# 豪斯多夫距离
def OneWayHausdorffDistance(ptSetA, ptSetB):
    ptSetA=ptSetA.reshape(-1,1)
    ptSetB=ptSetB.reshape(-1,1)
    dist = cdist(ptSetA, ptSetB, metric='euclidean')
    return np.max(np.min(dist, axis=1))
def HausdorffDistance(ptSetA, ptSetB):
    res = np.array([
        OneWayHausdorffDistance(ptSetA, ptSetB),
        OneWayHausdorffDistance(ptSetB, ptSetA)
    ])
    H=np.max(res)
    return 1/(1+H)


window = 6
ot = pd.DataFrame()
# A=list(float(x_valid.values))
# ot['value'] = pd.DataFrame(A)[0]
ot['value']=pd.DataFrame(test_labels)[0]
ot['is_outlier'] = 2
outlier = []
outlier_y = []
for i in range(0, len(test_labels) - window, 1):
    # if i+window<len(test_labels):
    # print("window=",i,"~",i+window)
    e = eculidDisSim(test_labels[i:i + window], test_preds[i:i + window])
    m = manhattanDisSim(test_labels[i:i + window], test_preds[i:i + window])
    c = cosSim(test_labels[i:i + window], test_preds[i:i + window])
    p = pearsonrSim(test_labels[i:i + window], test_preds[i:i + window])
    d = discret_frechet(test_labels[i:i + window], test_preds[i:i + window])
    h = HausdorffDistance(test_labels[i:i + window], test_preds[i:i + window])

    dis = (e + m + c + p + d + h) / 6
    # dis=(d+h)/2
    print("window=", i, "~", i + window, "Mean Similarity:", dis)
    # print("window = ",i,"~",i+window,"eculid distance Similarity = ",eculidDisSim(test_labels[i:i+window],test_preds[i:i+window]))
    # print("window = ",i,"~",i+window,"manhatan Similarity = ",manhattanDisSim(test_labels[i:i+window],test_preds[i:i+window]))
    # print("window = ",i,"~",i+window,"cos Similarity = ",cosSim(test_labels[i:i+window],test_preds[i:i+window]))
    # print("window = ",i,"~",i+window,"personr Similarity = ",pearsonrSim(test_labels[i:i+window],test_preds[i:i+window]))

    # print("window = ",i,"~",i+window,"Fréchet distance = ",discret_frechet(test_labels[i:i+window],test_preds[i:i+window]))
    # print("window = ",i,"~",i+window,"Hausdorff distance = ",HausdorffDistance(test_labels[i:i+window],test_preds[i:i+window]))
    # else:
    # print("window = ",i,"~",len(test_preds)-1,"Fréchet distance = ",discret_frechet(test_labels[i:len(test_labels)-1],test_preds[i:len(test_preds)-1]))
    if (dis < 0.8):
        outlier.append(i)
        outlier_y.append(dis)
        ot['is_outlier'].iloc[i:i + window] = 1
    else:
        if (ot['is_outlier'][i] == 2):
            ot['is_outlier'].iloc[i:] = 0
print(outlier)
ot.head(18)

fig=plt.figure(figsize=(18,3))
fig=ot['value'].plot(color='blue',label="True Value")
fig=ot['value'][ot['is_outlier']==1].plot(color='red', marker='o',markersize=6,label="异常时间段")
fig=plt.plot(test_preds,color='green',label="Pred Value")
fig=plt.title("test")
fig=plt.legend(loc='best')

plt.figure(figsize=(16,8))
plt.plot(test_labels,label="True Value")
plt.plot(test_preds,label='Pre Value')
plt.legend(loc='best')
plt.show()