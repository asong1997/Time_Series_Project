import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
# from pmdarima.arima import auto_arima, ARIMA
from sklearn.metrics import max_error as Max_Error
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_absolute_percentage_error as MAPE
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.stattools import adfuller as ADF



import tushare as ts
data = ts.get_hist_data('600519')
data = data['close']
# data = data.reindex(index=data.index[::-1])
data = data.iloc[::-1]
data_train = data[:-12]
data_test = data[-12:]

STL = seasonal_decompose(data, period=12, extrapolate_trend=1)
# 趋势+季节可视化
plt.figure(figsize=(15, 10))
plt.subplot(3, 1, 1)
plt.plot(STL.trend, 'g')
plt.title("Trand")
plt.subplot(3, 1, 2)
plt.plot(STL.seasonal, "y")
plt.title("Season")
plt.subplot(3, 1, 3)
plt.plot(STL.resid, "b")
plt.title("Residual")
plt.show()



def data_diff(data):
    fig = plt.figure(facecolor='white', figsize=(12, 8))
    data.plot(legend=True, title="raw_data")
    while True:
        adf_res = ADF(data)
        print("adf_result: {}".format(adf_res))
        i = 1
        if adf_res[1] > 0.05:
            data = data.diff().dropna()
            fig = plt.figure(facecolor='white', figsize=(12, 8))
            data.plot(legend=True, title="diff({})_data".format(i))
        else:
            break
        i += 1
    return data

# 平稳性检验
stationary_data = data_diff(data)

# 白噪声检验
# 原假设：是随机的，既是白噪声序列。
# 它主要返回一个p值。
# p值大，接受原假设；p值小，拒绝原假设。分割线：0.05。
# 0.05置信区间以下，可以认为出现显著的自回归关系，且序列为非白噪声。
print(acorr_ljungbox(stationary_data, lags=20))

# 画出差分后序列得到的ACF图和PACF图
plot_acf(stationary_data)
plot_pacf(stationary_data)

# BIC准则确定p、q值
pMax = 10
qMax = 10
bics = list()
for p in range(pMax + 1):
    tmp = list()
    for q in range(qMax + 1):
        try:
            tmp.append(sm.tsa.ARIMA(stationary_data, order=(p, 1, q)).fit().bic)
        except Exception as e:
            print(str(e))
            tmp.append(1e+10)  # 加入一个很大的数
    bics.append(tmp)
bics = pd.DataFrame(bics)
p, q = bics.stack().idxmin()

model = sm.tsa.ARIMA(stationary_data, order=(p, 1, q)).fit()
predict_data = model.predict(12)


pd.DataFrame({
    "data_train": data_train[-30:],
    # "fitted_data": pd.Series(fitted_data[-30:], index=data_train[-30:].index),
    "data_test": data_test,
    "predict_data": pd.Series(predict_data, index=data_test.index)
}).plot(legend=True, figsize=(10, 6))
plt.tick_params()
plt.xticks()
plt.yticks()
# plt.title("Arima: MAPE={}".format(MAPE(data_test, predict_data)))
# plt.savefig("./result/arima.png")
plt.show()


print("end..........................")





