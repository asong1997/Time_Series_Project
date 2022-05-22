import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import max_error as Max_Error
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_absolute_percentage_error as MAPE

# 导入数据
data = pd.read_csv("Illinois_3339000_2016-2021.csv")
data["datetime"] = pd.to_datetime(data["datetime"], format="%Y-%m-%d")
data.set_index("datetime", inplace=True)
data = data["no3_mg_per_l"]
data_train = data[:-7]
data_test = data[-7:]

# 建模
model = ExponentialSmoothing(data_train, trend="add", seasonal="add", seasonal_periods=365).fit()
predict_data = model.forecast(7)
fitted_data = model.fittedvalues


# 评价指标
def SMAPE(y_true, y_fore):
    return 2.0 * np.mean(np.abs(y_fore - y_true) / (np.abs(y_fore) + np.abs(y_true)))


print("Max_Error = {}".format(Max_Error(data_test, predict_data)))
print("MAE = {}".format(MAE(data_test, predict_data)))
print("MAPE = {}".format(MAPE(data_test, predict_data)))
print("SMAPE = {}".format(SMAPE(data_test, predict_data)))


# 趋势+季节可视化
plt.figure(figsize=(15, 10))
plt.subplot(3, 1, 1)
plt.plot(model.trend, 'g')
plt.title("Trand")
plt.yticks([-0.5, 1])
plt.subplot(3, 1, 2)
plt.plot(model.season, "y")
plt.title("Season")
plt.subplot(3, 1, 3)
plt.plot((data_train - model.season - model.trend), "b")
plt.title("Residual")
plt.savefig("./result/ES_STL.png")
plt.show()

# 可视化预测结果
pd.DataFrame({
    "data_train": data_train[-30:],
    "fitted_data": fitted_data[-30:],
    "data_test": data_test,
    "predict_data": predict_data
}).plot(legend=True, figsize=(10, 6))
plt.tick_params()
plt.xticks()
plt.yticks()
plt.title("Exponential Smoothing: MAPE={}".format(MAPE(data_test, predict_data)))
plt.savefig("./result/exponential_smoothing.png")
plt.show()

# 将评价指标写入txt文件
with open('./result/exponential_smoothing.txt', "w+") as f:
    print("pred:\n{}".format(predict_data), file=f)
    print("True:\n{}".format(data_test), file=f)
    print("Max_Error = {}".format(Max_Error(data_test, predict_data)), file=f)
    print("MAE = {}".format(MAE(data_test, predict_data)), file=f)
    print("MAPE = {}".format(MAPE(data_test, predict_data)), file=f)
    print("SMAPE = {}".format(SMAPE(data_test, predict_data)), file=f)
