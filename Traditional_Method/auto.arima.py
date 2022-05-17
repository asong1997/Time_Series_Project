import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.arima import auto_arima, ARIMA
from sklearn.metrics import max_error as Max_Error
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_absolute_percentage_error as MAPE


df = pd.read_excel(r"C:\Users\47382\Documents\Tencent Files\473826249\FileRecv\40dWLTCbatt02_re1.xlsx")
# 导入数据
data = pd.read_csv("Illinois_3339000_2016-2021.csv")
data["datetime"] = pd.to_datetime(data["datetime"], format="%Y-%m-%d")
data.set_index("datetime", inplace=True)
data = data["no3_mg_per_l"]
data_train = data[:-7]
data_test = data[-7:]

model = auto_arima(data_train, test="adf", start_p=1, start_q=1, max_p=3, max_q=3, stepwise=True, seasonal=True,
                   suppress_warnings=True, trace=True, error_action='ignore')

model.fit(data_train)
predict_data = model.predict(7)
fitted_data = model.predict_in_sample(data_train)


# 评价指标
def SMAPE(y_true, y_fore):
    return 2.0 * np.mean(np.abs(y_fore - y_true) / (np.abs(y_fore) + np.abs(y_true)))


print("Max_Error = {}".format(Max_Error(data_test, predict_data)))
print("MAE = {}".format(MAE(data_test, predict_data)))
print("MAPE = {}".format(MAPE(data_test, predict_data)))
print("SMAPE = {}".format(SMAPE(data_test, predict_data)))

STL = seasonal_decompose(data_train, period=365, extrapolate_trend=1)
# 趋势+季节可视化
plt.figure(figsize=(15, 10))
plt.subplot(3, 1, 1)
plt.plot(STL.trend, 'g')
plt.title("Trand")
plt.yticks([-0.5, 1])
plt.subplot(3, 1, 2)
plt.plot(STL.seasonal, "y")
plt.title("Season")
plt.subplot(3, 1, 3)
plt.plot(STL.resid, "b")
plt.title("Residual")
plt.savefig("./result/Arima_STL.png")
plt.show()

# 可视化预测结果
pd.DataFrame({
    "data_train": data_train[-30:],
    "fitted_data": pd.Series(fitted_data[-30:], index=data_train[-30:].index),
    "data_test": data_test,
    "predict_data": pd.Series(predict_data, index=data_test.index)
}).plot(legend=True, figsize=(10, 6))
plt.tick_params()
plt.xticks()
plt.yticks()
plt.title("Arima: MAPE={}".format(MAPE(data_test, predict_data)))
plt.savefig("./result/arima.png")
plt.show()

# 将评价指标写入txt文件
with open('./result/arima.txt', "w+") as f:
    print("pred:\n{}".format(predict_data), file=f)
    print("True:\n{}".format(data_test), file=f)
    print("arima_params(p,d,q) = {}".format(model.get_params()["order"]), file=f)
    print("Max_Error = {}".format(Max_Error(data_test, predict_data)), file=f)
    print("MAE = {}".format(MAE(data_test, predict_data)), file=f)
    print("MAPE = {}".format(MAPE(data_test, predict_data)), file=f)
    print("SMAPE = {}".format(SMAPE(data_test, predict_data)), file=f)
