from gluonts.mx import Trainer
import matplotlib.pyplot as plt
import pandas as pd
from gluonts.model import deepar
from gluonts.dataset import common
from gluonts.dataset.util import to_pandas
from sklearn.metrics import mean_absolute_percentage_error,max_error,mean_absolute_error
# 读取数据
df = pd.read_csv("Twitter_volume_AMZN.csv", header=0, index_col=0)
data = common.ListDataset([{
    "start": df.index[0],
    "target": df.value[:"2015-04-05 00:00:00"]
}], freq="5min")

train_ds = common.ListDataset([{
    "start": df.index[0],
    "target": df.value[:"2015-04-05 00:00:00"]
}], freq="5min")

test_ds = common.ListDataset([{
    "start": df.index[0],
    "target": df.value["2015-04-05 00:00:00":]
}], freq="5min")

# 初始化deepAR模型
trainer = Trainer(epochs=10)
estimator = deepar.DeepAREstimator(
    freq="5min", prediction_length=12, trainer=trainer)
predictor = estimator.train(training_data=train_ds, validation_data=test_ds)

for test_entry, forecast in zip(data, predictor.predict(data)):
    to_pandas(test_entry)[-60:].plot(linewidth=2)
    forecast.plot(color='g', prediction_intervals=[50.0, 90.0])
    y_pred = forecast.mean
    y_true = df.value["2015-04-05 00:00:00":][:12].values
    print("max_error: {}".format(max_error(y_pred, y_true)))    # max_error: 33.396217346191406
    print("mae: {}".format(mean_absolute_error(y_pred, y_true)))  # mae: 13.727095603942871
    print("mape: {}".format(mean_absolute_percentage_error(y_true, y_pred)))  # mape: 0.21627228519899847
    mape = mean_absolute_percentage_error(y_true, y_pred)
plt.grid(which='both')
plt.title(f"mape={mape}")
plt.savefig("deepar_result")
plt.show()


# 得到预测结果
# prediction = next(predictor.predict(data))
# print(prediction.mean)
# prediction.plot(output_file='graph.png')

