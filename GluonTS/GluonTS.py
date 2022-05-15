#Third-party imports
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from gluonts.model import deepar
from gluonts.dataset import common
from gluonts.dataset.util import to_pandas
from gluonts.model.predictor import Predictor
# 读取数据

df = pd.read_csv("Twitter_volume_AMZN.csv", header=0, index_col=0)
data = common.ListDataset([{"start": df.index[0],
                            "target": df.value[:"2015-04-23 00:00:00"]}], freq="H")

# train_data = common.ListDataset([{"start": df.index[0],
#                             "target": df.value[:"2015-04-01 00:00:00"]}], freq="H")

# train_data =
#
# # train dataset: cut the last window of length "prediction_length", add "target" and "start" fields
# train_ds = ListDataset(
#     [{'target': x, 'start': start} for x in custom_dataset[:, :-prediction_length]],
#     freq=freq
# )
# # test dataset: use the whole dataset, add "target" and "start" fields
# test_ds = ListDataset(
#     [{'target': x, 'start': start} for x in custom_dataset],
#     freq=freq
# )



from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.mx import Trainer

estimator = deepar.DeepAREstimator(freq="H", prediction_length=24,
                                   trainer=Trainer(
                                       epochs=5,
                                       learning_rate=1e-3,
                                       num_batches_per_epoch=100
                                   ))
predictor = estimator.train(training_data=data)

for test_entry, forecast in zip(data, predictor.predict(data)):
    to_pandas(test_entry)[-60:].plot(linewidth=2)
    forecast.plot(color='g', prediction_intervals=[50.0, 90.0])
plt.grid(which='both')
plt.show()

print("end.....................")