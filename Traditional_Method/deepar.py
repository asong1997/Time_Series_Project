import torch
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting import GroupNormalizer, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import NormalDistributionLoss
from pytorch_forecasting.models.deepar import DeepAR
from pytorch_forecasting.metrics import SMAPE, MAPE, MAE

# 导入数据
data = pd.read_csv("Illinois_3339000_2016-2021.csv")
data["datetime"] = pd.to_datetime(data["datetime"], format="%Y-%m-%d")
data["month"] = data["datetime"].dt.month.astype(str).astype("category")
data["time_idx"] = range(data.shape[0])

max_encoder_length = 60
max_prediction_length = 12
training_cutoff = data["time_idx"].max() - max_prediction_length

training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="no3_mg_per_l",

    categorical_encoders={"USGS_site": NaNLabelEncoder().fit(data.USGS_site)},
    group_ids=["USGS_site"],
    static_categoricals=[],
    min_encoder_length=max_encoder_length,
    max_encoder_length=max_encoder_length,
    min_prediction_length=max_prediction_length,
    max_prediction_length=max_prediction_length,
    time_varying_unknown_reals=["no3_mg_per_l"],

    time_varying_known_reals=["time_idx"],
    target_normalizer=GroupNormalizer(groups=["USGS_site"]),
    add_relative_time_idx=False,
    add_target_scales=True,
    randomize_length=None,
)
validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)
batch_size = 64
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

# save datasets
training.save("training.pkl")
validation.save("validation.pkl")

early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min")
lr_logger = LearningRateMonitor()

trainer = pl.Trainer(
    max_epochs=10,
    gpus=None,
    gradient_clip_val=0.1,
    limit_train_batches=30,
    limit_val_batches=3,
    callbacks=[lr_logger, early_stop_callback],
)

deepar = DeepAR.from_dataset(
    training,
    learning_rate=0.1,
    hidden_size=32,
    dropout=0.1,
    loss=NormalDistributionLoss(),
    log_interval=10,
    log_val_interval=3,
)
print(f"Number of parameters in network: {deepar.size()/1e3:.1f}k")

trainer.fit(
    deepar,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

best_model_path = trainer.checkpoint_callback.best_model_path
best_model = deepar.load_from_checkpoint(best_model_path)

# calcualte mean absolute error on validation set
actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)])
predictions = best_model.predict(val_dataloader)
max_error = (actuals - predictions).abs().max()
mae = MAE()(actuals,predictions)
mape = MAPE()(actuals,predictions)
smape = SMAPE()(actuals,predictions)

print(f"Max_error of model: {max_error}")
print(f"Mean absolute error of model: {mae}")
print(f"Mean absolute percentage error of model: {mape}")
print(f"Symmetric Mean absolute percentage error: {smape}")

# 预测结果可视化
actuals = actuals.numpy().reshape(-1,)
predictions = predictions.numpy().reshape(-1,)
pd.DataFrame({
    "data_test": actuals,
    "predict_data": predictions
}).plot(legend=True, figsize=(10, 6))
plt.tick_params()
plt.xticks()
plt.yticks()
plt.title("Deepar: MAPE={}".format(mape))
plt.savefig("./result/Deepar.png")
plt.show()

# 将评价指标写入txt文件
with open('./result/arima.txt', "w+") as f:
    print("Max_Error = {}".format(max_error), file=f)
    print("MAE = {}".format(mae), file=f)
    print("MAPE = {}".format(mape), file=f)
    print("SMAPE = {}".format(SMAPE(smape)), file=f)

