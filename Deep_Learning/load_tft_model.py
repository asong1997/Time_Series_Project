"""
使用时间融合变压器进行需求预测
在本教程中，我们将在一个非常小的数据集上训练 TemporalFusionTransformer，
以证明它甚至只在 20k 个样本上做得很好。一般来说，它是一个大型模型，因此使用更多数据将表现得更好。
我们的例子是来自Stallion kaggle竞争的需求预测。
"""
import warnings
warnings.filterwarnings("ignore")  # avoid printing out absolute paths
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss, MAPE, MAE


"""
加载数据
首先，我们需要将时间序列转换为 pandas 数据帧，其中每行都可以用时间步长和时间序列来标识。
幸运的是，大多数数据集已经采用这种格式。在本教程中，我们将使用Kaggle的Stallion数据集来描述各种饮料的销售情况。
我们的任务是按库存单位（SKU）对销售量进行为期六个月的预测，即由代理商销售的产品，即商店。
每月约有21 000条历史销售记录。除了历史销售外，我们还有关于销售价格，代理商位置，特殊日子（如节假日）以及整个行业销售量的信息。
数据集的格式已经正确，但缺少一些重要功能。最重要的是，我们需要添加一个时间索引，该索引在每个时间步长递增 
1。此外，添加日期要素是有益的，在这种情况下，这意味着从日期记录中提取月份。
"""
# data = get_stallion_data()
data = pd.read_csv("stallion_data.csv")
data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d")
# add time index
data["time_idx"] = data["date"].dt.year * 12 + data["date"].dt.month
data["time_idx"] -= data["time_idx"].min()
# add additional features
data["month"] = data.date.dt.month.astype(str).astype("category")  # categories have be strings
data["log_volume"] = np.log(data.volume + 1e-8)
data["avg_volume_by_sku"] = data.groupby(["time_idx", "sku"], observed=True).volume.transform("mean")
data["avg_volume_by_agency"] = data.groupby(["time_idx", "agency"], observed=True).volume.transform("mean")

# 我们想将特special days编码为一个变量，因此需要首先反转 one-hot 编码
special_days = [
    "easter_day",
    "good_friday",
    "new_year",
    "christmas",
    "labor_day",
    "independence_day",
    "revolution_day_memorial",
    "regional_games",
    "fifa_u_17_world_cup",
    "football_gold_cup",
    "beer_capital",
    "music_fest",
]
data[special_days] = data[special_days].apply(lambda x: x.map({0: "-", 1: x.name})).astype("category")
print(data.sample(10, random_state=521))

"""
创建数据集和数据加载器
下一步是将数据帧转换为PyTorch Forecasting TimeSeriesDataSet。除了告诉数据集哪些特征是分类的还是连续的，
哪些是静态的和在时间上变化的，我们还必须决定如何规范化数据。在这里，我们分别对每个时间序列进行标准刻度，
并指示值始终为正数。通常，编码器规范化程序（在训练时在每个编码器序列上动态缩放）是首选，以避免归一化引起的前瞻偏差。
但是，如果您在寻找合理稳定的规范化方面遇到困难，例如，您可能会接受前瞻偏差，因为数据中有很多零。
或者，您期望在推理中实现更稳定的规范化。在后面的例子中，你确保你不会学习在运行推理时不会出现的“奇怪”跳跃，
从而在更真实的数据集上进行训练。我们还选择使用过去六个月的验证集。
"""
max_prediction_length = 6
max_encoder_length = 24
training_cutoff = data["time_idx"].max() - max_prediction_length

training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="volume",
    group_ids=["agency", "sku"],
    min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["agency", "sku"],
    static_reals=["avg_population_2017", "avg_yearly_household_income_2017"],
    time_varying_known_categoricals=["special_days", "month"],
    variable_groups={"special_days": special_days},  # group of categorical variables can be treated as one variable
    time_varying_known_reals=["time_idx", "price_regular", "discount_in_percent"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=[
        "volume",
        "log_volume",
        "industry_volume",
        "soda_volume",
        "avg_max_temp",
        "avg_volume_by_agency",
        "avg_volume_by_sku",
    ],
    target_normalizer=GroupNormalizer(
        groups=["agency", "sku"], transformation="softplus"
    ),  # use softplus and normalize by group
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# 创建验证集 (predict=True) 表示预测最后一个 max_prediction_length 时间点
validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)

# 为模型创建数据加载器
batch_size = 128  # set this between 32 to 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

# 加载模型
best_tft = TemporalFusionTransformer.load_from_checkpoint('lightning_logs\\lightning_logs\\version_5\\checkpoints\\epoch=22-step=690.ckpt') # 'lightning_logs\\lightning_logs\\version_5\\checkpoints\\epoch=22-step=690.ckpt'
"""
训练后，我们可以使用predict（）进行预测。该方法允许对它返回的内容进行非常细粒度的控制，
例如，您可以轻松地将预测与 pandas 数据帧进行匹配。有关详细信息，请参阅其文档。
我们评估验证数据集上的指标和几个示例，以了解模型的性能。鉴于我们只处理21 000个样本，
结果非常令人放心，可以通过梯度助推器与结果竞争。我们的表现也比基线模型更好。鉴于数据嘈杂，这不是微不足道的。
"""
# calcualte mean absolute error on validation set
actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
predictions = best_tft.predict(val_dataloader)
# (actuals - predictions).abs().mean() = tensor(268.2687)
print("baseline_MAE:", MAE()(actuals, predictions))
print("baseline_MAX_Error:", (actuals - predictions).abs().max())
print("baseline_SMAPE:", SMAPE()(actuals, predictions))
# 原始预测是一个字典，可以从中提取包括分位数在内的所有类型的信息
raw_predictions, x = best_tft.predict(val_dataloader, mode="raw", return_x=True)
for idx in range(10):  # plot 10 examples
    best_tft.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True)

"""
表现最差的
查看表现最差的模型（例如，在 SMAPE 方面），可以让我们了解模型在可靠预测方面存在问题。
这些示例可以提供有关如何改进模型的重要指针。这种实际值与预测图适用于所有模型。
当然，使用在指标模块中定义的其他指标（如 MASE）也是明智的。但是，为了便于演示，我们在此处仅使用 SMAPE。
"""
# 计算要显示的指标
predictions = best_tft.predict(val_dataloader)
mean_losses = SMAPE(reduction="none")(predictions, actuals).mean(1)
indices = mean_losses.argsort(descending=True)  # sort losses
for idx in range(10):  # plot 10 examples
    best_tft.plot_prediction(
        x, raw_predictions, idx=indices[idx], add_loss_to_title=SMAPE(quantiles=best_tft.loss.quantiles)
    )

"""
实际值与变量预测
检查模型在不同数据切片中的表现使我们能够检测到弱点。下面绘制的是使用 Now 划分为 100 个条柱的每个变量的预测值与实际值的平均值，
我们可以使用 calculate_prediction_actual_by_variable（） 和 plot_prediction_actual_by_variable（） 
方法直接预测生成的数据。灰色条表示变量的频率，即是直方图。
"""
predictions, x = best_tft.predict(val_dataloader, return_x=True)
predictions_vs_actuals = best_tft.calculate_prediction_actual_by_variable(x, predictions)
best_tft.plot_prediction_actual_by_variable(predictions_vs_actuals)



"""
对所选数据进行预测
为了预测数据子集，我们可以使用 filter（） 方法过滤数据集中的子序列。在这里，
我们预测数据集中映射到组 id 的子序列“Agency_01”和“SKU_01”，其第一个预测值对应于时间索引“15”。
我们输出所有七个分位数。这意味着我们期望一个形状的张量，因为我们预测单个子序列提前六个时间步，
每个时间步长7个分位数。training1 x n_timesteps x n_quantiles = 1 x 6 x 7
"""
best_tft.predict(
    training.filter(lambda x: (x.agency == "Agency_01") & (x.sku == "SKU_01") & (x.time_idx_first_prediction == 15)),
    mode="quantiles",
)
# 当然，我们也可以很容易地绘制这个预测：
raw_prediction, x = best_tft.predict(
    training.filter(lambda x: (x.agency == "Agency_01") & (x.sku == "SKU_01") & (x.time_idx_first_prediction == 15)),
    mode="raw",
    return_x=True,
)
best_tft.plot_prediction(x, raw_prediction, idx=0)

"""
预测新数据
由于数据集中有协变量，因此对新数据的预测需要我们预先定义已知的协变量。
"""
# select last 24 months from data (max_encoder_length is 24)
encoder_data = data[lambda x: x.time_idx > x.time_idx.max() - max_encoder_length]

# 选择最后一个已知数据点并通过重复它并增加月份来从中创建解码器数据
# 在现实世界的数据集中，我们不应该只是向前填充协变量，而是指定它们来解释
# 特殊日子和价格的变化（你绝对应该这样做，但我们在这里太懒了）
last_data = data[lambda x: x.time_idx == x.time_idx.max()]
decoder_data = pd.concat(
    [last_data.assign(date=lambda x: x.date + pd.offsets.MonthBegin(i)) for i in range(1, max_prediction_length + 1)],
    ignore_index=True,
)

# 添加与“数据”一致的时间索引
decoder_data["time_idx"] = decoder_data["date"].dt.year * 12 + decoder_data["date"].dt.month
decoder_data["time_idx"] += encoder_data["time_idx"].max() + 1 - decoder_data["time_idx"].min()

# 调整额外的时间特征
decoder_data["month"] = decoder_data.date.dt.month.astype(str).astype("category")  # categories have be strings

# 合并编码器和解码器数据
new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)

# 现在，我们可以使用 predict（） 方法直接预测生成的数据。
new_raw_predictions, new_x = best_tft.predict(new_prediction_data, mode="raw", return_x=True)
for idx in range(10):  # plot 10 examples
    best_tft.plot_prediction(new_x, new_raw_predictions, idx=idx, show_future_observed=False)


"""
解释模型
可变重要性
该模型由于其架构的构建方式而具有内置的解释功能。让我们看看它看起来如何。
我们首先用interpret_output（）计算解释，然后用plot_interpretation（）绘制它们。
"""
interpretation = best_tft.interpret_output(raw_predictions, reduction="sum")
best_tft.plot_interpretation(interpretation)

"""
不出所料，过去观察到的作为编码器中顶部变量的交易量特征和价格相关变量是解码器中的主要预测因子之一。
一般的注意力模式似乎是，最近的观察更重要，而旧的观察。这证实了直觉。平均注意力通常不是很有用 
- 通过示例查看注意力更有见地，因为模式没有平均化。
"""


"""
部分依赖关系
偏依赖关系图通常用于更好地解释模型（假设特征具有独立性）。它们也可用于了解在模拟情况下的期望，
并且是使用predict_dependency（）创建的。
"""
dependency = best_tft.predict_dependency(
    val_dataloader.dataset, "discount_in_percent", np.linspace(0, 30, 30), show_progress_bar=True, mode="dataframe"
)
# 绘制中位数和 25% 和 75% 百分位数
agg_dependency = dependency.groupby("discount_in_percent").normalized_prediction.agg(
    median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75)
)
ax = agg_dependency.plot(y="median")
ax.fill_between(agg_dependency.index, agg_dependency.q25, agg_dependency.q75, alpha=0.3)
print("end.........................")

