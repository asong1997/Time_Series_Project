"""
快速入门教程
GluonTS 工具包包含用于使用 MXNet 构建时序模型的组件和工具。当前包含的模型是预测模型，但这些组件还支持其他时序用例，例如分类或异常检测。
该工具包并非旨在作为企业或最终用户的预测解决方案，而是针对想要调整算法或构建和试验自己的模型的科学家和工程师。
GluonTS包含：
    用于构建新模型的组件（似然、特征处理管道、日历特征等）
    数据加载和处理
    许多预建模型
    绘图和评估设施
    人工和真实数据集（仅限具有祝福许可证的外部数据集）
"""

import mxnet as mx
from mxnet import gluon
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

# # 提供的数据集
# # GluonTS带有许多公开可用的数据集
#
# from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
# from gluonts.dataset.util import to_pandas
# print(f"Available datasets: {list(dataset_recipes.keys())}")
# # 要下载其中一个内置数据集，只需使用上述名称之一调用get_dataset即可。
# # GluonTS可以重用保存的数据集，这样它就不需要再次下载：只需设置。regenerate=False
# dataset = get_dataset("m4_hourly", regenerate=True)
# """
# 通常，GluonTS提供的数据集是由三个主要成员组成的对象：
# dataset.train是用于训练的数据条目的可迭代集合。每个条目对应一个时间序列
# dataset.test是用于推理的数据条目的可迭代集合。测试数据集是训练数据集的扩展版本，其中包含在训练期间未看到的每个时间序列末尾的窗口。此窗口的长度等于建议的预测长度。
# dataset.metadata包含数据集的元数据，例如时间序列的频率、建议的预测范围、相关要素等。
# """
# entry = next(iter(dataset.train))
# train_series = to_pandas(entry)
# train_series.plot()
# plt.grid(which="both")
# plt.legend(["train series"], loc="upper left")
# plt.show()
#
# entry = next(iter(dataset.test))
# test_series = to_pandas(entry)
# test_series.plot()
# plt.axvline(train_series.index[-1], color='r') # end of train dataset
# plt.grid(which="both")
# plt.legend(["test series", "end of train series"], loc="upper left")
# plt.show()
#
# print(f"Length of forecasting window in test dataset: {len(test_series) - len(train_series)}")
# print(f"Recommended prediction horizon: {dataset.metadata.prediction_length}")
# print(f"Frequency of the time series: {dataset.metadata.freq}")

"""
自定义数据集
在这一点上，重要的是要强调GluonTS不需要用户可能拥有的自定义数据集的这种特定格式。
自定义数据集的唯一要求是可迭代并具有“目标”和“开始”字段。为了更清楚地说明这一点，
假设数据集采用 a 的形式，而时间序列的索引为 a（每个时间序列可能不同）：numpy.arraypandas.Timestamp
"""
N = 10  # number of time series
T = 100  # number of timesteps
prediction_length = 24
freq = "1H"
custom_dataset = np.random.normal(size=(N, T))
start = pd.Timestamp("01-01-2019", freq=freq)  # can be different for each time series

# 现在，您可以拆分数据集，只需两行代码即可将其以 GluonTS 适当的格式引入：
from gluonts.dataset.common import ListDataset

# 训练数据集：剪切长度为“prediction_length”的最后一个窗口，添加“target”和“start”字段
train_ds = ListDataset(
    [{'target': x, 'start': start} for x in custom_dataset[:, :-prediction_length]],
    freq=freq
)
# 测试数据集：使用整个数据集，添加“target”和“start”字段
test_ds = ListDataset(
    [{'target': x, 'start': start} for x in custom_dataset],
    freq=freq
)


"""
训练现有模型 （Estimator)
GluonTS带有许多预建模型。用户需要做的就是配置一些超参数。现有模型侧重于（但不限于）概率预测。概率预测是概率分布形式的预测，而不仅仅是单点估计。
我们将从GulonTS预先构建的前馈神经网络估计器开始，这是一个简单但功能强大的预测模型。我们将使用此模型来演示训练模型、生成预测和评估结果的过程。
GluonTS的内置前馈神经网络（）接受长度的输入窗口，并预测后续值的值的分布。在GluonTS的说法中，前馈神经网络模型就是一个例子。
在GluonTS中，对象表示预测模型以及其系数，权重等详细信息。
SimpleFeedForwardEstimatorcontext_lengthprediction_lengthEstimatorEstimator

通常，每个估计器（预构建或自定义）由许多超参数配置，这些超参数可以是所有估计器（例如，）中通用的（但不是绑定的），
也可以是特定于特定估计器的（例如，神经网络的层数或CNN中的步幅）。prediction_length
最后，每个估计器都由 a 配置，它定义了模型的训练方式，即 epoch 数、学习速率等。Trainer
"""

from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.mx import Trainer


estimator = SimpleFeedForwardEstimator(
    num_hidden_dimensions=[10],
    prediction_length=prediction_length,
    context_length=100,
    freq=freq,
    trainer=Trainer(
        ctx="cpu",
        epochs=5,
        learning_rate=1e-3,
        num_batches_per_epoch=100
    )
)


"""
在使用所有必要的超参数指定估计器后，我们可以通过调用估计器的方法使用我们的训练数据集对其进行训练。
训练算法返回一个拟合模型（或GluonTS术语中的一个），可用于构建预测。dataset.traintrainPredictor
"""
predictor = estimator.train(train_ds)


"""
可视化和评估预测
有了预测变量，我们现在可以预测模型的最后一个窗口并评估模型的性能。dataset.test
GluonTS具有自动执行预测和模型评估过程的功能。粗略地说，此函数执行以下步骤：make_evaluation_predictions
删除我们要预测的长度的最终窗口prediction_lengthdataset.test
估计器使用剩余数据来预测（以样本路径的形式）刚刚删除的“未来”窗口
该模块输出预测样本路径和（作为 python 生成器对象）dataset.test
"""

from gluonts.evaluation import make_evaluation_predictions
forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_ds,  # test dataset
    predictor=predictor,  # predictor
    num_samples=100,  # number of sample paths we want for evaluation
)
# 首先，我们可以将这些生成器转换为列表，以简化后续计算。
forecasts = list(forecast_it)
tss = list(ts_it)


# 我们可以检查这些列表的第一个元素（对应于数据集的第一个时间序列）。让我们从包含时间序列的列表开始，
# 即 .我们期望 的第一个条目包含 的第一个时间序列 的 （目标）。tsstssdataset.test

