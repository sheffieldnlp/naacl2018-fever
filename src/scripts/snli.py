import os

from shared.features.feature_function import BleuOverlapFeatureFuntion, Features

from common.dataset import DataSet
from common.dataset import JSONLineReader
from common.dataset import SNLIFormatter
from common.dataset import SNLILabelSchema

train_file = os.path.join("data","snli","snli_1.0" ,"snli_1.0_train.jsonl")
jlr = JSONLineReader()
formatter = SNLIFormatter(SNLILabelSchema())

train = DataSet(file=train_file,reader=jlr,formatter=formatter)
train.read()

print(train.data)

features = Features([BleuOverlapFeatureFuntion()])

features.generate_vocab(train)
fs = features.load(train)

print(fs[0])