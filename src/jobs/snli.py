import os

from dataset.data_set import DataSet
from dataset.formatter import SNLIFormatter
from dataset.label_schema import SNLILabelSchema
from dataset.reader import JSONLineReader
from features.feature_function import BleuOverlapFeatureFuntion, Features

train_file = os.path.join("data","snli","snli_1.0" ,"snli_1.0_train.jsonl")
jlr = JSONLineReader()
formatter = SNLIFormatter(SNLILabelSchema())

train = DataSet(file=train_file,reader=jlr,formatter=formatter)
train.read()

print(train.data)

features = Features([BleuOverlapFeatureFuntion()])

features.generate_vocab(train)
fs = features.load(train)

print(list(fs[0]))