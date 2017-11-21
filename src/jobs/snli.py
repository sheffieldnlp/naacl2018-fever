import os

from dataset.data_set import DataSet
from dataset.formatter import SNLIFormatter
from dataset.label_schema import SNLILabelSchema
from dataset.reader import JSONLineReader

train_file = os.path.join("data","snli","snli_1.0" ,"snli_1.0_train.jsonl")
jlr = JSONLineReader()
formatter = SNLIFormatter(SNLILabelSchema())

train = DataSet(file=train_file,reader=jlr,formatter=formatter)
train.read()

print(train.data)