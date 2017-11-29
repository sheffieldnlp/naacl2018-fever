from common.dataset.data_set import DataSet
from common.dataset.formatter import Formatter
from common.dataset.label_schema import LabelSchema
from common.dataset.reader import JSONLineReader
from common.features.feature_function import Features
from retrieval.fever_doc_db import FeverDocDB
from rte.riedel.features import TermFrequencyFeatureFunction
from common.training.batcher import Batcher
import torch
from torch import nn,autograd,rand
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm

def preprocess(p):
    return p.replace(" ","_").replace("(","-LRB-").replace(")","-RRB-").replace(":","-COLON-").split("#")[0]


def prepare2(data,labels):
    data = data.todense()
    v = torch.FloatTensor(np.array(data))
    return Variable(v), Variable(torch.LongTensor(labels))


class FEVERFormatter(Formatter):

    def __init__(self,index,label_schema):
        super().__init__(label_schema)
        self.index = index
    def format_line(self,line):
        annotation = line["verdict"]

        if not isinstance(line['evidence'][0],list):
            return None

        pages = [preprocess(ev[1]) for ev in line["evidence"]]

        if any(map(lambda p: p not in self.index, pages)):
            return None

        return {"claim":line["claim"], "evidence": pages, "label":self.label_schema.get_id(annotation)}



class FEVERLabelSchema(LabelSchema):
    def __init__(self):
        super().__init__(["supported","refuted","not enough information"])


class SimpleMLP(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,dropout_p=.6):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,output_dim)

        self.do = nn.Dropout(dropout_p)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.do(x)
        x = self.fc1(x)

        x = self.relu(x)

        x = self.do(x)
        x = self.fc2(x)

        return x


def train(model, fs, batch_size, lr, epochs):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    steps = 0

    data, labels = fs

    batcher = Batcher(data, batch_size)

    for epoch in tqdm(range(epochs)):
        for batch, size, start, end in batcher:
            d,gold = prepare2(batch,labels[start:end])

            optimizer.zero_grad()
            logits = model(d)

            loss = F.cross_entropy(logits, gold)

            loss.backward()
            optimizer.step()



if __name__ == "__main__":
    db = FeverDocDB("data/fever/drqa.db")
    idx = set(db.get_doc_ids())

    f = Features([TermFrequencyFeatureFunction(db)])
    jlr = JSONLineReader()

    formatter = FEVERFormatter(idx, FEVERLabelSchema())

    train_ds = DataSet(file="data/fever/fever.train.jsonl", reader=jlr, formatter=formatter)
    dev_ds = DataSet(file="data/fever/fever.dev.jsonl", reader=jlr, formatter=formatter)
    test_ds = DataSet(file="data/fever/fever.test.jsonl", reader=jlr, formatter=formatter)

    train_ds.read()
    dev_ds.read()
    test_ds.read()

    train_feats, dev_feats, test_feats = f.load(train_ds, dev_ds, test_ds)

    input_shape = train_feats[0].shape[1]

    model = SimpleMLP(input_shape,100,2)
    train(model, train_feats, 500, 1e-2, 90)


