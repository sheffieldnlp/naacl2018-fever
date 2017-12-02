from common.dataset.data_set import DataSet
from common.dataset.reader import JSONLineReader
from common.features.feature_function import Features
from common.training.options import gpu
from common.training.run import train
from retrieval.fever_doc_db import FeverDocDB
from rte.riedel.data import FEVERGoldFormatter, FEVERLabelSchema
from rte.riedel.fever_features import TermFrequencyFeatureFunction
from rte.riedel.model import SimpleMLP

if __name__ == "__main__":
    db = FeverDocDB("data/fever/drqa.db")
    idx = set(db.get_doc_ids())

    f = Features([TermFrequencyFeatureFunction(db)])
    jlr = JSONLineReader()

    formatter = FEVERGoldFormatter(idx, FEVERLabelSchema())

    train_ds = DataSet(file="data/fever/fever.train.ns.rand.jsonl", reader=jlr, formatter=formatter)
    dev_ds = DataSet(file="data/fever/fever.dev.ns.rand.jsonl", reader=jlr, formatter=formatter)
    test_ds = DataSet(file="data/fever/fever.test.ns.rand.jsonl", reader=jlr, formatter=formatter)

    train_ds.read()
    dev_ds.read()
    test_ds.read()

    train_feats, dev_feats, test_feats = f.load(train_ds, dev_ds, test_ds)

    input_shape = train_feats[0].shape[1]

    model = SimpleMLP(input_shape,100,3)

    if gpu():
        model.cuda()

    train(model, train_feats, 500, 1e-2, 90,dev_feats)

