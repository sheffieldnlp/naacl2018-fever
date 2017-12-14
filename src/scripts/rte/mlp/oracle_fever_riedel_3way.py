import torch
import os

from common.dataset.data_set import DataSet
from common.dataset.reader import JSONLineReader
from common.features.feature_function import Features
from common.training.early_stopping import EarlyStopping
from common.training.options import gpu
from common.training.run import train, print_evaluation
from retrieval.fever_doc_db import FeverDocDB
from rte.riedel.data import FEVERGoldFormatter, FEVERLabelSchema
from rte.riedel.fever_features import TermFrequencyFeatureFunction
from rte.riedel.model import SimpleMLP

def model_exists(mname):
    return os.path.exists(os.path.join("models","{0}.model".format(mname)))

if __name__ == "__main__":
    db = FeverDocDB("data/fever/drqa.db")
    idx = set(db.get_doc_ids())

    mname = "3way_rand_ns_oracle"
    f = Features([TermFrequencyFeatureFunction(db,naming=mname)])
    jlr = JSONLineReader()

    formatter = FEVERGoldFormatter(idx, FEVERLabelSchema())

    train_ds = DataSet(file="data/fever/train.ns.rand.jsonl", reader=jlr, formatter=formatter)
    dev_ds = DataSet(file="data/fever/dev.ns.rand.jsonl", reader=jlr, formatter=formatter)
    test_ds = DataSet(file="data/fever/test.ns.rand.jsonl", reader=jlr, formatter=formatter)

    train_ds.read()
    dev_ds.read()
    test_ds.read()

    train_feats, dev_feats, test_feats = f.load(train_ds, dev_ds, test_ds)

    input_shape = train_feats[0].shape[1]

    model = SimpleMLP(input_shape,100,3)

    if gpu():
        model.cuda()


    if model_exists(mname) and os.getenv("TRAIN").lower() not in ["y","1","t","yes"]:
        model.load_state_dict(torch.load("models/{0}.model".format(mname)))
        final_model = model
    else:
        final_model = train(model, train_feats, 500, 1e-2, 90,dev_feats,early_stopping=EarlyStopping())
        torch.save(model.state_dict(), "models/{0}.model".format(mname))



    print_evaluation(final_model, dev_feats, FEVERLabelSchema())
    print_evaluation(final_model, test_feats, FEVERLabelSchema())