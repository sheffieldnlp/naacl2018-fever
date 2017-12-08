import sys

import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from common.dataset.data_set import DataSet
from common.dataset.reader import JSONLineReader
from common.features.feature_function import Features
from common.training.early_stopping import EarlyStopping
from common.training.options import gpu
from common.training.run import train, predict, print_evaluation
from common.util.random import SimpleRandom
from retrieval.fever_doc_db import FeverDocDB
from rte.riedel.data import FEVERGoldFormatter, FEVERLabelSchema, FEVERPredictionsFormatter
from rte.riedel.fever_features import TermFrequencyFeatureFunction
from rte.riedel.model import SimpleMLP
import os

def model_exists(mname):
    return os.path.exists(os.path.join("models","{0}.model".format(mname)))

if __name__ == "__main__":
    SimpleRandom.set_seeds()

    ns_docsize = sys.argv[2]

    db = FeverDocDB("data/fever/drqa.db")
    idx = set(db.get_doc_ids())

    mname = "oraclepred3wdrqa-p{0}".format(ns_docsize)

    f = Features([TermFrequencyFeatureFunction(db,naming=mname)])

    jlr = JSONLineReader()

    gold_formatter = FEVERGoldFormatter(idx, FEVERLabelSchema())
    formatter = FEVERPredictionsFormatter(idx, FEVERLabelSchema())


    train_ds = DataSet(file="data/fever/train.ns.pages.p{0}.jsonl".format(ns_docsize), reader=jlr, formatter=gold_formatter)
    dev_ds = DataSet(file="data/fever/dev.ns.pages.p{0}.jsonl".format(ns_docsize), reader=jlr, formatter=gold_formatter)
    test_ds = DataSet(file="data/fever/test.ns.pages.p{0}.jsonl".format(ns_docsize), reader=jlr, formatter=gold_formatter)

    train_ds.read()
    dev_ds.read()
    test_ds.read()

    train_feats, dev_feats, test_feats = f.load(train_ds, dev_ds, test_ds)
    input_shape = train_feats[0].shape[1]

    model = SimpleMLP(input_shape,100,3)

    if gpu():
        model.cuda()


    if model_exists(mname):
        model.load_state_dict(torch.load("models/{0}.model".format(mname)))
        final_model = model
    else:
        final_model = train(model, train_feats, 500, 1e-2, 90,dev_feats,early_stopping=EarlyStopping())
        torch.save(model.state_dict(), "models/{0}.model".format(mname))


    print_evaluation(final_model, dev_feats, FEVERLabelSchema())
    print_evaluation(final_model, test_feats, FEVERLabelSchema())

