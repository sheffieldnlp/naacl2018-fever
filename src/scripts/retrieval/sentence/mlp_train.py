import sys

import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from common.dataset.data_set import DataSet
from common.dataset.label_schema import LabelSchema
from common.dataset.reader import JSONLineReader
from common.features.feature_function import Features
from common.training.early_stopping import EarlyStopping
from common.training.options import gpu
from common.training.run import train, predict, print_evaluation
from common.util.random import SimpleRandom
from retrieval.fever_doc_db import FeverDocDB
from retrieval.sent_features import SentenceTermFrequencyFeatureFunction
from retrieval.sentence import FEVERSentenceRelatednessFormatter, FEVERSentenceFormatter
from rte.riedel.data import FEVERGoldFormatter, FEVERLabelSchema, FEVERPredictionsFormatter, FeverFormatter, preprocess
from rte.riedel.fever_features import TermFrequencyFeatureFunction
from rte.riedel.model import SimpleMLP
import os

def model_exists(mname):
    return os.path.exists(os.path.join("models","{0}.model".format(mname)))


class RelatedLabelSchema(LabelSchema):
    def __init__(self):
        super().__init__(["related","unrelated"])




if __name__ == "__main__":
    SimpleRandom.set_seeds()

    maxdoc = sys.argv[1]
    ns_docsize = sys.argv[2]

    db = FeverDocDB("data/fever/fever.db")
    idx = set(db.get_doc_ids())

    mname = "2way-p{0}-{1}".format(maxdoc,ns_docsize)

    f = Features([SentenceTermFrequencyFeatureFunction(db,naming=mname)])
    jlr = JSONLineReader()

    formatter = FEVERSentenceFormatter(idx, db, RelatedLabelSchema())


    train_ds = DataSet(file="data/fever/train.ns.pages.p{0}.jsonl".format(ns_docsize), reader=jlr, formatter=formatter)
    dev_ds = DataSet(file="data/fever/dev.pages.p{0}.jsonl".format(maxdoc), reader=jlr, formatter=formatter)
    test_ds = DataSet(file="data/fever/test.pages.p{0}.jsonl".format(maxdoc), reader=jlr, formatter=formatter)

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

