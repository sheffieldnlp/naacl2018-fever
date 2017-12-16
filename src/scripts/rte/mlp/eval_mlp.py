import argparse

import torch
import os

from common.dataset.data_set import DataSet
from common.dataset.reader import JSONLineReader
from common.features.feature_function import Features
from common.training.early_stopping import EarlyStopping
from common.training.options import gpu
from common.training.run import train, print_evaluation
from common.util.log_helper import LogHelper
from retrieval.fever_doc_db import FeverDocDB
from rte.riedel.data import FEVERGoldFormatter, FEVERLabelSchema
from rte.riedel.fever_features import TermFrequencyFeatureFunction
from rte.riedel.model import SimpleMLP
from rte.riedel.sent_features import SentenceLevelTermFrequencyFeatureFunction


def model_exists(mname):
    return os.path.exists(os.path.join("models","{0}.model".format(mname)))

if __name__ == "__main__":

    LogHelper.setup()
    logger = LogHelper.get_logger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument('db', type=str, help='db file path')
    parser.add_argument('test', type=str, help='test file path')
    parser.add_argument("--model", type=str, help="model name")
    parser.add_argument("--sentence",type=bool, default=False)
    parser.add_argument("--log",type=str,default=None)
    args = parser.parse_args()


    logger.info("Loading DB {0}".format(args.db))
    db = FeverDocDB(args.db)
    idx = set(db.get_doc_ids())
    logger.info("{0} documents loaded".format(len(idx)))

    mname = args.model
    logger.info("Model name is {0}".format(mname))


    ffns = []

    if args.sentence:
        print("Sentence level")
        ffns.append(SentenceLevelTermFrequencyFeatureFunction(db, naming=mname))
    else:
        print("Doc level")
        ffns.append(TermFrequencyFeatureFunction(db,naming=mname))


    f = Features(ffns)
    f.load_vocab(mname)

    jlr = JSONLineReader()
    formatter = FEVERGoldFormatter(idx, FEVERLabelSchema())

    test_ds = DataSet(file=args.test, reader=jlr, formatter=formatter)
    test_ds.read()
    feats = f.lookup(test_ds)

    input_shape = feats[0].shape[1]
    model = SimpleMLP(input_shape,100,3)

    if gpu():
        model.cuda()

    model.load_state_dict(torch.load("models/{0}.model".format(mname)))
    print_evaluation(model, feats, FEVERLabelSchema(),args.log)
