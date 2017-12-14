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
from rte.riedel.model import SimpleMLP
from rte.riedel.sent_features import SentenceLevelTermFrequencyFeatureFunction


def model_exists(mname):
    return os.path.exists(os.path.join("models","{0}.model".format(mname)))

if __name__ == "__main__":

    LogHelper.setup()
    logger = LogHelper.get_logger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument('db', type=str, help='db file path')
    parser.add_argument('train', type=str, help='train file path')
    parser.add_argument('dev', type=str, help='dev file path')
    parser.add_argument('--test', required=False ,type=str, default=None ,help="test file path")
    parser.add_argument("--model", type=str, help="model name")

    args = parser.parse_args()


    logger.info("Loading DB {0}".format(args.db))
    db = FeverDocDB(args.db)
    idx = set(db.get_doc_ids())
    logger.info("{0} documents loaded".format(len(idx)))

    mname = args.model
    logger.info("Model name is {0}".format(mname))

    f = Features([SentenceLevelTermFrequencyFeatureFunction(db,naming=mname)])
    jlr = JSONLineReader()

    formatter = FEVERGoldFormatter(idx, FEVERLabelSchema())

    train_ds = DataSet(file=args.train, reader=jlr, formatter=formatter)
    dev_ds = DataSet(file=args.dev, reader=jlr, formatter=formatter)

    train_ds.read()
    dev_ds.read()

    test_ds = None
    if args.test is not None:
        test_ds = DataSet(file=args.test, reader=jlr, formatter=formatter)
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

    if args.test is not None:
        print_evaluation(final_model, test_feats, FEVERLabelSchema())