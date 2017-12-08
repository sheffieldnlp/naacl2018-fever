import sys

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from common.dataset.data_set import DataSet
from common.dataset.reader import JSONLineReader
from common.features.feature_function import Features
from common.training.early_stopping import EarlyStopping
from common.training.options import gpu
from common.training.run import train, predict
from retrieval.fever_doc_db import FeverDocDB
from rte.riedel.data import FEVERGoldFormatter, FEVERLabelSchema, FEVERPredictionsFormatter
from rte.riedel.fever_features import TermFrequencyFeatureFunction
from rte.riedel.model import SimpleMLP

if __name__ == "__main__":
    maxdoc = sys.argv[1]
    db = FeverDocDB("data/fever/drqa.db")
    idx = set(db.get_doc_ids())

    f = Features([TermFrequencyFeatureFunction(db,naming="pred3wdrqa-p{0}".format(maxdoc))])

    jlr = JSONLineReader()

    gold_formatter = FEVERGoldFormatter(idx, FEVERLabelSchema())
    formatter = FEVERPredictionsFormatter(idx, FEVERLabelSchema())


    train_ds = DataSet(file="data/fever/train.ns.pages.p{0}.jsonl".format(maxdoc), reader=jlr, formatter=gold_formatter)
    dev_ds = DataSet(file="data/fever/dev.pages.p{0}.jsonl".format(maxdoc), reader=jlr, formatter=formatter)
    test_ds = DataSet(file="data/fever/test.pages.p{0}.jsonl".format(maxdoc), reader=jlr, formatter=formatter)

    train_ds.read()
    dev_ds.read()
    test_ds.read()

    train_feats, dev_feats, test_feats = f.load(train_ds, dev_ds, None)

    input_shape = train_feats[0].shape[1]

    model = SimpleMLP(input_shape,100,3)

    if gpu():
        model.cuda()

    final_model = train(model, train_feats, 500, 1e-2, 90,dev_feats,5,early_stopping=EarlyStopping())

    test_data, actual = test_ds.data

    predictions = predict(final_model, test_data, 500)

    ls = FEVERLabelSchema()

    labels = [ls.labels[i] for i, _ in enumerate(ls.labels)]
    print(accuracy_score(actual, predictions))
    print(classification_report(actual, predictions, labels=labels))
    print(confusion_matrix(actual, predictions, labels=labels))