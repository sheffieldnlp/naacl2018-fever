from common.dataset.data_set import DataSet
from common.dataset.formatter import Formatter
from common.dataset.label_schema import LabelSchema
from common.dataset.reader import CSVReader, JSONLineReader
from common.features.feature_function import Features
from common.training.early_stopping import EarlyStopping
from common.training.options import gpu
from common.training.run import train, evaluate
from rte.riedel.fnc_features import FNCTermFrequencyFeatureFunction
from rte.riedel.fnc_fever_transfer_features import FeverOrFNCTermFrequencyFeatureFunction
from rte.riedel.model import SimpleMLP
from rte.riedel.data import FEVERGoldFormatter, FEVERLabelSchema, FEVERPredictionsFormatter
from retrieval.fever_doc_db import FeverDocDB
class Bodies():
    def __init__(self,*paths):
        self.bodies = {}

        for path in paths:
            csvr = CSVReader()
            data = csvr.read(path)

            for body in data:
                self.bodies[body["Body ID"]] = body["articleBody"]

    def get_doc_text(self, doc_id):
        return self.bodies[doc_id]

class FNCSimpleLabelSchema(LabelSchema):
    def __init__(self):
        super().__init__(["agree","disagree","not enough info"])


class FNCFormatter2(Formatter):
    def __init__(self,label_schema):
        super().__init__(label_schema)

    def format_line(self,line):
        annotation = self.label_schema.get_id(line["Stance"]) if "Stance" in line else None
        if annotation is None:
            annotation = self.label_schema.get_id("not enough info")

        return {"claim":line["Headline"], "evidence": line["Body ID"], "label":annotation}


if __name__ == "__main__":

    db = FeverDocDB("data/fever/drqa.db")
    idx = set(db.get_doc_ids())

    fnc_bodies = Bodies("data/fnc-1/train_bodies.csv","data/fnc-1/competition_test_bodies.csv")
    fever_bodies = db

    f = Features([FeverOrFNCTermFrequencyFeatureFunction(fever_bodies,fnc_bodies)])
    csvr = CSVReader()
    jlr = JSONLineReader()
    fnc_formatter = FNCFormatter2(FNCSimpleLabelSchema())
    fever_formatter = FEVERPredictionsFormatter(idx,FEVERLabelSchema())


    train_ds = DataSet(file="data/fnc-1/train_stances.csv", reader=csvr, formatter=fnc_formatter)
    dev_ds = DataSet(file="data/fnc-1/competition_test_stances.csv", reader=csvr, formatter=fnc_formatter)
    test_ds = DataSet(file="data/fever/fever.dev.pages.p5.jsonl", reader=jlr, formatter=fever_formatter)

    train_ds.read()
    test_ds.read()
    dev_ds.read()

    train_feats, dev_feats, test_feats = f.load(train_ds, dev_ds, test_ds)

    input_shape = train_feats[0].shape[1]
    model = SimpleMLP(input_shape,100,4)

    if gpu():
        model.cuda()

    model = train(model, train_feats, 500, 1e-2, 90,dev_feats,clip=5,early_stopping=EarlyStopping())

    test_data, test_labels = test_feats
    print("FEVER SCORE {0}".format(evaluate(model,test_data, test_labels, 500)))
