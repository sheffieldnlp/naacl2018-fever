from common.dataset.data_set import DataSet
from common.dataset.formatter import Formatter
from common.dataset.label_schema import LabelSchema
from common.dataset.reader import CSVReader
from common.features.feature_function import Features
from common.training.options import gpu
from common.training.run import train
from rte.riedel.fnc_features import FNCTermFrequencyFeatureFunction
from rte.riedel.model import SimpleMLP


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

class FNCLabelSchema(LabelSchema):
    def __init__(self):
        super().__init__(["agree","disagree","discuss","unrelated"])


class FNCFormatter(Formatter):
    def __init__(self,label_schema):
        super().__init__(label_schema)

    def format_line(self,line):
        annotation = self.label_schema.get_id(line["Stance"]) if "Stance" in line else None
        return {"claim":line["Headline"], "evidence": line["Body ID"], "label":annotation}


if __name__ == "__main__":
    bodies = Bodies("data/fnc-1/train_bodies.csv","data/fnc-1/competition_test_bodies.csv")

    f = Features([FNCTermFrequencyFeatureFunction(bodies)])
    csvr = CSVReader()
    formatter = FNCFormatter(FNCLabelSchema())

    train_ds = DataSet(file="data/fnc-1/train_stances.csv", reader=csvr, formatter=formatter)
    test_ds = DataSet(file="data/fnc-1/competition_test_stances.csv", reader=csvr, formatter=formatter)

    train_ds.read()
    test_ds.read()

    train_feats, _, test_feats = f.load(train_ds, None, test_ds)

    input_shape = train_feats[0].shape[1]
    model = SimpleMLP(input_shape,100,4)

    if gpu():
        model.cuda()

    train(model, train_feats, 500, 1e-2, 90,test_feats,clip=5)

