from common.dataset.data_set import DataSet
from common.dataset.formatter import Formatter
from common.dataset.label_schema import LabelSchema
from common.dataset.reader import JSONLineReader
from common.features.feature_function import Features
from retrieval.fever_doc_db import FeverDocDB
from rte.riedel.features import TermFrequencyFeatureFunction


def preprocess(p):
    return p.replace(" ","_").replace("(","-LRB-").replace(")","-RRB-").replace(":","-COLON-").split("#")[0]

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


if __name__ == "__main__":
    db = FeverDocDB("data/fever/drqa.db")
    idx = set(db.get_doc_ids())

    f = Features([TermFrequencyFeatureFunction(db)])
    jlr = JSONLineReader()

    formatter = FEVERFormatter(idx, FEVERLabelSchema())

    train = DataSet(file="data/fever/fever.train.jsonl", reader=jlr, formatter=formatter)
    dev = DataSet(file="data/fever/fever.dev.jsonl", reader=jlr, formatter=formatter)
    test = DataSet(file="data/fever/fever.test.jsonl", reader=jlr, formatter=formatter)

    train.read()
    dev.read()
    test.read()

    train_feats, dev_feats, test_feats = f.load(train,dev,test)
    print(train_feats)