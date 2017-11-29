from common.dataset.data_set import DataSet
from common.dataset.formatter import Formatter
from common.dataset.label_schema import LabelSchema
from common.dataset.reader import JSONLineReader
from common.features.feature_function import Features
from retrieval.fever_doc_db import FeverDocDB
from rte.riedel.features import TermFrequencyFeatureFunction



class FEVERFormatter(Formatter):
    def format_line(self,line):
        annotation = line["evidence"]

        if annotation == "-":
            return None

        return {"claim":"", "evidence": [], "label":self.label_schema.get_id(annotation)}


class FEVERLabelSchema(LabelSchema):
    def __init__(self):
        super().__init__(["supported","refuted","not enough information"])


if __name__ == "__main__":
    db = FeverDocDB("data/fever/drqa.db")
    f = Features([TermFrequencyFeatureFunction(db)])





    jlr = JSONLineReader()
    formatter = FEVERFormatter(FEVERLabelSchema())

    train = DataSet(file="data/fever/fever.train.jsonl", reader=jlr, formatter=formatter)
    dev = DataSet(file="data/fever/fever.dev.jsonl", reader=jlr, formatter=formatter)
    test = DataSet(file="data/fever/fever.test.jsonl", reader=jlr, formatter=formatter)

    f.inform(train,dev,test)