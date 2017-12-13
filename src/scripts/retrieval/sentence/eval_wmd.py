import wmd
import spacy

from common.dataset.data_set import DataSet
from common.dataset.reader import JSONLineReader
from retrieval.fever_doc_db import FeverDocDB
from retrieval.sentence import FEVERSentenceRelatednessFormatter, FEVERSentenceFormatter, FEVERSentenceTextFormatter
from scripts.retrieval.sentence.mlp_train import RelatedLabelSchema

nlp = spacy.load("en", create_pipeline=wmd.WMD.create_spacy_pipeline)


def wmd_sim(claim,lines):
    cl = nlp(claim)
    scores = []
    for line in lines:
        scores.append((cl.similarity(nlp(line)),line))
    return scores

db = FeverDocDB("data/fever/fever.db")
idx = set(db.get_doc_ids())


jlr = JSONLineReader()
formatter = FEVERSentenceTextFormatter(idx,db,RelatedLabelSchema())
dev_ds = DataSet(file="data/fever-data/dev.jsonl", reader=jlr, formatter=formatter)

dev_ds.read()

for data in dev_ds.data:
    print(data)