import os
import wmd
import spacy
import numpy as np
import json
from sklearn import metrics
from tqdm import tqdm

from common.dataset.data_set import DataSet
from common.dataset.reader import JSONLineReader
from common.util.array import flatten
from retrieval.fever_doc_db import FeverDocDB
from retrieval.sentence import FEVERSentenceRelatednessFormatter, FEVERSentenceFormatter, FEVERSentenceTextFormatter
from scripts.retrieval.sentence.mlp_train import RelatedLabelSchema

nlp = spacy.load("en", create_pipeline=wmd.WMD.create_spacy_pipeline)


def wmd_sim(claim,lines):
    cl = nlp(claim)
    scores = []
    for line in lines:
        scores.append(cl.similarity(nlp(line)))
    return scores

db = FeverDocDB("data/fever/fever.db")
idx = set(db.get_doc_ids())


jlr = JSONLineReader()
formatter = FEVERSentenceTextFormatter(idx,db,RelatedLabelSchema())
dev_ds = DataSet(file="data/fever-data/dev.jsonl", reader=jlr, formatter=formatter)

dev_ds.read()

def doc_lines(db,doc):
    lines = db.get_doc_lines(doc)
    return [line.split("\t")[1] if len(line.split("\t"))>1 else "" for line in lines.split("\n")]


#thresh = 0.8

y_true = []
y_scores = []

if os.getenv("TEST") is None:
    for data in tqdm(dev_ds.data):
        if data["label_text"] != "NOT ENOUGH INFO":
            all_lines = []
            for doc in data["docs"]:
                lines = doc_lines(db,doc)
                all_lines.extend(list(zip(wmd_sim(data["claim"], lines),enumerate(lines),[doc]*len(lines))))

            #filtered = list(filter(lambda line: line[0] > thresh,all_lines))
            gold = set([(ev[0],ev[1]) for ev in data["evidence"]])
            predicted = set([(ev[0],(ev[2],ev[1][0])) for ev in all_lines])

            for score, prediction in predicted:
                y_true.append(1 if prediction in gold else 0)
                y_scores.append(score)


    json.dump({"true":y_true,"scores":y_scores},open("roc.all.json","w+"))
else:
    roc = json.load(open("roc.all.json","r"))
    y_true = [r>0 for r in roc["true"]]
    y_scores = roc["scores"]


fpr,tpr,thresh = metrics.roc_curve(y_true,y_scores)

fnr = [1-t for t in tpr]
tnr = [1-t for t in fpr]

fp = [f * len(y_true) for f in fpr]
tp = [f * len(y_true) for f in tpr]
tn = [f * len(y_true) for f in tnr]
fn = [f * len(y_true) for f in fnr]


f1 = [ (tp[i] + tp[i])/(tp[i]+tp[i]+fp[i]+fn[i] ) for i in range(len(tp)) ]

best = np.argmax(f1)

print(f1[best], tpr[best])
print(thresh[best])