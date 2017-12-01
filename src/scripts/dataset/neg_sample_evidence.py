from common.dataset.reader import JSONLineReader
from common.util.random import SimpleRandom
from retrieval.fever_doc_db import FeverDocDB
import numpy as np



jlr = JSONLineReader()

docdb = FeverDocDB("data/fever/drqa.db")

idx = docdb.get_doc_ids()

r = SimpleRandom.get_instance()



for line in jlr.read("data/fever/fever.dev.jsonl"):
    if isinstance(line["evidence"][0],int):
        print(line["evidence"])
        print(idx[r.next_rand(0,len(idx))])






