from common.dataset.reader import JSONLineReader
from common.util.random import SimpleRandom
from retrieval.fever_doc_db import FeverDocDB
import json


jlr = JSONLineReader()

docdb = FeverDocDB("data/fever/drqa.db")

idx = docdb.get_doc_ids()

r = SimpleRandom.get_instance()



with open("data/fever/fever.dev.ns.rand.jsonl", "w+") as f:
    for line in jlr.read("data/fever/fever.dev.jsonl"):
        if isinstance(line["evidence"][0],int):
            ev = []
            for e in line['evidence']:
                ev.append((e,idx[r.next_rand(0,len(idx))]))
            line['evidence'] = ev
        f.write(json.dumps(line)+"\n")




with open("data/fever/fever.train.ns.rand.jsonl", "w+") as f:
    for line in jlr.read("data/fever/fever.train.jsonl"):
        if isinstance(line["evidence"][0],int):
            ev = []
            for e in line['evidence']:
                ev.append((e,idx[r.next_rand(0,len(idx))]))
            line['evidence'] = ev
        f.write(json.dumps(line)+"\n")


with open("data/fever/fever.test.ns.rand.jsonl", "w+") as f:
    for line in jlr.read("data/fever/fever.test.jsonl"):
        if isinstance(line["evidence"][0],int):
            ev = []
            for e in line['evidence']:
                ev.append((e,idx[r.next_rand(0,len(idx))]))
            line['evidence'] = ev
        f.write(json.dumps(line)+"\n")