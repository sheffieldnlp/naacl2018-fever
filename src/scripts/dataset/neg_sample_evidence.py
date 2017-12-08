from common.dataset.reader import JSONLineReader
from common.util import random
from common.util.random import SimpleRandom
from retrieval.fever_doc_db import FeverDocDB
import json
import random

jlr = JSONLineReader()

docdb = FeverDocDB("data/fever/drqa.db")

idx = docdb.get_doc_ids()

SimpleRandom.set_seeds()

r = random.Random(SimpleRandom.get_seed())


with open("data/fever/test.ns.rand.jsonl", "w+") as f:
    for line in jlr.read("data/fever-data/test.jsonl"):
        if line["VERIFIABLE"] == "NOT ENOUGH INFO":
            ev = []
            for e in line['evidence']:
                ev.append((e[0], idx[r.next_rand(0, len(idx))], -1))
            line['evidence'] = ev
        f.write(json.dumps(line)+"\n")

with open("data/fever/dev.ns.rand.jsonl", "w+") as f:
    for line in jlr.read("data/fever-data/dev.jsonl"):
        if line["VERIFIABLE"]=="NOT ENOUGH INFO":
            ev = []
            for e in line['evidence']:
                ev.append((e[0],idx[r.next_rand(0,len(idx))],-1))
            line['evidence'] = ev
        f.write(json.dumps(line)+"\n")




with open("data/fever/train.ns.rand.jsonl", "w+") as f:
    for line in jlr.read("data/fever-data/train.jsonl"):
        if line["VERIFIABLE"] == "NOT ENOUGH INFO":
            ev = []
            for e in line['evidence']:
                ev.append((e[0], idx[r.next_rand(0, len(idx))], -1))
            line['evidence'] = ev
        f.write(json.dumps(line)+"\n")
