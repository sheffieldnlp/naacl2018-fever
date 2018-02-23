import argparse
import json

from tqdm import tqdm

from common.dataset.reader import JSONLineReader
from common.util.random import SimpleRandom
from retrieval.fever_doc_db import FeverDocDB
from retrieval.filter_uninformative import uninformative

parser = argparse.ArgumentParser()
parser.add_argument('db_path', type=str, help='/path/to/fever.db')

args = parser.parse_args()


jlr = JSONLineReader()

docdb = FeverDocDB(args.db_path)

idx = docdb.get_non_empty_doc_ids()
idx = list(filter(lambda item: not uninformative(item),tqdm(idx)))


r = SimpleRandom.get_instance()

with open("data/fever/test.ns.rand.jsonl", "w+") as f:
    for line in jlr.read("data/fever-data/test.jsonl"):
        if line["label"] == "NOT ENOUGH INFO":

            for evidence_group in line['evidence']:
                for evidence in evidence_group:
                    evidence[2] = idx[r.next_rand(0, len(idx))]
                    evidence[3] = -1


        f.write(json.dumps(line)+"\n")

with open("data/fever/dev.ns.rand.jsonl", "w+") as f:
    for line in jlr.read("data/fever-data/dev.jsonl"):
        if line["label"]=="NOT ENOUGH INFO":
            for evidence_group in line['evidence']:
                for evidence in evidence_group:
                    evidence[2] = idx[r.next_rand(0, len(idx))]
                    evidence[3] = -1

        f.write(json.dumps(line)+"\n")



with open("data/fever/train.ns.rand.jsonl", "w+") as f:
    for line in jlr.read("data/fever-data/train.jsonl"):
        if line["label"] == "NOT ENOUGH INFO":
            for evidence_group in line['evidence']:
                for evidence in evidence_group:
                    evidence[2] = idx[r.next_rand(0, len(idx))]
                    evidence[3] = -1

        f.write(json.dumps(line)+"\n")
