import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str)
parser.add_argument('--count', type=int, default=1)
args = parser.parse_args()

split = args.split
k = args.count

q = 0
hits = 0

recalls = []
with open("data/fever/{0}.pages.p{1}.jsonl".format(split,k),"r") as f:
    for idx,line in enumerate(f):
        js = json.loads(line)

        flattened_evidence = [evidence for evidence_group in js["evidence"] for evidence in evidence_group]

        predicted = [t[0] for t in js['predicted_pages']]

        if js["label"] != "NOT ENOUGH INFO":

            actual = [annotation[2] for annotation in flattened_evidence]

            for page in set(actual):
                q += 1
                if page in predicted:
                    hits+=1



    print(hits/q)
