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
        predicted = [t[0] for t in js['predicted_pages']]

        if js["label"] != "NOT ENOUGH INFO":
            q+=1
            for ev in js['evidence']:
                pages = [annotation[2] for annotation in ev]
                if all(page in predicted for page in pages):
                    hits+=1
                    break

    print(hits/q)
