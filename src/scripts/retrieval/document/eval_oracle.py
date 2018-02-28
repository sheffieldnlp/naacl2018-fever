import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str)
parser.add_argument('--count', type=int, default=1)
args = parser.parse_args()

split = args.split
k = args.count

def preprocess(p):
    return p.replace(" ","_").replace("(","-LRB-").replace(")","-RRB-").replace(":","-COLON-")

q = 0
hits = 0

recalls = []
with open("data/fever/{0}.pages.p{1}.jsonl".format(split,k),"r") as f:

    for idx,line in enumerate(f):

        q = 1
        hits = 0


        js = json.loads(line)
        predicted = [t[0] for t in js['predicted_pages']]

        if js["verifiable"] == "NOT ENOUGH INFO":
            hits += 1
        else:
            for ev in js['evidence']:
                pages = [annotation[2] for annotation in ev]
                if all(page in predicted for page in pages):
                    hits+=1
                    break

        recalls.append(hits/q)


    print(sum(recalls)/len(recalls))
