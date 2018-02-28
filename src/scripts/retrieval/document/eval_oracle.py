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

        q = 0
        hits = 0


        js = json.loads(line)
        evidence = set([t[1] for t in js["evidence"] if isinstance(t,list) and len(t)>1])
        predicted = [t[0] for t in js['predicted_pages']]

        if js["label"] == "NOT ENOUGH INFO":
            q += 1
            hits += 1
        else:
            for p in evidence:
                q += 1
                if preprocess(p) in predicted:
                    hits+= 1
                break


        recalls.append(hits/q)


    print(sum(recalls)/len(recalls))
