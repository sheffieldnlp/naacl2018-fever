import argparse
import math
import json
import random

parser = argparse.ArgumentParser()
parser.add_argument('--in_file', type=str)
parser.add_argument('--out_file', type=str)
parser.add_argument('--split', type=float)

args = parser.parse_args()


print(args.split)

with open(args.in_file,"r") as in_file:
    lines = in_file.readlines()

total = len(lines)
total_s = len(list(filter(lambda line: json.loads(line)["label"]=="SUPPORTS",lines)))
total_n = len(list(filter(lambda line: json.loads(line)["label"]=="NOT ENOUGH INFO",lines)))
total_r = len(list(filter(lambda line: json.loads(line)["label"]=="REFUTES",lines)))

keep_s = math.ceil(total_s * args.split)
keep_n = math.ceil(total_n * args.split)
keep_r = math.ceil(total_r * args.split)

found_s = []
found_n = []
found_r = []

for id,line in enumerate(lines):
    if json.loads(line)["label"] == "SUPPORTS" and len(found_s) < keep_s:
        found_s.append(id)
    elif json.loads(line)["label"] == "REFUTES" and len(found_r) < keep_r:
        found_r.append(id)
    elif json.loads(line)["label"] == "NOT ENOUGH INFO" and len(found_n) < keep_n:
        found_n.append(id)

keep = found_s+found_n+found_r

rand = random.Random(1236789)
rand.shuffle(keep)

with open(args.out_file,"w+") as out_file:
    out_file.writelines([lines[i] for i in keep])