import argparse
import json
import sys
from fever.scorer import fever_score

parser = argparse.ArgumentParser()
parser.add_argument("--predicted_labels",type=str)

parser.add_argument("--predicted_evidence",type=str)
parser.add_argument("--out_file",type=str)

args = parser.parse_args()

predicted_labels =[]
predicted_evidence = []
actual = []

with open(args.predicted_labels,"r") as predictions_file:
    for line in predictions_file:
        predicted_labels.append(json.loads(line)["predicted"])


with open(args.predicted_evidence,"r") as predictions_file:
    for line in predictions_file:
        predicted_evidence.append(json.loads(line)["predicted_sentences"])

predictions = []
for ev,label in zip(predicted_evidence,predicted_labels):
    predictions.append({"predicted_evidence":ev,"predicted_label":label})

with open(args.out_file,"w+") as f:
    for line in predictions:
        f.write(json.dumps(line)+"\n")
