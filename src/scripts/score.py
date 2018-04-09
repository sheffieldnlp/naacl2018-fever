import argparse
import json
import sys
from fever.scorer import fever_score

parser = argparse.ArgumentParser()
parser.add_argument("--predicted_labels",type=str)

parser.add_argument("--predicted_evidence",type=str)
parser.add_argument("--actual",type=str)

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

with open(args.actual, "r") as actual_file:
    for line in actual_file:
        actual.append(json.loads(line))

predictions = []
for ev,label in zip(predicted_evidence,predicted_labels):
    predictions.append({"predicted_evidence":ev,"predicted_label":label})

score,acc = fever_score(predictions,actual)
print("Score:\t{0}\t\t\tAccuracy:\t{1}".format(round(score,4),round(acc,4)))