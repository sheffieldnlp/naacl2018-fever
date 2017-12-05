import sys
import json

from sklearn.metrics import accuracy_score, classification_report

fp = sys.argv[1]


predictions = []
actual = []
with open(fp,"r") as f:

    for line in f:
        js = json.loads(line)

        predictions.append(js["prediction"])
        actual.append(js["verdict"] if js["verdict"] is not None else js["verifiable"])



print(accuracy_score(actual,predictions))
print(classification_report(actual,predictions))