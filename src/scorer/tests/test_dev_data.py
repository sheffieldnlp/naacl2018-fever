import unittest
import json
from fever.scorer import *

class TestRealData(unittest.TestCase):

    def setUp(self):
        self.predictions = []
        self.actual = []
        with open('dev.jsonl') as f:
            for ix,line in enumerate(f):
                line = json.loads(line)
                self.predictions.append({"predicted_label":line["label"],
                                         "predicted_evidence":[[e[2],e[3]]
                                                               for e in line["all_evidence"] if e[2] is not None]})

        with open('dev.jsonl') as f:
            for line in f:

                line = json.loads(line)
                self.actual.append({"label":line["label"],
                                    "evidence":line["evidence"]})


    def test_scores(self):
        score,acc,pr,rec,f1 = fever_score(self.predictions,self.actual,max_evidence=None)
        self.assertEqual(score,1.0)

    def test_acc(self):
        score, acc, pr, rec, f1 = fever_score(self.predictions,self.actual,max_evidence=None)
        self.assertEqual(acc,1.0)

    def test_f1(self):
        score, acc, pr, rec, f1 = fever_score(self.predictions, self.actual,max_evidence=None)
        self.assertEqual(f1, 1.0)

    def test_pr(self):
        score, acc, pr, rec, f1 = fever_score(self.predictions, self.actual,max_evidence=None)
        self.assertEqual(pr, 1.0)

    def test_rec(self):
        score, acc, pr, rec, f1 = fever_score(self.predictions, self.actual,max_evidence=None)
        self.assertEqual(rec, 1.0)
