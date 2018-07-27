import unittest
import json
from fever.scorer import *


class TestRealData(unittest.TestCase):

    def setUp(self):
        self.predictions = []
        self.actual = []
        with open('predictions.jsonl') as f:
            for line in f:
                self.predictions.append(json.loads(line))

        with open('actual.jsonl') as f:
            for line in f:
                self.actual.append(json.loads(line))


    def test_scores(self):
        score, acc, pr, rec, f1 = fever_score(self.predictions,self.actual)
        self.assertAlmostEqual(score,0.32573257325732574)

    def test_acc(self):
        score, acc, pr, rec, f1 = fever_score(self.predictions,self.actual)
        self.assertAlmostEqual(acc,0.5208520852085209)

    def test_pr(self):
        score, acc, pr, rec, f1 = fever_score(self.predictions, self.actual)
        self.assertAlmostEqual(pr, 0.1043804380438068)

    def test_rec(self):
        score, acc, pr, rec, f1 = fever_score(self.predictions, self.actual)
        self.assertAlmostEqual(rec, 0.44224422442244227)

    def test_f1(self):
        score, acc, pr, rec, f1 = fever_score(self.predictions, self.actual)
        self.assertAlmostEqual(f1, 0.1688970477815144)