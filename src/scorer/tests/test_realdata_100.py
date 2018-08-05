import unittest
import json
from fever.scorer import *


class TestRealData100(unittest.TestCase):

    def setUp(self):
        self.predictions = []
        self.actual = []
        with open('predictions.jsonl') as f:
            for line in f:
                self.predictions.append(json.loads(line))

        with open('actual.jsonl') as f:
            for line in f:
                self.actual.append(json.loads(line))

        self.predictions = self.predictions[:100]
        self.actual = self.actual[:100]

    def test_scores(self):
        score, acc, pr, rec, f1 = fever_score(self.predictions,self.actual)
        self.assertAlmostEqual(score,0.26)

    def test_acc(self):
        score, acc, pr, rec, f1 = fever_score(self.predictions,self.actual)
        self.assertAlmostEqual(acc,0.48)

    def test_pr(self):
        score, acc, pr, rec, f1 = fever_score(self.predictions, self.actual)
        self.assertAlmostEqual(pr, 0.09189189189189195)

    def test_rec(self):
        score, acc, pr, rec, f1 = fever_score(self.predictions, self.actual)
        self.assertAlmostEqual(rec, 0.3783783783783784)

    def test_f1(self):
        score, acc, pr, rec, f1 = fever_score(self.predictions, self.actual)
        self.assertAlmostEqual(f1, 0.14787200994097552)