import unittest
from fever.scorer import *


class FeverScorerTestCase(unittest.TestCase):
    def test_fever_perfect_scorer_strict(self):
        instance1 = {"label": "not enough info", "predicted_label": "not enough info", "evidence":[],"predicted_evidence":[]}
        instance2 = {"label": "not enough info", "predicted_label": "not enough info", "evidence":[],"predicted_evidence":[]}

        predictions = [instance1,instance2]
        score, acc, pr, rec, f1 = fever_score(predictions)
        self.assertEqual(1.0,score)


    def test_fever_perfect_scorer_acc(self):
        instance1 = {"label": "not enough info", "predicted_label": "not enough info", "evidence":[],"predicted_evidence":[]}
        instance2 = {"label": "not enough info", "predicted_label": "not enough info", "evidence":[],"predicted_evidence":[]}

        predictions = [instance1,instance2]
        score, acc, pr, rec, f1 = fever_score(predictions)
        self.assertEqual(1.0,acc)

    def test_fever_imperfect_scorer_strict(self):
        instance1 = {"label": "not enough info", "predicted_label": "refutes", "evidence":[],"predicted_evidence":[]}
        instance2 = {"label": "not enough info", "predicted_label": "supports", "evidence":[],"predicted_evidence":[]}

        predictions = [instance1,instance2]
        score, acc, pr, rec, f1 = fever_score(predictions)
        self.assertEqual(0.0,score)


    def test_fever_imperfect_scorer_acc(self):
        instance1 = {"label": "not enough info", "predicted_label": "refutes", "evidence":[],"predicted_evidence":[]}
        instance2 = {"label": "not enough info", "predicted_label": "supports", "evidence":[],"predicted_evidence":[]}

        predictions = [instance1,instance2]
        score, acc, pr, rec, f1 = fever_score(predictions)
        self.assertEqual(0.0,acc)

    def test_fever_blind_no_data(self):
        instance1 = {"predicted_label": "not enough info", "predicted_evidence":[]}
        instance2 = {"predicted_label": "not enough info", "predicted_evidence":[]}

        predictions = [instance1,instance2]
        self.assertRaises(AssertionError, lambda: fever_score(predictions))

    def test_fever_blind_diff_instances(self):
        instance1 = {"predicted_label": "not enough info", "predicted_evidence":[]}
        instance2 = {"predicted_label": "not enough info", "predicted_evidence":[]}

        actual = [{}]

        predictions = [instance1,instance2]
        self.assertRaises(AssertionError, lambda: fever_score(predictions,actual))

    def test_fever_blind_no_evidence_nei(self):
        instance1 = {"predicted_label": "not enough info", "predicted_evidence":[]}
        instance2 = {"predicted_label": "not enough info", "predicted_evidence":[]}

        actual = [{"label":"not enough info"},{"label":"not enough info"}]
        predictions = [instance1,instance2]

        self.assertRaises(AssertionError, lambda: fever_score(predictions,actual))

    def test_fever_blind_perfect_strict(self):
        instance1 = {"predicted_label": "not enough info", "predicted_evidence":[]}
        instance2 = {"predicted_label": "not enough info", "predicted_evidence":[]}

        actual = [{"label": "not enough info","evidence":[]}, {"label": "not enough info","evidence":[]}]
        predictions = [instance1,instance2]
        score, acc, pr, rec, f1 = fever_score(predictions,actual)
        self.assertEqual(1.0,score)

    def test_fever_blind_perfect_acc(self):
        instance1 = {"predicted_label": "not enough info", "predicted_evidence": []}
        instance2 = {"predicted_label": "not enough info", "predicted_evidence": []}

        actual = [{"label": "not enough info","evidence":[]}, {"label": "not enough info","evidence":[]}]
        predictions = [instance1, instance2]
        score, acc, pr, rec, f1 = fever_score(predictions, actual)
        self.assertEqual(1.0, acc)

    def test_fever_blind_imperfect_strict(self):
        instance1 = {"predicted_label": "supports", "predicted_evidence": []}
        instance2 = {"predicted_label": "refutes", "predicted_evidence": []}


        actual = [{"label": "not enough info","evidence":[]}, {"label": "not enough info","evidence":[]}]
        predictions = [instance1,instance2]
        score, acc, pr, rec, f1 = fever_score(predictions,actual)
        self.assertEqual(0.0,score)

    def test_fever_blind_imperfect_acc(self):
        instance1 = {"predicted_label": "supports", "predicted_evidence": []}
        instance2 = {"predicted_label": "refutes", "predicted_evidence": []}

        actual = [{"label": "not enough info","evidence":[]}, {"label": "not enough info","evidence":[]}]
        predictions = [instance1, instance2]
        score, acc, pr, rec, f1= fever_score(predictions, actual)
        self.assertEqual(0, acc)

    def test_fever_blind_imperfect2_strict(self):
        instance1 = {"predicted_label": "not enough info", "predicted_evidence": []}
        instance2 = {"predicted_label": "refutes", "predicted_evidence": []}

        actual = [{"label": "not enough info", "evidence": []}, {"label": "not enough info", "evidence": []}]
        predictions = [instance1, instance2]
        strict, acc, pr, rec, f1 = fever_score(predictions, actual)
        self.assertEqual(0.5, strict)

    def test_fever_blind_imperfect2_acc(self):
        instance1 = {"predicted_label": "not enough info", "predicted_evidence": []}
        instance2 = {"predicted_label": "refutes", "predicted_evidence": []}

        actual = [{"label": "not enough info", "evidence": []}, {"label": "not enough info", "evidence": []}]
        predictions = [instance1, instance2]
        score, acc, pr, rec, f1 = fever_score(predictions, actual)
        self.assertEqual(0.5, acc)

    def test_fever_blind_imperfect3_strict(self):


        instance1 = {"predicted_label": "refutes", "predicted_evidence": [
            ["page3", 2]
        ]}

        instance2 = {"predicted_label": "refutes", "predicted_evidence": [
            ["page1", 1],
            ["page2", 0],
            ["page3", 2]
        ]}

        actual = [
            {"label": "refutes", "evidence":
                [
                    [
                        [None, None, "page1", 1],
                        [None, None, "page2", 0],
                    ]
                ]},
            {"label": "refutes", "evidence":
                [
                    [
                        [None, None, "page1", 1],
                        [None, None, "page2", 0],
                    ]
                ]}
            ]

        predictions = [instance1, instance2]
        strict,acc,pr,rec,f1 = fever_score(predictions, actual)
        self.assertEqual(0.5, strict)

    def test_fever_blind_imperfect3_acc(self):
        instance1 = {"predicted_label": "refutes", "predicted_evidence": [
            ["page3", 2]
        ]}

        instance2 = {"predicted_label": "refutes", "predicted_evidence": [
            ["page1", 1],
            ["page2", 0],
            ["page3", 2]
        ]}

        actual = [
            {"label": "refutes", "evidence":
                [
                    [
                        [None, None, "page1", 1],
                        [None, None, "page2", 0],
                    ]
                ]},
            {"label": "refutes", "evidence":
                [
                    [
                        [None, None, "page1", 1],
                        [None, None, "page2", 0],
                    ]
                ]}
        ]

        predictions = [instance1, instance2]
        score, acc, pr, rec, f1 = fever_score(predictions, actual)
        self.assertEqual(1.0, acc)
