import unittest
from fever.scorer import *


class IsStrictlyCorrectTestCase(unittest.TestCase):
    def test_nei_no_evidence(self):
        instance = {"label": "not enough info", "predicted_label": "not enough info"}
        self.assertTrue(is_strictly_correct(instance))

    def test_supports_no_predicted_evidence(self):
        instance = {"label": "supports", "predicted_label": "supports", "evidence":[[[None,None,"page",0]]]}
        self.assertRaises(AssertionError, lambda: is_strictly_correct(instance))

    def test_supports_evidence_subset(self):
        instance = {"label": "supports",
                    "predicted_label": "supports",
                    "evidence":[
                        [
                            [None,None,"page",0],
                            [None,None,"page",1]
                        ]
                    ],
                    "predicted_evidence":[
                        ["page",0]
                    ]}
        self.assertFalse(is_strictly_correct(instance))

    def test_supports_evidence_wrong_line(self):
        instance = {"label": "supports",
                    "predicted_label": "supports",
                    "evidence":[
                        [
                            [None,None,"page",0],
                        ]
                    ],
                    "predicted_evidence":[
                        ["page",2]
                    ]}
        self.assertFalse(is_strictly_correct(instance))

    def test_supports_evidence_wrong_page(self):
        instance = {"label": "supports",
                    "predicted_label": "supports",
                    "evidence":[
                        [
                            [None,None,"page",0],
                        ]
                    ],
                    "predicted_evidence":[
                        ["page1",0]
                    ]}
        self.assertFalse(is_strictly_correct(instance))

    def test_supports_evidence_correct_single(self):
        instance = {"label": "supports",
                    "predicted_label": "supports",
                    "evidence":[
                        [
                            [None,None,"page",0],
                        ]
                    ],
                    "predicted_evidence":[
                        ["page",0]
                    ]}
        self.assertTrue(is_strictly_correct(instance))

    def test_supports_evidence_matches_1_group(self):
        instance = {"label": "supports",
                    "predicted_label": "supports",
                    "evidence":[
                        [
                            [None, None, "page", 0],
                        ],
                        [
                            [None, None, "page1", 0],
                        ]
                    ],
                    "predicted_evidence":[
                        ["page",0]
                    ]}
        self.assertTrue(is_strictly_correct(instance))



    def test_supports_evidence_matches_part_group(self):
        instance = {"label": "supports",
                    "predicted_label": "supports",
                    "evidence":[
                        [
                            [None,None,"page2",0],
                            [None,None,"page1",1],
                        ],
                        [
                            [None, None, "page2", 0],
                        ]
                    ],
                    "predicted_evidence":[
                        ["page2",0]
                    ]}
        self.assertTrue(is_strictly_correct(instance))


    def test_supports_evidence_matches_disjount(self):
        instance = {"label": "supports",
                    "predicted_label": "supports",
                    "evidence":[
                        [
                            [None,None,"page1",1],
                        ],
                        [
                            [None, None, "page2", 0],
                        ]
                    ],
                    "predicted_evidence":[
                        ["page1",1],
                        ["page2",0]
                    ]}
        self.assertTrue(is_strictly_correct(instance))

    def test_supports_evidence_matches_superset(self):
        instance = {"label": "supports",
                    "predicted_label": "supports",
                    "evidence":[
                        [
                            [None, None, "page1", 1],
                            [None, None, "page2", 0],
                        ]
                    ],
                    "predicted_evidence":[
                        ["page1",1],
                        ["page2",0],
                        ["page3",2]
                    ]}
        self.assertTrue(is_strictly_correct(instance))

