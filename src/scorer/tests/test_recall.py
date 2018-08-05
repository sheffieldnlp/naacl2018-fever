from fever.scorer import evidence_macro_recall
import unittest


class RecallTestCase(unittest.TestCase):
    def test_recall_nei_no_contribution_to_score(self):
        instance = {"label": "not enough info", "predicted_label": "not enough info"}

        p,h = evidence_macro_recall(instance)
        self.assertEqual(p,0)

    def test_recall_nei_no_contribution_to_hits(self):
        instance = {"label": "not enough info", "predicted_label": "not enough info"}

        p,h = evidence_macro_recall(instance)
        self.assertEqual(h,0)

    def test_recall_not_nei_contribution_to_hits(self):
        instance = {"label": "supports", "predicted_label": "supports","evidence":[],"predicted_evidence":[]}

        p,h = evidence_macro_recall(instance)
        self.assertEqual(h,1)


    def test_recall_no_evidence_perfect_score(self):
        instance = {"label": "supports", "predicted_label": "supports","evidence":[],"predicted_evidence":[]}

        p,h = evidence_macro_recall(instance)
        self.assertEqual(p,1)

    def test_recall_no_evidence_prediction_perfect_score(self):
        instance = {"label": "supports", "predicted_label": "supports","evidence":[],"predicted_evidence":[["page",0]]}

        p,h = evidence_macro_recall(instance)
        self.assertEqual(p,1)

    def test_recall_no_predictions_zero_score(self):
        instance = {"label": "supports", "predicted_label": "supports", "evidence": [[[None,None,"other",0]]], "predicted_evidence": []}

        p, h = evidence_macro_recall(instance)
        self.assertEqual(p, 0)

    def test_recall_wrong_predictions_pname_zero_score(self):
        instance = {"label": "supports", "predicted_label": "supports","evidence":[[[None,None,"other",0]]],"predicted_evidence":[["page",0]]}

        p,h = evidence_macro_recall(instance)
        self.assertEqual(p,0)

    def test_recall_wrong_predictions_line_zero_score(self):
        instance = {"label": "supports", "predicted_label": "supports","evidence":[[[None,None,"page",1]]],"predicted_evidence":[["page",0]]}

        p,h = evidence_macro_recall(instance)
        self.assertEqual(p,0)


    def test_recall_partial_predictions_zero_score(self):
        instance = {"label": "supports", "predicted_label": "supports","evidence":[[[None,None,"page",0],[None,None,"page",1]]],"predicted_evidence":[["page",0]]}

        p,h = evidence_macro_recall(instance)
        self.assertEqual(p,0)

    def test_recall_partial_predictions_diff_groups_zero_score(self):
        instance = {"label": "supports", "predicted_label": "supports","evidence":[[[None,None,"page",0],[None,None,"page",1]]],"predicted_evidence":[["page",0],["page", 2]]}

        p,h = evidence_macro_recall(instance)
        self.assertAlmostEqual(p,0)

    def test_recall_correct_predictions_exact_one_score(self):
        instance = {"label": "supports", "predicted_label": "supports","evidence":[[[None,None,"page",0]]],"predicted_evidence":[["page",0]]}

        p,h = evidence_macro_recall(instance)
        self.assertEqual(p,1)


    def test_recall_correct_predictions_exact_one_score_multi_evidence(self):
        instance = {"label": "supports", "predicted_label": "supports","evidence":[[[None,None,"page",0],[None,None,"page",1]]],"predicted_evidence":[["page",0],["page",1]]}

        p,h = evidence_macro_recall(instance)
        self.assertEqual(p,1)

    def test_recall_correct_predictions_exact_one_score_multi_evidence_groups(self):
        instance = {"label": "supports", "predicted_label": "supports","evidence":[[[None,None,"page",0]],[[None,None,"page",1]]],"predicted_evidence":[["page",0],["page",1]]}

        p,h = evidence_macro_recall(instance)
        self.assertEqual(p,1)

    def test_recall_correct_predictions_partial_one_score_multi_evidence_groups(self):
        instance = {"label": "supports", "predicted_label": "supports",
                    "evidence": [[[None, None, "page", 0]], [[None, None, "page", 1]]],
                    "predicted_evidence": [["page", 0]]}

        p,h = evidence_macro_recall(instance)
        self.assertAlmostEqual(p,1)

    def test_recall_correct_predictions_partial_other_one_score_multi_evidence_groups(self):
        instance = {"label": "supports", "predicted_label": "supports",
                    "evidence": [[[None, None, "page", 0]], [[None, None, "page", 1]]],
                    "predicted_evidence": [["page", 0],["page",2]]}

        p,h = evidence_macro_recall(instance)
        self.assertAlmostEqual(p,1)