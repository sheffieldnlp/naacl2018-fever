from fever.scorer import evidence_macro_precision
import unittest


class PrecisionTestCase(unittest.TestCase):
    def test_precision_nei_no_contribution_to_score(self):
        instance = {"label": "not enough info", "predicted_label": "not enough info"}

        p,h = evidence_macro_precision(instance)
        self.assertEqual(p,0)

    def test_precision_nei_no_contribution_to_hits(self):
        instance = {"label": "not enough info", "predicted_label": "not enough info"}

        p,h = evidence_macro_precision(instance)
        self.assertEqual(h,0)

    def test_precision_not_nei_contribution_to_hits(self):
        instance = {"label": "supports", "predicted_label": "supports","evidence":[],"predicted_evidence":[]}

        p,h = evidence_macro_precision(instance)
        self.assertEqual(h,1)

    def test_precision_no_predictions_perfect_score(self):
        instance = {"label": "supports", "predicted_label": "supports","evidence":[],"predicted_evidence":[]}

        p,h = evidence_macro_precision(instance)
        self.assertEqual(p,1)

    def test_precision_wrong_predictions_zero_score(self):
        instance = {"label": "supports", "predicted_label": "supports","evidence":[],"predicted_evidence":[["page",0]]}

        p,h = evidence_macro_precision(instance)
        self.assertEqual(p,0)

    def test_precision_wrong_predictions_pname_zero_score(self):
        instance = {"label": "supports", "predicted_label": "supports","evidence":[[[None,None,"other",0]]],"predicted_evidence":[["page",0]]}

        p,h = evidence_macro_precision(instance)
        self.assertEqual(p,0)

    def test_precision_wrong_predictions_line_zero_score(self):
        instance = {"label": "supports", "predicted_label": "supports","evidence":[[[None,None,"page",1]]],"predicted_evidence":[["page",0]]}

        p,h = evidence_macro_precision(instance)
        self.assertEqual(p,0)


    def test_precision_partial_predictions_half_score(self):
        instance = {"label": "supports", "predicted_label": "supports","evidence":[[[None,None,"page",0]]],"predicted_evidence":[["page",0],["page", 1]]}

        p,h = evidence_macro_precision(instance)
        self.assertEqual(p,0.5)

    def test_precision_partial_predictions_diff_groups_third_score(self):
        instance = {"label": "supports", "predicted_label": "supports","evidence":[[[None,None,"page",0]],[[None,None,"page",1]]],"predicted_evidence":[["page",0],["page",1],["page", 2]]}

        p,h = evidence_macro_precision(instance)
        self.assertAlmostEqual(p,0.66666667)
