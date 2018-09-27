from fever.scorer import evidence_macro_recall, evidence_macro_precision, fever_score
import unittest


class MaxEvidenceTestCase(unittest.TestCase):


    def test_recall_partial_predictions_same_groups_zero_score(self):
        instance = {"label": "supports", "predicted_label": "supports","evidence":[[[None,None,"page",0],[None,None,"page",1]]],"predicted_evidence":[["page",0],["page", 1]]}

        p,h = evidence_macro_recall(instance,max_evidence=1)
        self.assertEqual(p,0)

    def test_recall_partial_predictions_same_groups_one_score(self):
        instance = {"label": "supports", "predicted_label": "supports","evidence":[[[None,None,"page",0],[None,None,"page",1]]],"predicted_evidence":[["page",0],["page", 1]]}

        p,h = evidence_macro_recall(instance,max_evidence=2)
        self.assertEqual(p,1)


    def test_precision_partial_predictions_same_groups_zero_score(self):
        instance = {"label": "supports", "predicted_label": "supports","evidence":[[[None,None,"page",0],[None,None,"page",2]]],"predicted_evidence":[["page",0],["page", 1]]}

        p,h = evidence_macro_precision(instance,max_evidence=1)
        self.assertEqual(p,1)

    def test_precision_partial_predictions_same_groups_one_score(self):
        instance = {"label": "supports", "predicted_label": "supports","evidence":[[[None,None,"page",0],[None,None,"page",2]]],"predicted_evidence":[["page",0],["page", 1]]}

        p,h = evidence_macro_precision(instance,max_evidence=2)
        self.assertEqual(p,0.5)


    def test_strict_partial_one(self):
        instance = {"label": "supports", "predicted_label": "supports","evidence":[[[None,None,"page",0],[None,None,"page",1]]],"predicted_evidence":[["page",0],["page", 1]]}

        strict,_,_,_,_ = fever_score([instance],max_evidence=2)
        self.assertEqual(strict,1)

    def test_strict_partial_zero(self):
        instance = {"label": "supports", "predicted_label": "supports","evidence":[[[None,None,"page",0],[None,None,"page",1]]],"predicted_evidence":[["page",0],["page", 1]]}

        strict,_,_,_,_ = fever_score([instance],max_evidence=1)
        self.assertEqual(strict,0)


    def test_global_precision_partial_two_sents(self):
        instance = {"label": "supports", "predicted_label": "supports","evidence":[[[None,None,"page",0],[None,None,"page",2]]],"predicted_evidence":[["page",0],["page", 1]]}

        _,_,p,_,_ = fever_score([instance],max_evidence=2)
        self.assertEqual(p,0.5)

    def test_global_precision_partial_one_sent(self):
        instance = {"label": "supports", "predicted_label": "supports","evidence":[[[None,None,"page",0],[None,None,"page",2]]],"predicted_evidence":[["page",0],["page", 1]]}

        _,_,p,_,_ = fever_score([instance],max_evidence=1)
        self.assertEqual(p,1)


    def test_global_recall_partial_two_sents(self):
        instance = {"label": "supports", "predicted_label": "supports","evidence":[[[None,None,"page",0],[None,None,"page",1]]],"predicted_evidence":[["page",0],["page", 1]]}

        _,_,_,r,_ = fever_score([instance],max_evidence=2)
        self.assertEqual(r,1)

    def test_global_recall_partial_one_sent(self):
        instance = {"label": "supports", "predicted_label": "supports","evidence":[[[None,None,"page",0],[None,None,"page",1]]],"predicted_evidence":[["page",0],["page", 1]]}

        _,_,_,r,_ = fever_score([instance],max_evidence=1)
        self.assertEqual(r,0)

    def test_non_modification(self):
        instance = {"label": "supports", "predicted_label": "supports","evidence":[[[None,None,"page",0],[None,None,"page",1]]],"predicted_evidence":[["page",0],["page", 1]]}
        instance_copy = instance.copy()
        _,_,_,_,_ = fever_score([instance],max_evidence=0)
        self.assertEqual(instance_copy,instance)

