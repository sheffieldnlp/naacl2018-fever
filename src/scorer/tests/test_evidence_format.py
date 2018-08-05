import unittest
from fever.scorer import *


class TestEvidenceRequirements(unittest.TestCase):
    def test_no_evidence_no_error(self):
        instance = {"predicted_evidence": []}
        check_predicted_evidence_format(instance)
        self.assertTrue(True)

    def test_list_of_strings_evidence_error(self):
        instance = {"predicted_evidence": ["line","text"]}
        with self.assertRaises(AssertionError):
            check_predicted_evidence_format(instance)


    def test_list_of_lists_evidence_error(self):
        instance = {"predicted_evidence": [[["page",0]]]}
        with self.assertRaises(AssertionError):
            check_predicted_evidence_format(instance)


    def test_first_param_not_str_error(self):
        instance = {"predicted_evidence": [[[0,0]]]}
        with self.assertRaises(AssertionError):
            check_predicted_evidence_format(instance)



    def test_second_param_not_int_error(self):
        instance = {"predicted_evidence": [[["page","some text"]]]}
        with self.assertRaises(AssertionError):
            check_predicted_evidence_format(instance)

