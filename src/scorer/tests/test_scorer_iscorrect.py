import unittest
from fever.scorer import *


class IsCorrectTestCase(unittest.TestCase):
    def test_equal_labels(self):
        instance = {"label":"not enough info", "predicted_label":"not enough info"}
        self.assertTrue(is_correct_label(instance))

    def test_equal_labels_different_case1(self):
        instance = {"label": "not enough info", "predicted_label": "NOT ENOUGH INFO"}
        self.assertTrue(is_correct_label(instance))

    def test_equal_labels_different_case2(self):
        instance = {"label": "NOT ENOUGH INFO", "predicted_label": "not enough info"}
        self.assertTrue(is_correct_label(instance))

    def test_different_labels(self):
        instance = {"label":"supports", "predicted_label":"not enough info"}
        self.assertFalse(is_correct_label(instance))


