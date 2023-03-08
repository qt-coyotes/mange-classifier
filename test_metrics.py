import unittest

import torch

from metrics import BinaryExpectedCost


class TestMetrics(unittest.TestCase):
    def test_binary_expected_cost_tn(self):
        target = torch.tensor([0])
        preds = torch.tensor([0])
        expected_cost = BinaryExpectedCost()
        cost = expected_cost(preds, target)
        self.assertEqual(cost, 0)

    def test_binary_expected_cost_fp(self):
        target = torch.tensor([0])
        preds = torch.tensor([1])
        expected_cost = BinaryExpectedCost()
        cost = expected_cost(preds, target)
        self.assertEqual(cost, 1)

    def test_binary_expected_cost_fn(self):
        target = torch.tensor([1])
        preds = torch.tensor([0])
        expected_cost = BinaryExpectedCost()
        cost = expected_cost(preds, target)
        self.assertEqual(cost, 5)

    def test_binary_expected_cost_tp(self):
        target = torch.tensor([1])
        preds = torch.tensor([1])
        expected_cost = BinaryExpectedCost()
        cost = expected_cost(preds, target)
        self.assertEqual(cost, 0)

    def test_binary_expected_cost(self):
        target = torch.tensor([1, 1, 0, 0])
        preds = torch.tensor([0, 1, 1, 0])
        expected_cost = BinaryExpectedCost()
        cost = expected_cost(preds, target)
        self.assertEqual(cost, 1.5)
