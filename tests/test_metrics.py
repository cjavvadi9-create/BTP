"""
tests/test_metrics.py
---------------------
Unit tests for evaluation metrics (DTA regression & DTI classification).
"""

import math
import unittest
import numpy as np
from metrics import (
    rmse,
    mse,
    mae,
    pearson,
    spearman,
    ci,
    rm2,
    compute_classification_metrics,
)


class TestMetrics(unittest.TestCase):

    def test_rmse_mse_mae(self):
        y = np.array([1.0, 2.0, 3.0])
        f = np.array([1.0, 3.0, 5.0])
        # errors: [0, -1, -2]
        # sq errors: [0, 1, 4], mean = 5/3
        self.assertAlmostEqual(mse(y, f), 5.0 / 3.0, places=6)
        self.assertAlmostEqual(rmse(y, f), math.sqrt(5.0 / 3.0), places=6)
        # abs errors: [0, 1, 2], mean = 1.0
        self.assertAlmostEqual(mae(y, f), 1.0, places=6)

    def test_pearson_spearman(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        f_pos = np.array([2.0, 4.0, 6.0, 8.0])
        f_neg = np.array([8.0, 6.0, 4.0, 2.0])

        self.assertAlmostEqual(pearson(y, f_pos), 1.0, places=6)
        self.assertAlmostEqual(pearson(y, f_neg), -1.0, places=6)
        self.assertAlmostEqual(spearman(y, f_pos), 1.0, places=6)
        self.assertAlmostEqual(spearman(y, f_neg), -1.0, places=6)

        # Monotonic nonlinear
        f_mono = np.array([1.0, 10.0, 100.0, 1000.0])
        self.assertAlmostEqual(spearman(y, f_mono), 1.0, places=6)

        # Constant array
        f_const = np.array([2.0, 2.0, 2.0, 2.0])
        self.assertEqual(pearson(y, f_const), 0.0)

    def test_concordance_index_basic(self):
        # Perfect concordance
        y = np.array([1.0, 2.0, 3.0, 4.0])
        f_perf = np.array([10.0, 20.0, 30.0, 40.0])
        self.assertAlmostEqual(ci(y, f_perf), 1.0, places=6)

        # Perfect discordance
        f_disc = np.array([40.0, 30.0, 20.0, 10.0])
        self.assertAlmostEqual(ci(y, f_disc), 0.0, places=6)

    def test_concordance_index_ties(self):
        # Prediction ties count 0.5
        y = np.array([1.0, 2.0, 3.0])
        f = np.array([5.0, 5.0, 5.0])
        # 3 pairs where y_i > y_j: (2>1, 3>1, 3>2). For all, f_i == f_j, so score = 3 * 0.5 = 1.5. CI = 1.5/3 = 0.5.
        self.assertAlmostEqual(ci(y, f), 0.5, places=6)

        # Ground truth ties are excluded from denominator
        y_ties = np.array([1.0, 1.0, 2.0])
        f_ties = np.array([10.0, 20.0, 30.0])
        # pairs with y_i > y_j: (index 2 vs index 0), (index 2 vs index 1). Index 0 vs 1 has y_0 == y_1, excluded!
        # Both pairs have f[2] = 30 > 10 and 30 > 20, so concordant. CI = 2 / 2 = 1.0.
        self.assertAlmostEqual(ci(y_ties, f_ties), 1.0, places=6)

    def test_concordance_index_chunked(self):
        # Test >5000 elements branch
        rng = np.random.RandomState(42)
        y = rng.randn(5200)
        f = y + rng.randn(5200) * 0.5

        # Compare ci(y, f) with a small subsample direct test
        val = ci(y, f)
        self.assertTrue(0.5 < val < 1.0)

    def test_rm2_roy_formula(self):
        # Perfect match: rm2 should be exactly 1.0
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(rm2(y, y), 1.0, places=6)

        # Hand-calculated toy inputs
        y = np.array([1.0, 2.0, 3.0, 4.0])
        f = np.array([1.1, 1.9, 3.2, 3.8])

        r = np.corrcoef(y, f)[0, 1]
        r2 = r * r
        k = np.sum(y * f) / np.sum(f * f)
        y_mean = np.mean(y)
        ss_tot = np.sum((y - y_mean) ** 2)
        ss_res_0 = np.sum((y - k * f) ** 2)
        r02 = 1.0 - (ss_res_0 / ss_tot)
        expected = r2 * (1.0 - math.sqrt(abs(r2 - r02)))

        self.assertAlmostEqual(rm2(y, f), expected, places=6)

    def test_classification_metrics(self):
        # Perfect binary classification with 2-class logits
        y_true = np.array([0, 0, 1, 1])
        y_logits = np.array([
            [10.0, -10.0],
            [5.0, -5.0],
            [-5.0, 5.0],
            [-10.0, 10.0]
        ])
        metrics = compute_classification_metrics(y_true, y_logits)
        self.assertAlmostEqual(metrics['accuracy'], 1.0, places=6)
        self.assertAlmostEqual(metrics['aupr'], 1.0, places=6)
        self.assertAlmostEqual(metrics['auroc'], 1.0, places=6)
        self.assertAlmostEqual(metrics['f1'], 1.0, places=6)

        # 1D probabilities with a mistake
        y_true = np.array([0, 1, 0, 1])
        y_probs = np.array([0.1, 0.4, 0.2, 0.9])  # 2nd sample has prob 0.4 (< 0.5), predicted 0
        metrics = compute_classification_metrics(y_true, y_probs)
        # True: [0, 1, 0, 1], Pred: [0, 0, 0, 1] -> 3/4 correct = 0.75
        self.assertAlmostEqual(metrics['accuracy'], 0.75, places=6)


if __name__ == '__main__':
    unittest.main()
