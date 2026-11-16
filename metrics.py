"""
metrics.py
----------
Rigorous evaluation metrics for DTA (regression) and DTI (classification).
Implements:
  - Regression: RMSE, MAE, MSE, Pearson r, Spearman rho, Concordance Index (CI) with tie handling,
    and Modified Squared Correlation Coefficient r_m^2 (Roy et al., 2013).
  - Classification: AUROC, AUPR (positive-class softmax), Accuracy (0.5 threshold), and F1-score.
"""

import math
import numpy as np
from math import sqrt
from scipy import stats
from sklearn.metrics import precision_recall_curve, roc_curve, auc, f1_score, accuracy_score


def rmse(y_true, y_pred):
    """Root Mean Squared Error: sqrt(mean((y - f)^2))."""
    y = np.asarray(y_true, dtype=np.float64)
    f = np.asarray(y_pred, dtype=np.float64)
    return float(sqrt(np.mean((y - f) ** 2)))


def mse(y_true, y_pred):
    """Mean Squared Error: mean((y - f)^2)."""
    y = np.asarray(y_true, dtype=np.float64)
    f = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean((y - f) ** 2))


def mae(y_true, y_pred):
    """Mean Absolute Error: mean(|y - f|)."""
    y = np.asarray(y_true, dtype=np.float64)
    f = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y - f)))


def pearson(y_true, y_pred):
    """Pearson correlation coefficient r."""
    y = np.asarray(y_true, dtype=np.float64)
    f = np.asarray(y_pred, dtype=np.float64)
    if np.all(y == y[0]) or np.all(f == f[0]):
        return 0.0
    r = np.corrcoef(y, f)[0, 1]
    return float(r) if not np.isnan(r) else 0.0


def spearman(y_true, y_pred):
    """Spearman rank correlation coefficient rho."""
    y = np.asarray(y_true, dtype=np.float64)
    f = np.asarray(y_pred, dtype=np.float64)
    res = stats.spearmanr(y, f)[0]
    return float(res) if not np.isnan(res) else 0.0


def ci(y_true, y_pred):
    """
    Concordance Index (CI) with strict tie handling.
    
    Formula:
      CI = (1 / Z) * sum_{i, j: y_i > y_j} [ 1(f_i > f_j) + 0.5 * 1(f_i == f_j) ]
      where Z is the number of pairs with y_i > y_j.
      Ties on predictions f count 0.5. Pairs with y_i == y_j are excluded.
    
    Uses chunked vectorization to compute efficiently on large sets (e.g. 48,992 test pairs)
    without high memory footprint.
    """
    y = np.asarray(y_true, dtype=np.float64).flatten()
    f = np.asarray(y_pred, dtype=np.float64).flatten()
    n = len(y)
    if n <= 1:
        return 0.0

    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]

    if n <= 5000:
        diff_y = y[:, None] - y[None, :]
        diff_f = f[:, None] - f[None, :]
        valid = diff_y > 0
        z = np.sum(valid)
        if z == 0:
            return 0.0
        concordant = np.sum((diff_f > 0) & valid)
        ties = np.sum((diff_f == 0) & valid)
        return float((concordant + 0.5 * ties) / z)
    else:
        z = 0
        score = 0.0
        chunk_size = 2000
        for start_i in range(0, n, chunk_size):
            end_i = min(start_i + chunk_size, n)
            yi = y[start_i:end_i, None]
            fi = f[start_i:end_i, None]
            yj = y[None, :end_i]
            fj = f[None, :end_i]

            diff_y = yi - yj
            diff_f = fi - fj

            row_idx = np.arange(start_i, end_i)[:, None]
            col_idx = np.arange(end_i)[None, :]
            valid = (row_idx > col_idx) & (diff_y > 0)

            z += np.sum(valid)
            score += np.sum((diff_f > 0) & valid) + 0.5 * np.sum((diff_f == 0) & valid)

        return float(score / z) if z > 0 else 0.0


def rm2(y_true, y_pred):
    """
    Modified squared correlation coefficient r_m^2 (Roy et al., 2013).
    
    Exact Formula:
        r_m^2 = r^2 * (1 - sqrt(|r^2 - r_0^2|))
    where:
        r^2 = (pearson_r)^2
        k = sum(y * f) / sum(f^2)   (slope through origin)
        r_0^2 = 1 - sum((y - k * f)^2) / sum((y - y_mean)^2)
    """
    y = np.asarray(y_true, dtype=np.float64).flatten()
    f = np.asarray(y_pred, dtype=np.float64).flatten()

    if np.all(y == y[0]) or np.all(f == f[0]):
        return 0.0

    r = np.corrcoef(y, f)[0, 1]
    if np.isnan(r):
        return 0.0
    r2 = float(r * r)

    f_sq_sum = float(np.sum(f * f))
    if f_sq_sum == 0.0:
        return 0.0
    k = float(np.sum(y * f) / f_sq_sum)

    y_mean = float(np.mean(y))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    if ss_tot == 0.0:
        return 0.0

    ss_res_0 = float(np.sum((y - k * f) ** 2))
    r02 = 1.0 - (ss_res_0 / ss_tot)

    diff = abs(r2 - r02)
    return float(r2 * (1.0 - math.sqrt(diff)))


def compute_classification_metrics(y_true, y_scores):
    """
    Compute AUROC, AUPR, Accuracy, and F1-score from positive-class probabilities.
    Threshold for binary classification is 0.5.
    """
    y_true = np.asarray(y_true, dtype=np.int64).flatten()
    y_scores = np.asarray(y_scores, dtype=np.float64)

    if y_scores.ndim == 2:
        # Softmax along class dimension
        exp_scores = np.exp(y_scores - np.max(y_scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        y_prob = probs[:, 1]
    else:
        y_prob = y_scores.flatten()

    y_pred = (y_prob >= 0.5).astype(int)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    aupr = float(auc(recall, precision))

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auroc = float(auc(fpr, tpr))

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return {
        'accuracy': acc,
        'aupr': aupr,
        'auroc': auroc,
        'f1': f1,
    }
