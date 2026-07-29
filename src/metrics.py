"""
Evaluation metrics for imbalanced binary classification, matching the formulae
in the capstone synopsis (docs/synopsis.docx, Analytic Approach section).

"Positive" class = wafer fail throughout.
"""

import numpy as np
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score


def balanced_error_rate(y_true, y_pred) -> float:
    """BER = 0.5 * (FPR + FNR). Matches the metric used in published SECOM baselines."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    return 0.5 * (fpr + fnr)


def mcc(y_true, y_pred) -> float:
    """Matthews Correlation Coefficient (Chicco & Jurman, 2020)."""
    return matthews_corrcoef(y_true, y_pred)


def auc_roc(y_true, y_scores) -> float:
    return roc_auc_score(y_true, y_scores)


def summarize(y_true, y_pred, y_scores=None) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    result = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ber": balanced_error_rate(y_true, y_pred),
        "mcc": mcc(y_true, y_pred),
    }
    if y_scores is not None:
        result["auc_roc"] = auc_roc(y_true, y_scores)
    return result
