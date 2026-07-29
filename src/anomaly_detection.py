"""
Exploratory robustness check (not a formal RQ): an unsupervised anomaly
detector trained on the full 591-feature set, run in parallel with the
supervised reduced model from RQ3.

Rationale (see docs/synopsis.docx, "Robustness Consideration"): a genuinely
novel failure mode has never appeared in the labeled fail class, so no
supervised, label-driven feature-selection method -- SHAP-guided or
otherwise -- can ever be trained to catch it. Isolation Forest (Liu, Ting,
& Zhou, 2008) does not use the label at all, so it can flag "this wafer
looks unusual" even when nothing in the top-10 reduced feature set explains
why. This is a coverage safety net, not a replacement for the interpretable
reduced model.
"""

import pandas as pd
from sklearn.ensemble import IsolationForest


def fit_anomaly_detector(
    X: pd.DataFrame, contamination: float = 0.066, random_state: int = 42
) -> IsolationForest:
    """
    contamination defaults to SECOM's observed fail rate (~6.6%) as a
    starting assumption for the expected proportion of anomalies; this
    should be tuned/reported explicitly rather than left as a silent default.
    """
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(X)
    return model


def score_wafers(model: IsolationForest, X: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with an anomaly score (lower = more anomalous) and
    a binary flag, indexed the same as X, so it can be joined against the
    reduced supervised model's predictions for comparison.
    """
    scores = model.decision_function(X)
    flags = model.predict(X)  # -1 = anomaly, 1 = normal
    return pd.DataFrame(
        {
            "anomaly_score": scores,
            "is_anomaly": flags == -1,
        },
        index=X.index,
    )


def compare_against_supervised_flags(
    anomaly_df: pd.DataFrame, y_true: pd.Series, y_pred_supervised: pd.Series
) -> dict:
    """
    Reports how often the unsupervised detector flags a wafer that the
    supervised reduced model does NOT flag as fail -- these are the cases
    most relevant to the "novel failure mode" concern, worth manual review.
    """
    supervised_missed = (y_true == 1) & (y_pred_supervised == 0)
    caught_by_anomaly_detector = supervised_missed & anomaly_df["is_anomaly"]

    return {
        "supervised_false_negatives": int(supervised_missed.sum()),
        "of_those_flagged_by_anomaly_detector": int(caught_by_anomaly_detector.sum()),
        "anomaly_detector_recall_on_missed_fails": (
            float(caught_by_anomaly_detector.sum() / supervised_missed.sum())
            if supervised_missed.sum() > 0
            else None
        ),
    }
