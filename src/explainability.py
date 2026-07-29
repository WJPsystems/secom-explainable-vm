"""
Accumulated Local Effects (ALE) for the top-k SHAP-ranked features (RQ2).

Complementary to SHAP, not a substitute: SHAP answers "why did the model
flag this specific wafer" (local); ALE answers "on average, how does fail
probability move as one sensor's reading changes" (global). ALE is used in
place of Partial Dependence Plots (PDP) because PDP assumes feature
independence when averaging, which is a poor assumption given the
correlation clustering already required elsewhere in this pipeline to
handle SECOM's known multicollinearity (Apley & Zhu, 2020).

See docs/synopsis.docx, "Solution to RQ2" for the full write-up.
"""

import numpy as np
import pandas as pd


def compute_ale(
    model, X: pd.DataFrame, feature: str, n_bins: int = 20, predict_fn=None
) -> pd.DataFrame:
    """
    Compute a 1D ALE curve for a single feature.

    model: a fitted classifier with a .predict_proba(X) method (or pass
        predict_fn explicitly for models with a different interface, e.g.
        model.predict for a raw probability output).
    Returns a DataFrame with bin midpoints and the accumulated local effect
    at each midpoint, centered so the curve's weighted mean is zero (the
    standard ALE convention -- values are effects relative to the average
    prediction, not absolute probabilities).
    """
    if predict_fn is None:
        predict_fn = lambda data: model.predict_proba(data)[:, 1]

    x = X[feature].values
    quantiles = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    if len(quantiles) < 3:
        raise ValueError(
            f"Feature '{feature}' has too few unique values for {n_bins} ALE bins; "
            "reduce n_bins."
        )

    bin_idx = np.clip(np.digitize(x, quantiles[1:-1], right=True), 0, len(quantiles) - 2)

    local_effects = np.zeros(len(quantiles) - 1)
    counts = np.zeros(len(quantiles) - 1)

    for b in range(len(quantiles) - 1):
        mask = bin_idx == b
        if not mask.any():
            continue
        X_lo = X.loc[mask].copy()
        X_hi = X.loc[mask].copy()
        X_lo[feature] = quantiles[b]
        X_hi[feature] = quantiles[b + 1]

        pred_lo = predict_fn(X_lo)
        pred_hi = predict_fn(X_hi)

        local_effects[b] = np.mean(pred_hi - pred_lo)
        counts[b] = mask.sum()

    accumulated = np.concatenate([[0], np.cumsum(local_effects)])
    bin_centers = (quantiles[:-1] + quantiles[1:]) / 2

    # Center the curve so its weighted mean (over observed data density) is zero
    midpoint_values = (accumulated[:-1] + accumulated[1:]) / 2
    weighted_mean = np.average(midpoint_values, weights=np.maximum(counts, 1))
    accumulated_centered = accumulated - weighted_mean

    return pd.DataFrame(
        {
            "bin_edge": quantiles,
            "accumulated_local_effect": accumulated_centered,
        }
    )


def ale_report_for_top_features(
    model, X: pd.DataFrame, top_features: list[str], n_bins: int = 20
) -> dict:
    """
    Convenience wrapper: compute ALE curves for each of RQ3's top-k
    SHAP-selected features, for inclusion alongside the SHAP importance
    ranking in the RQ2/RQ3 write-up.
    """
    return {feat: compute_ale(model, X, feat, n_bins=n_bins) for feat in top_features}
