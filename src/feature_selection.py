"""
Feature-selection helpers for RQ3: SHAP-guided reduction to ~10 features,
done in a way that avoids two common pitfalls:

1. Information leakage: SHAP importance must be computed only on each
   training fold, never on the full dataset before splitting, or the
   "reduced" model has effectively seen the test labels twice.
2. Single-method bias: SHAP alone can be unstable across correlated
   features. An independent LASSO-based selection is used as a
   cross-check; convergence between the two methods is evidence the
   reduced set reflects real signal.

See docs/synopsis.docx, "Solution to RQ3" for the full methodology.
"""

from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
import shap
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def extract_positive_class_shap(shap_values) -> np.ndarray:
    """
    Normalize shap.TreeExplainer's output to a plain (n_samples, n_features)
    array of positive-class SHAP values, regardless of which output
    convention the installed shap version uses for binary classification:

    1. A list of 2 arrays, one per class (older shap versions) -- index 1
       is the positive class.
    2. A single 3D array of shape (n_samples, n_features, n_classes)
       (newer shap versions, observed in practice for RandomForestClassifier
       and the reason this function exists -- the older list-only check
       caused a real ValueError: "Data must be 1-dimensional, got ndarray
       of shape (432, 2)").
    3. A single 2D array of shape (n_samples, n_features) already (some
       models/versions return this directly for binary classification).
    """
    if isinstance(shap_values, list):
        return shap_values[1]
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        return shap_values[:, :, 1]
    return shap_values


def cluster_correlated_features(X: pd.DataFrame, corr_threshold: float = 0.85) -> dict:
    """
    Hierarchically cluster features by pairwise correlation so that
    near-duplicate sensors (e.g. three temperature probes on one chamber)
    compete as a single cluster rather than splitting SHAP credit.

    Returns a dict mapping feature name -> cluster id. Columns dropped as
    fold-locally-constant (see below) simply won't appear as keys -- callers
    (representative_per_cluster) already only iterate over what's present,
    so this is safe without further changes downstream.

    Five layers of defense, in order, against two distinct numerical
    failures observed in practice on real SECOM folds (a NaN crash in
    linkage(), then a separate negative-distance crash in fcluster()):
    1. Constant-column dropping (primary NaN fix): a column can pass the
       GLOBAL variance screening (over all 1,567 rows) but still be
       exactly constant within one CV fold's ~80% training subset --
       correlation is undefined (0/0) for a constant column. Dropping such
       columns before computing correlations is more principled than
       keeping them with a patched-up correlation value.
    2. fillna(0) + forced diagonal (NaN safety net): catches any remaining
       NaN short of exact constancy.
    3. np.isfinite check before linkage() (NaN last-resort net).
    4. Clamping distance to [0, None] before linkage(): guards against the
       input distance matrix containing tiny negative values.
    5. Clamping Z's merge-height column to [0, None] after linkage(),
       before fcluster(): this is the layer that actually matters here --
       confirmed by deliberately constructing a Z matrix with a tiny
       negative merge height and reproducing scipy's exact "Linkage 'Z'
       contains negative distances" error, then confirming this clamp
       resolves it. Average-linkage's recursive merge-height computation
       can produce a tiny negative value from floating-point rounding
       across many merges even when every input distance was already
       non-negative, so layer 4 alone is not guaranteed sufficient -- this
       is why both layers are present rather than just the input-side one.
    """
    constant_cols = X.columns[X.nunique() <= 1]
    X_varying = X.drop(columns=constant_cols) if len(constant_cols) else X

    corr = X_varying.corr().abs().fillna(0.0)
    corr_values = corr.values.copy()  # .values can be a read-only view; copy before in-place mutation
    np.fill_diagonal(corr_values, 1.0)
    distance = 1 - corr_values
    distance = np.clip(distance, 0, None)  # layer 4: guard the input side

    condensed = squareform(distance, checks=False)
    if not np.isfinite(condensed).all():
        condensed = np.nan_to_num(condensed, nan=0.0, posinf=1.0, neginf=0.0)

    Z = linkage(condensed, method="average")
    Z[:, 2] = np.clip(Z[:, 2], 0, None)  # layer 5: guard linkage's own output

    cluster_ids = fcluster(Z, t=1 - corr_threshold, criterion="distance")
    return dict(zip(X_varying.columns, cluster_ids))


def representative_per_cluster(X: pd.DataFrame, clusters: dict, importances: pd.Series) -> list[str]:
    """Keep the highest-importance feature from each correlation cluster."""
    cluster_to_features: dict[int, list[str]] = {}
    for feature, cid in clusters.items():
        cluster_to_features.setdefault(cid, []).append(feature)

    representatives = []
    for cid, features in cluster_to_features.items():
        best = max(features, key=lambda f: importances.get(f, 0.0))
        representatives.append(best)
    return representatives


def nested_cv_shap_selection(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory,
    k: int = 10,
    n_folds: int = 5,
    corr_threshold: float = 0.85,
    random_state: int = 42,
) -> dict:
    """
    For each CV fold: cluster correlated features on the training split,
    fit `model_factory()` on the training split, compute SHAP importance
    on the training split only, pick the top-k cluster representatives,
    and evaluate on the held-out fold.

    Returns a dict with:
        - per_fold_features: list of selected feature lists (one per fold)
        - feature_stability: Counter of how often each feature was selected
        - watch_list: next-ranked features that narrowly missed top-k, per fold
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    per_fold_features = []
    watch_lists = []

    for train_idx, _ in skf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]

        clusters = cluster_correlated_features(X_train, corr_threshold)

        model = model_factory()
        model.fit(X_train, y_train)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
        sv = extract_positive_class_shap(shap_values)
        mean_abs_shap = pd.Series(np.abs(sv).mean(axis=0), index=X_train.columns)

        candidates = representative_per_cluster(X_train, clusters, mean_abs_shap)
        ranked = mean_abs_shap.loc[candidates].sort_values(ascending=False)

        top_k = ranked.index[:k].tolist()
        watch_list = ranked.index[k : k + 25].tolist()

        per_fold_features.append(top_k)
        watch_lists.append(watch_list)

    stability = Counter(f for fold in per_fold_features for f in fold)

    return {
        "per_fold_features": per_fold_features,
        "feature_stability": stability,
        "watch_list_by_fold": watch_lists,
    }


def lasso_cross_check(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> list[str]:
    """
    Independent, methodologically different feature-selection cross-check.
    Fits L1-regularized logistic regression and returns the features with
    non-zero coefficients (Tibshirani, 1996).

    Features are standardized first -- this is not optional. L1-penalized
    liblinear on raw, unscaled sensor features (which here span wildly
    different numeric ranges) converges extremely slowly or effectively
    hangs: 50 fits (Cs=10 x cv=5) each fighting to converge on unscaled data
    caused a real 1,800s timeout in practice before this fix. Standardizing
    first is the actual fix, not a larger max_iter or fewer Cs candidates.

    scoring="roc_auc" is also not optional. LogisticRegressionCV's default
    internal scoring is plain accuracy, and on this ~93.4%-majority-class
    dataset an all-zero-coefficient (majority-class-only) model already
    scores ~93% accuracy trivially -- in practice this caused the CV
    selection to pick the most-regularized candidate and return zero
    selected features every time. AUC-ROC rewards actual discrimination
    between classes instead of raw accuracy, which fixes this.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg_cv", LogisticRegressionCV(
            Cs=10,
            cv=5,
            penalty="l1",
            solver="liblinear",
            class_weight="balanced",
            random_state=random_state,
            max_iter=1000,  # lower than before on purpose: with scaled data this
                             # is plenty, and a lower ceiling fails fast/loud if
                             # something is still wrong, instead of hanging silently.
            scoring="roc_auc",
        )),
    ])
    pipeline.fit(X, y)
    model = pipeline.named_steps["logreg_cv"]
    coefs = pd.Series(model.coef_.ravel(), index=X.columns)
    return coefs[coefs != 0].abs().sort_values(ascending=False).index.tolist()


def agreement_report(shap_features: list[str], lasso_features: list[str]) -> dict:
    """
    Simple overlap report between SHAP-selected and LASSO-selected features.
    High overlap is corroborating evidence the reduced feature set reflects
    real signal rather than an artifact of one importance measure.
    """
    shap_set, lasso_set = set(shap_features), set(lasso_features)
    overlap = shap_set & lasso_set
    return {
        "shap_only": sorted(shap_set - lasso_set),
        "lasso_only": sorted(lasso_set - shap_set),
        "overlap": sorted(overlap),
        "jaccard_similarity": len(overlap) / len(shap_set | lasso_set) if (shap_set | lasso_set) else 0.0,
    }
