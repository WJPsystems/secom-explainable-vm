"""
Lightweight save/load helpers for passing real results between notebooks.

Without this, each notebook silently re-derives its own assumptions instead
of using what an earlier notebook actually found -- e.g. 03_shap_analysis_rq2
retraining its own fresh XGBoost instead of explaining RQ1's actual winning
model. If RQ1's real results show a different model winning, notebooks that
don't load from here would quietly explain/reduce-features-for the wrong
model without any error or warning.
"""

import json
from pathlib import Path

import joblib

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "data" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def save_model(name: str, model) -> None:
    """Save a scikit-learn/XGBoost model via joblib. Not for TabNet -- use
    TabNet's own .save_model()/.load_model() instead (see save_tabnet_path)."""
    joblib.dump(model, ARTIFACTS_DIR / f"{name}.joblib")


def load_model(name: str):
    path = ARTIFACTS_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"No saved model at {path} -- run the notebook that produces it first "
            f"(check CHANGELOG.md / the notebook docstrings for which one)."
        )
    return joblib.load(path)


def tabnet_save_path(name: str) -> str:
    """TabNet's .save_model() appends .zip itself; return the base path it expects."""
    return str(ARTIFACTS_DIR / name)


def tabnet_load_path(name: str) -> str:
    path = ARTIFACTS_DIR / f"{name}.zip"
    if not path.exists():
        raise FileNotFoundError(
            f"No saved TabNet model at {path} -- run 02_modeling_rq1.ipynb first."
        )
    return str(path)


def save_json(name: str, data: dict) -> None:
    with open(ARTIFACTS_DIR / f"{name}.json", "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(name: str) -> dict:
    path = ARTIFACTS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No saved artifact at {path} -- run the notebook that produces it first "
            f"(check CHANGELOG.md / the notebook docstrings for which one)."
        )
    with open(path) as f:
        return json.load(f)
