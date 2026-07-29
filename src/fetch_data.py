"""
Fetch the SECOM dataset and save it locally under data/raw/.

Usage:
    python src/fetch_data.py

Requires internet access. Two methods are tried, in order:

1. Direct download of UCI's raw data files (secom.data, secom_labels.data).
   This is the primary method: it depends only on pandas' ability to read a
   CSV/whitespace-delimited file from a URL, not on any third-party wrapper
   package, and has proven the most reliable in practice.
2. ucimlrepo's fetch_ucirepo(id=179), as a fallback. This wrapper has been
   observed to silently return None for .data.features on some versions/
   environments (a known flakiness, not something this project caused), so
   it is not used as the primary path.

If both fail, the SECOM files can be downloaded manually from
https://archive.ics.uci.edu/dataset/179/secom and placed in data/raw/ as
secom_features.csv and secom_labels.csv.
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"
LABELS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data"


def _fetch_direct() -> tuple[pd.DataFrame, pd.Series]:
    print("Downloading SECOM raw data files directly from UCI...")
    features = pd.read_csv(FEATURES_URL, sep=r"\s+", header=None, na_values="NaN")
    features.columns = [f"Sensor_{i + 1}" for i in range(features.shape[1])]

    labels_raw = pd.read_csv(LABELS_URL, sep=r"\s+", header=None, names=["label", "timestamp"])
    labels = labels_raw["label"]

    return features, labels


def _fetch_via_ucimlrepo() -> tuple[pd.DataFrame, pd.Series]:
    from ucimlrepo import fetch_ucirepo

    print("Fetching SECOM (id=179) via ucimlrepo...")
    secom = fetch_ucirepo(id=179)

    features = secom.data.features
    labels = secom.data.targets

    if features is None or labels is None:
        raise RuntimeError(
            "ucimlrepo returned None for features or labels -- known flakiness "
            "with this package, falling back is not possible from here."
        )

    return features, labels.squeeze("columns") if hasattr(labels, "squeeze") else labels


def fetch_secom() -> None:
    features_path = RAW_DIR / "secom_features.csv"
    labels_path = RAW_DIR / "secom_labels.csv"

    try:
        features, labels = _fetch_direct()
    except Exception as direct_exc:
        print(f"Direct download failed ({direct_exc!r}); trying ucimlrepo as a fallback...")
        try:
            features, labels = _fetch_via_ucimlrepo()
        except Exception as uci_exc:
            raise SystemExit(
                "Both fetch methods failed. Direct download error: "
                f"{direct_exc!r}. ucimlrepo error: {uci_exc!r}. "
                "You can download the files manually from "
                "https://archive.ics.uci.edu/dataset/179/secom and place them "
                "in data/raw/ as secom_features.csv and secom_labels.csv."
            ) from uci_exc

    features.to_csv(features_path, index=False)
    labels.to_csv(labels_path, index=False)

    print(f"Saved {features.shape[0]} rows x {features.shape[1]} features -> {features_path}")
    print(f"Saved labels -> {labels_path}")
    print(f"Label distribution:\n{labels.value_counts()}")


if __name__ == "__main__":
    fetch_secom()
