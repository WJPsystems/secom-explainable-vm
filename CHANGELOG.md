# Changelog

Tracks what changed in each delivered version of this repo. Individual files
inside the repo are not renamed/suffixed per version -- Git's own commit
history is the source of truth for file-level changes. This changelog and
each notebook's "Notebook version" marker (in its first cell) exist so you
can tell, at a glance, whether the copy you're looking at in Colab/GitHub is
current, without needing to diff files by hand.

## v4 -- 2026-07-29

- Replaced the hidden `.gitkeep` placeholder files in `data/raw/`,
  `data/model_ready/`, and `data/artifacts/` with visible `PLACEHOLDER.md`
  files. `.gitkeep` starts with a dot, so most file pickers/browsers hide it
  by default when selecting a folder to drag-and-drop -- this is why the
  `data/` folder was silently dropped from a prior upload. The new files are
  visible and also explain what populates each folder and when.

## v3 -- 2026-07-28

- **Correctness fix:** `02_modeling_rq1.ipynb` now saves its actual winning
  model(s) (`data/artifacts/`, via new `src/artifacts.py`) instead of every
  downstream notebook silently re-deriving its own assumption. Two artifacts
  are saved -- the best tree ensemble specifically (for RQ2/RQ3's
  TreeExplainer-based SHAP work) and the true overall winner (for RQ1's own
  "does TabNet beat trees" question) -- since these can legitimately differ.
- `03_shap_analysis_rq2.ipynb` no longer retrains its own fresh XGBoost; it
  loads RQ1's actual saved best-tree model.
- `04_feature_reduction_rq3.ipynb`'s nested-CV feature selection now uses
  RQ1's actual winning tree architecture (RF or XGBoost) instead of a
  hardcoded XGBoost assumption.
- Colab setup cell (all 7 notebooks) now runs `git pull` if the repo is
  already cloned (was previously skipped, risking a stale local copy), and
  prints the current commit hash + timestamp so you can verify against
  GitHub at a glance.

## v2 -- 2026-07-28

- Added `00_run_all.ipynb`: executes `01`-`06` in sequence, one command,
  with a pass/fail run summary. Handles the fact that each notebook's own
  live-Colab-frontend HTML export cell doesn't work in unattended/batch
  execution, by exporting HTML separately via plain `nbconvert` against the
  already-executed files instead.
- README updated with a "Running everything at once" usage section.

## v1 -- 2026-07-28

- `02_modeling_rq1.ipynb` built out with real analysis: 4 models (LogReg, RF,
  XGBoost, TabNet), stratified 5-fold CV with out-of-fold predictions,
  imbalance correction per model, results table (AUC-ROC, AUC-PR, BER, MCC),
  comparison plot.
- Added "Notebook version" marker to every notebook's intro cell.
- Established this changelog and the versioned-delivery naming convention.

## Pre-v1 (unversioned)

Everything prior to this point: repo scaffold, `01_data_screening.ipynb`
(real SECOM fetch + screening), `src/` modules (`preprocessing.py`,
`feature_selection.py`, `explainability.py`, `anomaly_detection.py`,
`metrics.py`, `fetch_data.py`), Colab setup cells, HTML export cells, and the
synopsis document itself. Not individually versioned since the naming
convention didn't exist yet.
