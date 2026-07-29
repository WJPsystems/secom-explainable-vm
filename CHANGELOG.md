# Changelog

Tracks what changed in each delivered version of this repo. Individual files
inside the repo are not renamed/suffixed per version -- Git's own commit
history is the source of truth for file-level changes. This changelog and
each notebook's "Notebook version" marker (in its first cell) exist so you
can tell, at a glance, whether the copy you're looking at in Colab/GitHub is
current, without needing to diff files by hand.

## v8 -- 2026-07-29

- **Session-marker optimization**, requested a few turns back and delayed
  until `04`'s timeout bug was fixed: the Colab setup cell now writes
  `/content/.secom_setup_done` after the first successful `git pull` +
  `pip install` in a session, and subsequent notebooks (e.g. `01`-`06`
  opened right after `00_run_all` already set things up) check for it and
  skip both steps if present. A brand new session (or a fresh clone) always
  runs the full setup regardless, since the marker lives only in `/content`,
  never in the repo -- verified with a standalone simulation of the control
  flow across 4 scenarios (first run, 2nd/3rd run same session, new session
  after restart) before shipping, not just a read-through.
- Still prints the commit hash every time either way, so version
  verification isn't affected by the skip.

## v7 -- 2026-07-29

- **Real bug fix: `04_feature_reduction_rq3.ipynb` timed out after 1,800s**
  inside its LASSO cross-check step. Root cause: `lasso_cross_check()` in
  `src/feature_selection.py` fit L1-penalized logistic regression
  (`liblinear` solver) directly on raw, unscaled sensor features -- with
  432 features on wildly different numeric scales, this converges
  extremely slowly (50 fits: `Cs=10 x cv=5`, each fighting to converge).
  Fixed by standardizing features first via an sklearn `Pipeline`
  (`StandardScaler` -> `LogisticRegressionCV`); also lowered `max_iter`
  from 5000 to 1000 so a future regression fails fast and loud instead of
  hanging silently for 30 minutes.
- Same latent issue existed in `02_modeling_rq1.ipynb`'s baseline
  `LogisticRegression` (didn't time out there, but same root cause) --
  fixed identically, wrapped in a `StandardScaler` Pipeline in both
  `make_models()` and `refit_full()`. `predict_proba`/`.fit()` interfaces
  are unchanged, so no other code needed to change.
- `05` and `06` were correctly never exported to HTML after `04`'s failure
  -- `00_run_all.ipynb` intentionally stops the whole run on first failure
  and only exports notebooks that actually completed; this was expected
  behavior, not a separate bug.

## v6 -- 2026-07-29

- **Free speedups, no effect on results:** added `n_jobs=-1` to every
  RandomForest and XGBoost instantiation (`02`, `04`) and to `03`'s
  `permutation_importance` call, so these use all available CPU cores
  instead of a single one by default. Same computation, same numbers,
  faster wall-clock time.
- All 7 notebook version markers bumped to v6 and independently re-verified
  by re-reading file content after the edit (not just trusting the script's
  print output, per the v5 lesson).

## v5 -- 2026-07-29

- **Bug fix in the version-marker system itself.** The v3 "bump to v3"
  script used `line.startswith("**Notebook version:**")`, which never
  matched because the marker line actually started with a leading newline
  character from how it was first inserted -- so 6 of 7 notebooks silently
  kept showing "v1" in their intro cell even though their actual code was
  correctly updated to v3. The script also printed a success message
  unconditionally, regardless of whether a match was found, which is why
  this went unnoticed until directly verified against the live GitHub repo.
  Fixed with a substring check instead, and verified this time by
  independently re-reading the file content after the edit, not just
  trusting the script's own print statement.
- No functional/code changes in this version -- only the version-marker
  text itself was wrong; the underlying pipeline logic was already correct
  as of v3.

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
