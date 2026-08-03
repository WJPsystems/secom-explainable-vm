# Changelog

Tracks what changed in each delivered version of this repo. Individual files
inside the repo are not renamed/suffixed per version -- Git's own commit
history is the source of truth for file-level changes. This changelog and
each notebook's "Notebook version" marker (in its first cell) exist so you
can tell, at a glance, whether the copy you're looking at in Colab/GitHub is
current, without needing to diff files by hand.

## v20 -- 2026-08-02

- **Real bug found: `data/model_ready/` was never populated by a v18 run**
  (only `raw/` and `artifacts/` appeared in the downloaded zip). Root
  cause: `run_screening_pipeline()` -- the only code that writes to
  `data/model_ready/` -- was only reachable via
  `if __name__ == "__main__":`, i.e. running `python src/preprocessing.py`
  directly, which nothing in this notebook-based pipeline ever does. This
  was architecturally orphaned from the start, not a "hasn't run yet" gap.
- **Fixed by calling `run_screening_pipeline()` explicitly in
  `01_data_screening.ipynb`.** Confirmed this doesn't introduce a second,
  inconsistent train/test split: `stratified_split()` uses the identical
  parameters (`test_size=0.2, random_state=42, stratify=y`) as RQ1's
  index-based holdout split used everywhere else in the pipeline -- this
  materializes that same split as CSV files for direct inspection, it
  doesn't compete with it.
- Verified the underlying screen/split/write logic directly (synthetic
  data, temp directory) before wiring it in.
- `00_run_all.ipynb`'s existing data-download cell already iterates over
  all three `data/` subfolders, so no changes needed there -- it will pick
  up the newly-populated `model_ready/` files automatically on the next run.

## v19 -- 2026-07-29

- **Real bug found from `05`'s first run against actual SECOM data**: the
  MLP control's training loss exploded from 2.22 to 174.23 and then stayed
  *exactly* flat for 100+ epochs -- a saturated/dead network, not a trained
  model. This silently produced garbage explanations: MLP's SHAP-vs-
  Integrated-Gradients agreement came back as exactly 0.0000, and a
  `ConstantInputWarning` confirmed the model's output barely varied with
  input at all. This invalidated the RQ4 conclusion drawn from that run,
  since the "control" the comparison depends on was broken.
- **Root cause, confirmed by direct reproduction**: the MLP was trained on
  raw, unscaled SECOM features (which span wildly different numeric
  ranges -- the same issue that broke LASSO earlier in this project).
  TabNet handles this internally; a hand-built PyTorch MLP does not.
  Reproduced the exact failure pattern (loss explodes and flatlines
  immediately) with synthetic SECOM-scale features, then confirmed the fix
  resolves it on the same adversarial data (MLP reaches 95.6% train
  accuracy with smoothly decreasing loss instead).
- **Fix, three parts**: (1) `StandardScaler` fit on the training split,
  applied to both train and test -- not fit separately on test, which
  would leak test statistics into the transform; (2) switched from
  `Sigmoid()` + `BCELoss` to raw logits + `BCEWithLogitsLoss` (more
  numerically stable) and lowered the learning rate from 0.01 to 0.001;
  (3) added gradient clipping as a safety net against future divergence,
  plus an explicit train-accuracy diagnostic with a printed warning if it's
  at or below chance, so this exact failure mode is caught immediately in
  future runs instead of requiring someone to notice a suspiciously flat
  loss log.
- **Downstream cells updated for consistency**: Integrated Gradients and
  the MLP's SHAP explainer now both operate on the same scaled inputs the
  MLP was actually trained on (using the same fitted scaler, not a fresh
  one per set), and both are wrapped with an explicit `torch.sigmoid()`
  now that the model outputs raw logits -- keeping IG and SHAP on the same
  probability scale for a fair comparison.

## v18 -- 2026-07-29

- **Synopsis updated with the real repo link**
  (https://github.com/WJPsystems/secom-explainable-vm/tree/main), replacing
  the "link to be added" placeholder in the Open Access paragraph.
- **Figure 1's folder tree was stale** (still showed `.gitkeep` instead of
  `PLACEHOLDER.md`, missing `data/artifacts/`, `00_run_all.ipynb`,
  `artifacts.py`, `CHANGELOG.md`) -- refreshed to match the actual current
  repository structure.
- **Open Access paragraph rewritten to be honest about current state**:
  previously claimed "the cleaned dataset... are published," which wasn't
  accurate -- `data/raw/`, `data/model_ready/`, and `data/artifacts/`
  contain placeholder files, not the actual generated data, since Colab
  sessions are ephemeral and nothing syncs back to GitHub automatically.
  Now states this explicitly rather than overclaiming reproducibility that
  doesn't exist yet.
- **Added a real fix, not just a disclosure**: `00_run_all.ipynb` now has a
  final step that zips up everything actually generated in `data/raw/`,
  `data/model_ready/`, and `data/artifacts/` (excluding the placeholder
  files themselves) and triggers a download, so the real data/artifacts can
  be uploaded to GitHub after a run -- tested the filtering logic directly
  (real files included, placeholders correctly excluded) before shipping.

## v17 -- 2026-07-29

- **`04` now persists its own results** (`rq3_summary.json`: `final_features`,
  `watch_list`, `tuned_threshold`, LASSO Jaccard overlap) -- a real gap found
  while building `05`/`06`: `04` previously saved nothing at all, so neither
  downstream notebook could have loaded RQ3's actual reduced feature set or
  tuned threshold even if their own code were otherwise correct.
- **`05_attention_comparison_rq4.ipynb` built out with real code**: trains
  TabNet and a class-weighted PyTorch MLP control on RQ3's actual reduced
  feature set, extracts TabNet's native `.explain()` attention output and
  the MLP's Integrated Gradients attributions (captum), computes SHAP via
  `KernelExplainer` for both (neither is a tree model, so `TreeExplainer`
  doesn't apply), and reports per-instance Spearman agreement + top-3
  feature-set agreement between each model's SHAP values and its own
  intrinsic explanation -- RQ4's actual question, answered with real
  computed numbers instead of a stub.
- **`06_anomaly_safety_net.ipynb` built out with real code**: fits an
  Isolation Forest on the full 432-feature set (train portion of the
  holdout split), scores the held-out test portion, and compares against
  RQ3's actual reduced model at its actual tuned threshold -- reporting
  concretely how many real failures the supervised model missed and how
  many of those the unsupervised safety net still caught.
- **New `per_instance_agreement()`** in `src/explainability.py`, verified
  against known-answer cases (identical rankings -> correlation ~1.0 and
  100% top-k agreement; independent random rankings -> correlation ~0)
  before use.
- Verified the full artifact chain end-to-end twice: `02`-save ->
  `04`-load/save -> `06`-load (Isolation Forest comparison), and
  `02`-save -> `04`-load/save -> `05`-load (TabNet/MLP/SHAP/IG comparison)
  -- both using the actual `src/` modules with synthetic data, not mocks.
  **Neither `05` nor `06` has been run against the real SECOM dataset yet**
  -- that confirmation is still outstanding (see README Status).
- **Synopsis (`docs/synopsis.docx`) updated with a new "Preliminary
  Results" section**, reporting real RQ1/RQ2/RQ3 findings (model comparison
  table, SHAP vs. permutation importance top-10 with real values and their
  2/10 overlap, feature stability table, and the full-vs-reduced comparison
  at both the default and tuned thresholds) in place of the
  planned-methodology framing alone.
- README status checklist updated to reflect what's actually confirmed on
  real data (`01`-`04`) versus what's code-complete but synthetic-tested
  only (`05`/`06`) -- deliberately not marked complete until run for real,
  per the same "verify before claiming done" discipline used throughout
  this project.

## v16 -- 2026-07-29

- **Threshold calibration implemented** (Step 5 in `04_feature_reduction_rq3.ipynb`),
  addressing the confirmed finding that both full and reduced models predict
  "pass" for every held-out wafer at the default 0.5 threshold (identical
  accuracy, Cohen's h = 0, McNemar's degenerate) despite real discriminative
  power (AUC-ROC 0.77/0.80).
- **Fixed a real gap in an externally-proposed implementation plan before
  building on it**: the plan assumed `02`'s in-memory `oof_probs` variable
  would be accessible from `04` ("assuming they're saved as an artifact...
  let's assume it's saved") -- verified this was false. Each notebook runs
  as a fully separate kernel process (even within a single `00_run_all`
  run via `nbconvert --execute`), so nothing in `02`'s memory is visible to
  `04` regardless of execution mode. Fixed by having `02` explicitly persist
  the best tree model's out-of-fold probabilities as a new artifact
  (`rq1_oof_probs.json`), which `04` now loads.
- **Added `find_f1_optimal_threshold()`** to `src/metrics.py`, sweeping
  `precision_recall_curve`'s output for the F1-maximizing threshold. Guards
  a real (confirmed, not hypothetical) sklearn edge case: the last
  precision/recall pair has no corresponding threshold, and naively
  indexing `thresholds[argmax(f1_scores)]` over the full-length arrays can
  raise `IndexError` if that threshold-less point happens to be the F1
  maximum -- verified this is a real risk, not just theoretical, before
  adding the guard (slicing to the threshold-having range before argmax).
- The SAME tuned threshold is applied to both the full and reduced models
  in `04`, deliberately, so the full-vs-reduced comparison isn't confounded
  with separately tuning a threshold per model.
- Verified with: a targeted test on SECOM-like synthetic imbalance data
  (recall jumping from 17% to 48% at the tuned vs. default threshold, AUC-
  ROC unchanged), and a full `02`-save -> `04`-load integration test that
  reproduces the exact real-world pattern (precision/recall/F1=0, BER=0.5
  at 0.5 threshold; sensible numbers at the tuned threshold).
- Documented the cost-sensitive threshold (`cost_FP / (cost_FP + cost_FN)`)
  as the more defensible alternative if the fab can state a real cost
  ratio, in the new step's markdown explanation -- not computed here since
  no cost ratio has been specified, but flagged as the better answer if one
  becomes available.

## v15 -- 2026-07-29

- **Real bug found from a fully clean v14 run (all 6 notebooks completed):
  `04`'s full-vs-reduced accuracy comparison came back exactly 0.0000 for
  both models**, with McNemar's test showing "degenerate, no discordant
  pairs." The impossible-looking exact zero (not ~93%, the majority-class
  baseline one would expect) was the tell.
- **Root cause: `load_raw()` returned SECOM's raw `{-1, 1}` label encoding.**
  Only `02_modeling_rq1.ipynb` remapped this locally to `{0, 1}` before
  training; `03` and `04` called `load_raw()` directly and never remapped,
  so their `y` stayed in `{-1, 1}` while the saved models (trained in `02`
  on 0/1 labels) predict in `{0, 1}`. Any direct equality comparison
  between the two (accuracy, McNemar's) silently breaks -- confirmed by
  reproducing the exact 0.0000 result with synthetic data using the same
  mismatched encoding.
- **AUC-ROC-based results are confirmed NOT affected** -- verified directly
  that `sklearn.metrics.roc_auc_score` treats the larger label value as
  positive regardless of whether it's `{-1,1}` or `{0,1}` encoded, so RQ1's
  model comparison, `03`'s permutation importance, and `04`'s AUC-ROC
  comparison (0.7670 full / 0.8003 reduced) all remain valid as reported.
- **Fixed centrally in `load_raw()`** (`src/preprocessing.py`) rather than
  patching each notebook separately -- that per-notebook inconsistency
  (`02` remembered to remap, `03`/`04` didn't) is what caused the bug, so
  centralizing it prevents the same class of bug from recurring in `05`/
  `06` once those are built out. `02`'s existing local remap line is now
  redundant but confirmed harmless (idempotent on already-0/1 labels).
- Verified end-to-end with a realistic reproduction: fit/predict/compare
  using the corrected encoding produces a sensible ~95% accuracy on
  synthetic data, versus the broken 0.0000 with mismatched encoding.

## v14 -- 2026-07-29

- **v13's fix resolved the NaN crash but exposed a second, distinct
  numerical bug: `ValueError: Linkage 'Z' contains negative distances`**,
  raised by `fcluster()` on a real v13 run. Investigated rather than
  assumed: stress-tested whether `pandas.corr()` itself can exceed 1.0
  (2,000 trials with perfectly-correlated synthetic columns -- never
  observed), so the likely source is average-linkage's own recursive
  merge-height computation producing a tiny negative value from floating-
  point rounding, not necessarily the input distances. Confirmed this by
  deliberately constructing a `Z` matrix with a `-1e-15` merge height and
  reproducing scipy's exact error message, then confirming a fix resolves it.
- **Fixed with two additional defense layers** (five total now):
  `distance = np.clip(distance, 0, None)` before `linkage()` (guards the
  input side, addresses the originally-proposed fix), and
  `Z[:, 2] = np.clip(Z[:, 2], 0, None)` after `linkage()`, before
  `fcluster()` (guards linkage's own output -- this is the layer that
  actually matters for this specific failure, confirmed by the
  deliberately-constructed reproduction above).
- Re-ran the full test suite from v12/v13 (constant-column dropping,
  normal clustering correctness, multiple constant columns, no column/
  cluster-id misalignment) plus a new large-scale test (200 features, 25
  underlying correlated factor groups, near-duplicate columns) confirming
  25 clusters form correctly -- and re-ran the full
  `nested_cv_shap_selection()` reproduction. All pass.

## v13 -- 2026-07-29

- **v12's fix for the `cluster_correlated_features()` NaN crash was
  insufficient -- confirmed on a genuine v12 run, not a stale-session
  issue.** Replaced with a three-layer fix: (1) drop fold-locally-constant
  columns entirely before computing correlations (more principled than
  patching their correlation value, since such a column has no correlated
  signal to cluster in that fold), (2) keep `fillna(0)` + forced diagonal
  as a safety net for near-constant edge cases, (3) a defensive
  `np.isfinite` check immediately before `linkage()` as a last-resort net.
- **Found and fixed a second, silent bug while combining these fixes**:
  the return statement zipped cluster IDs with the original `X.columns`
  instead of `X_varying.columns` (the list actually clustered, after
  dropping constants) -- with a constant column anywhere but the end of
  the list, this would have silently misaligned column names to the wrong
  cluster IDs with no crash and no warning. Caught by a dedicated test
  (dropping a column mid-list, then asserting two genuinely-correlated
  features still land in the same cluster) rather than by inspection.
- Verified with 4 tests before shipping: original crash reproduction (now
  correctly drops rather than patches), normal clustering unaffected,
  multiple simultaneous constant columns, and the column-misalignment
  check specifically. Also re-ran the full `nested_cv_shap_selection()`
  reproduction from v12 to confirm no regression.

## v12 -- 2026-07-29

- **Real bug from a clean v11 run: `04` crashed with `ValueError: The
  condensed distance matrix must contain only finite values`** inside
  `cluster_correlated_features()`. Root cause: a column can pass the
  GLOBAL variance screening (over all 1,567 rows) but still be exactly
  constant within one CV fold's ~80% training subset -- correlation is
  undefined (0/0) for a constant column, producing `NaN` that crashed
  `scipy.cluster.hierarchy.linkage()`. Reproduced directly with a synthetic
  fold-locally-constant column before fixing, and re-tested after.
  Fixed by treating undefined (NaN) correlations as 0 (maximally distant --
  a locally-constant column carries no correlated signal to anything in
  that fold, which is the correct interpretation, not a workaround), with
  the diagonal explicitly forced back to 1.0 so the resulting distance
  matrix still has proper zero self-distance.
- **A second bug surfaced while testing the first fix**: naively doing
  `np.fill_diagonal(corr.values, 1.0)` failed with "underlying array is
  read-only" -- pandas can return a read-only view from `.values` in some
  cases. Fixed by explicitly copying before in-place mutation.
- Verified with: (1) a direct reproduction of the original crash on an
  isolated synthetic constant column, confirmed fixed; (2) confirmation
  that normal, non-degenerate correlation clustering still correctly groups
  genuinely correlated features and separates unrelated ones; (3) a full
  `nested_cv_shap_selection()` run with a near-constant column; (4) the
  complete RQ1 -> RQ2 -> RQ3 synthetic integration test re-run end-to-end
  with this fix in place.

## v11 -- 2026-07-29

- **Real bug found from a clean v9 run: permutation importance in
  `03_shap_analysis_rq2.ipynb` came back ~1e-17 (floating-point noise) for
  every single feature.** Root cause: it evaluated `rq1_best_tree` -- which
  RQ1 refits on 100% of the data -- on that SAME data. An unregularized
  RandomForest (no `max_depth` cap) can essentially memorize 1,567 training
  rows, so permuting any one feature barely hurts performance on data the
  model has already seen. This wasn't a code bug, but genuinely wrong
  methodology.
- **Fix: `02_modeling_rq1.ipynb` now saves a second, genuinely-held-out
  model** (`rq1_best_tree_holdout`, trained only on a fixed stratified 80/20
  split saved as `rq1_holdout_split.json`) alongside the existing full-data
  `rq1_best_tree` (kept as-is for SHAP's global-explanation use, which is
  standard practice and doesn't need held-out data the way permutation
  importance does). `03` now evaluates permutation importance with the
  holdout model on the held-out 20% only.
- **Implemented `04`'s previously-stubbed full-vs-reduced comparison**,
  reusing the same shared holdout split: trains full-feature and reduced-
  feature models on the train portion, evaluates both on the held-out test
  portion, and reports three real statistical comparisons via new
  functions in `src/metrics.py`:
  - `bootstrap_auc_comparison()` -- a paired bootstrap substitute for
    DeLong's test (documented reasoning for the substitution in the
    docstring: simpler to verify correctly than re-deriving DeLong's
    structural-component formula from scratch).
  - `mcnemar_test()` -- continuity-corrected chi-square, with an exact
    binomial fallback for small discordant-pair counts (<25).
  - `cohens_h_two_proportion_test()` -- Cohen's h effect size + a
    two-proportion z-test for accuracy comparison.
  All three verified against hand-constructed known-answer cases (clearly-
  different models detected as significant, identical/near-identical
  models detected as not significant) before shipping, not just read
  through.
- Verified the full RQ1 -> RQ2 -> RQ3 artifact chain end-to-end with a
  synthetic-data integration test (save holdout model/split in a RQ1-like
  step, load and use them in RQ2-like and RQ3-like steps) -- confirmed
  permutation importance is non-degenerate and matches SHAP's top features
  on synthetic data where the true signal is known.

## v10 -- 2026-07-29

- **Real failure found: a notebook ran with pre-v9 code even after v9 was
  correctly uploaded** (confirmed by cloning the live GitHub repo directly --
  it was genuinely on v9). Root cause: Colab's **Runtime -> Restart session**
  resets the Python kernel but does not wipe `/content` on disk -- unlike
  **Disconnect and delete runtime**, which does. If `/content/secom-
  explainable-vm` and the `.secom_setup_done` marker from an earlier,
  pre-v9 session both survived a "Restart session," the v9 session-marker
  logic would (correctly, by its prior design) skip `git pull` entirely,
  leaving stale code in place.
- **Fixed by un-bundling the two steps the marker was skipping together:**
  `git pull` now always runs unconditionally (cheap, a few seconds) --
  code can no longer go stale regardless of what state `/content` was left
  in. Only `pip install` (the actually slow step) is still skipped when the
  session marker is present. Verified with a simulation of the exact
  failure scenario (marker present from a prior session) confirming
  `git pull` now runs anyway.
- Practical note for using Colab going forward: prefer **Runtime ->
  Disconnect and delete runtime** over "Restart session" when starting a
  genuinely new attempt, since it guarantees a fully clean VM -- this fix
  makes stale code impossible either way, but a full disconnect avoids
  relying on that fix at all.

## v9 -- 2026-07-29

Three real bugs found from the first full end-to-end run (RQ1 result:
RandomForest won overall, TabNet ranked last of 4 by AUC-ROC -- but `03` and
`04` crashed partway through, so this needs a clean re-run to confirm).

- **SHAP shape bug (real `ValueError`, not cosmetic):** `shap.TreeExplainer`
  returns a single 3D array `(samples, features, classes)` for
  `RandomForestClassifier` in the shap version actually installed in Colab --
  the old code only checked for the older list-of-per-class-arrays
  convention, so `np.abs(sv).mean(axis=0)` produced a 2D result and crashed
  building a pandas Series from it. Fixed with a new shared
  `extract_positive_class_shap()` helper in `src/feature_selection.py`
  handling all 3 known shap output conventions (list, 3D array, already-2D),
  used by both `nested_cv_shap_selection()` and `03_shap_analysis_rq2.ipynb`'s
  inline SHAP cell. Verified against a real `RandomForestClassifier` +
  `TreeExplainer` reproduction of the exact bug, not just unit-style mocks.
- **LASSO selecting zero features:** `LogisticRegressionCV`'s default
  internal scoring is plain accuracy; on this ~93.4%-majority-class dataset,
  an all-zero-coefficient (majority-class-only) model already scores ~93%
  trivially, so C-selection kept picking maximum regularization every time.
  Fixed by setting `scoring="roc_auc"`. Verified on synthetic imbalanced
  data (6.67% positive, matching SECOM's real rate) -- selected the actually-
  predictive features instead of zero.
- **`00_run_all.ipynb`'s `--allow-errors` flag was masking real failures.**
  It was only meant to tolerate each notebook's own trailing live-export
  cell failing in batch mode, but it suppressed *any* cell error, which is
  why `03` and `04` were reported `[OK]` despite the SHAP crash above
  happening inside them. Fixed properly: each notebook's export cell now
  checks whether `_message.blocking_request` returned `None` (the actual
  batch-mode signal) and skips gracefully instead of raising, so
  `--allow-errors` is no longer needed at all and has been removed --
  `00_run_all` now fails loudly and correctly on genuine errors.

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
