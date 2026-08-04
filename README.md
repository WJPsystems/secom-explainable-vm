# Explainable Virtual Metrology for Semiconductor Yield Prediction Under Severe Class Imbalance

QM640 Data Analytics Capstone — SHAP-enhanced machine learning pipeline for wafer
pass/fail prediction using the SECOM sensor dataset.

## Project summary

Semiconductor fabs generate hundreds of correlated, high-dimensional sensor
readings per wafer. This project builds a virtual metrology pipeline that:

1. Predicts wafer pass/fail from SECOM sensor data under severe class imbalance, comparing classical models (logistic regression, Random Forest, XGBoost) against a TabNet neural model (RQ1)
2. Identifies which sensors drive predictions using SHAP (RQ2)
3. Tests whether a SHAP-guided reduced feature set matches full-feature performance, using nested cross-validation, correlation clustering, and an independent LASSO cross-check to avoid leakage and single-method bias (RQ3)
4. Tests whether TabNet's built-in attention-based interpretability is actually trustworthy, using SHAP as the reference and a plain MLP as a control to isolate the attention mechanism's contribution (RQ4)
5. As a secondary, exploratory robustness check (not a formal RQ): an unsupervised Isolation Forest anomaly detector on the full feature set, to catch novel failure modes that the reduced supervised model can't learn from labels alone

See `docs/synopsis.docx` for the full capstone synopsis, including the research
questions, sample-size justification, and evaluation metrics.

## Data source

SECOM [Dataset]. UCI Machine Learning Repository (McCann & Johnston, 2008).
https://doi.org/10.24432/C54305 — licensed CC BY 4.0, so the dataset is
redistributed in `data/raw/` for reproducibility rather than requiring a
separate download.

- 1,567 wafers, 591 continuous sensor features, binary label (-1 = pass, 1 = fail)
- 104 fail cases (~6.6%) — significant class imbalance


**On HTML run exports and synopsis updates not appearing on GitHub
automatically:** nothing in this pipeline has write access to push to
GitHub — not Colab, not the tools used to edit this repo. Every update
(code fixes, synopsis revisions, HTML exports) has to be manually
downloaded and re-uploaded, the same way the code itself gets updated. For
HTML exports specifically, `00_run_all.ipynb`'s final cells download all
six as one zip — upload that zip's contents to a folder such as
`docs/run_exports/` if you want them version-controlled.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── data/
│   ├── raw/            # original SECOM data + labels, as distributed by UCI
│   ├── model_ready/     # cleaned/screened data used for modeling (generated)
│   └── artifacts/       # saved models + JSON summaries passed between notebooks (generated)
├── notebooks/
│   ├── 00_run_all.ipynb               # runs 01-06 in sequence, one command (see usage below)
│   ├── 01_data_screening.ipynb        # missingness, variance, correlation screening
│   ├── 02_modeling_rq1.ipynb          # RQ1: LogReg/RF/XGBoost + TabNet, imbalance-corrected
│   ├── 03_shap_analysis_rq2.ipynb     # RQ2: SHAP feature importance
│   ├── 04_feature_reduction_rq3.ipynb # RQ3: nested-CV SHAP selection, clustering, LASSO cross-check
│   ├── 05_attention_comparison_rq4.ipynb  # RQ4: SHAP vs. TabNet attention vs. MLP control
│   └── 06_anomaly_safety_net.ipynb    # exploratory: Isolation Forest on full feature set
├── src/
│   ├── fetch_data.py         # pulls SECOM via ucimlrepo
│   ├── preprocessing.py      # missingness/variance screening, train/test split
│   ├── feature_selection.py  # nested-CV SHAP selection, correlation clustering, LASSO cross-check
│   ├── explainability.py     # ALE curves for top SHAP features (complementary to SHAP, not PDP)
│   ├── artifacts.py          # save/load helpers so 02-06 pass real results to each other, not assumptions
│   ├── anomaly_detection.py  # Isolation Forest safety net (full feature set, unsupervised)
│   └── metrics.py            # BER, MCC, and other evaluation metric helpers
└── docs/
    ├── synopsis.docx           # capstone synopsis (mentor-review draft)
    └── data_dictionary_template.csv
```

## Setup

```bash
python -m venv venv
source venv/bin/activate   # venv\Scripts\activate on Windows
pip install -r requirements.txt
python src/fetch_data.py   # downloads SECOM into data/raw/ if not already present
```

## Working in Google Colab (recommended if you're not set up locally)

GitHub renders `.py` files and `.ipynb` notebooks (including saved outputs) for
viewing, but it does not execute code — there's no "Run" button on GitHub itself.
To actually run these notebooks:

1. In Colab: **File → Open notebook → GitHub tab**, paste this repo's URL, pick a notebook.
2. Every notebook's second cell is a **Colab setup cell** that automatically
   clones this repo and installs `requirements.txt` when it detects it's running
   in Colab. Just run it first. It does nothing if you're running locally in
   Jupyter from the `notebooks/` folder.
3. Before first use, edit the `REPO_URL` placeholder in that cell (or in this
   README) to point at your actual GitHub repo once it exists.
4. To save changes back: **File → Save a copy in GitHub** in Colab, which commits
   directly without needing separate `git` commands.

### Running everything at once

Instead of opening and running `01`-`06` individually, open `00_run_all.ipynb`
and run its cells. It executes all six in sequence, prints a pass/fail summary,
exports each successfully-run notebook to HTML, and (in Colab) offers to
download all the HTML files as one zip. Expect a full run to take a while --
RQ1's TabNet cross-validation loop alone is the slowest single step. If you're
only iterating on one notebook's logic, open and run that notebook directly
instead; use `00_run_all.ipynb` when you want a full, clean, start-to-finish
run of everything.

## Status

- [x] Synopsis drafted, mentor-reviewed, Preliminary Results section added with real data
- [x] Data screening complete (confirmed on real SECOM data, multiple runs)
- [x] RQ1 baseline models (LogReg, RF, XGBoost, TabNet) -- confirmed on real data
- [x] RQ2 SHAP analysis + permutation importance -- confirmed on real data
- [x] RQ3 nested-CV feature reduction + LASSO cross-check + threshold calibration -- confirmed on real data
- [ ] RQ4 SHAP vs. TabNet attention vs. MLP control -- code fixed after a real bug
      found on the first real-data run (MLP training divergence from unscaled
      features -- see CHANGELOG v19); **needs to be re-run** with the fix before
      any RQ4 numbers can be trusted
- [x] Exploratory: Isolation Forest anomaly safety net -- confirmed on real data
      (flagged 17/314 held-out wafers as anomalous; caught 0 of 7 fails the
      supervised model missed at its tuned threshold -- an honest null result,
      not a bug)
- [ ] Final report

## Versioning

Delivered zip/docx files are named `<name>_vN_YYYY-MM-DD.<ext>` so two
deliveries are always visibly distinguishable by filename. Files *inside*
this repo are not individually suffixed with version numbers -- that would
break the relative imports between notebooks and `src/`, and Git's own
commit history already tracks file-level changes. Instead, each notebook's
first cell has a **Notebook version** line, and `CHANGELOG.md` at the repo
root summarizes what changed in each delivered version -- check either one
if you're ever unsure whether you're looking at a current or stale copy.

## License

Code in this repository: MIT (adjust as your program requires).
SECOM data: CC BY 4.0, per UCI Machine Learning Repository.
