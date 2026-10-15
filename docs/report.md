# SpendSense Campus
## ML-Powered Expense Categorization and Student Budget Coach

**Academic report draft · 15-mark college mini-project**  
**Experimental status: pending verified dataset.** No performance values below are fabricated. The application and automated tests are separate from a real-data experiment. Intended layout: approximately eight to ten A4 pages at 11-point text, including figures and references; verify final page count after typesetting.

### 1. Abstract

SpendSense Campus combines multiclass expense-description classification with a local student budgeting dashboard. Four scikit-learn pipelines—Logistic Regression, Multinomial Naive Bayes, Linear SVM and Decision Tree—use TF-IDF representations and a common holdout. Hyperparameters and the deployed model are selected by cross-validated macro-F1 on training data. SQLite stores personal expenses and corrections; Streamlit presents imports, history, budgets and benchmark evidence. Budget guidance uses disclosed calendar arithmetic. A source-verification manifest prevents silently treating undocumented or synthetic records as real transactions. At this stage a verified labelled dataset has not been acquired, so model results and real-world usefulness remain unmeasured. The source recipe, split membership, saved pipelines and evaluation artifacts support reproducibility.

### 2. Introduction and problem statement

Students make many small purchases, yet recording categories consistently requires effort. A missed purchase or incorrect category can distort a budget summary. The project asks whether short descriptions can provide useful category suggestions while keeping the user's correction and review workflow clear. It also asks how recorded spending can be translated into understandable daily spending room.

The scope is a local, single-user application with manual entry, CSV import/export, category correction and monthly/category budgets. The classification task is supervised, multiclass and single-label. The budget component is deterministic. There are no bank integrations, payment flows, cloud databases or generated financial advice. Originality lies in the integration of fair benchmarking, transparent uncertainty and a campus-oriented workflow, rather than a new classifier.

### 3. Literature review

Recent literature explores several routes beyond manual categorization. Toran et al. [1] study weak supervision for transaction classification, emphasizing the cost of reliable labels. Busson et al. [2] address hierarchical transaction taxonomies with transformer representations; preserving label meaning is a lesson applicable even to simple models. Kotios et al. [3] connect transaction classification with cash-flow modelling, while González-González et al. [4] examine explanations of classifications used in downstream estimates. Tang et al. [5] discuss imbalance in the adjacent fraud-detection task; its target and architecture are not adopted here. Earlier directly relevant work by García-Méndez et al. [6] investigates SVM classification of short transaction descriptions and similarity-based data reduction.

This project favors interpretable, accessible baselines over those more complex architectures. It separates the quality of the learned classifier from the correctness of budget calculations and the usability of the ledger. Full verified titles, author lists, years, venue records and stable links are in Section 12 of the generated submission and `docs/literature_review.md`.

### 4. Dataset and provenance

The primary candidate is the AuroraGate Expense Categorization Challenge on Kaggle. Initial repository inspection found no files. Browsing the candidate page did not reveal a readable file list or schema. Consequently, actual filenames, record counts, headers, target categories, license and real/synthetic origin are pending verification. There is no claim that this source contains actual bank transactions. No fallback dataset was silently introduced.

The implemented acquisition workflow preserves the original CSV in `data/raw/`, profiles its structure and class counts, and requires a manifest containing field mappings, label taxonomy, license evidence, provenance notes and SHA-256. The manifest supports another legitimate documented transaction dataset. Source assertions still require human review: software cannot prove a license or the truth of a collection claim. Amount and date are accounting fields; only description text enters the implemented model.

### 5. Cleaning and TF-IDF representation

Unicode and whitespace normalization are conservative: merchant punctuation, numbers and accents remain. Blank or invalid descriptions/targets are excluded. All exact normalized descriptions with conflicting labels are removed. Same-label repetitions collapse to the first record. Each exclusion records a source row and reason. Class counts before and after cleaning expose imbalance and unsupported classes. This policy makes evaluation about unique descriptions, rather than transaction-frequency-weighted use.

TF-IDF represents text using word unigrams/bigrams, sublinear term frequency and normalized weights, with a maximum of 20,000 features. Common corpus terms receive less distinctive weighting than rare terms. A Pipeline fits vocabulary and inverse document frequencies inside training folds only. A word present only in test data cannot influence training. Saved pipelines perform the identical transformation at inference.

### 6. Classifiers, splitting and model selection

Logistic Regression learns regularized linear class boundaries and supports multiclass probabilities. Multinomial Naive Bayes combines class priors with smoothed nonnegative feature evidence. Linear SVM learns maximum-margin boundaries and provides decision scores, not probabilities. Decision Tree recursively tests feature thresholds; restricted depth, minimum leaf size and pruning limit fragmentation. Sparse text provides many possible local splits, so a tree can generalize differently from linear models that combine distributed evidence.

A seeded, stratified holdout contains approximately 20% of the cleaned unique descriptions. Every model uses its exact records. At least five descriptions per class and two classes are required. Within the remaining training partition, two to five stratified folds tune fixed search grids. The model with highest mean CV macro-F1 is selected before primary test evaluation. No feature selection, calibration or hyperparameter choice uses the holdout.

Exact descriptions are disjoint between train and test. A second evaluation is attempted using groups formed by removing numeric and punctuation variation. It independently tunes within its own training groups and never changes the primary selected model. If class support is inadequate, the reason is recorded. This is a template sensitivity analysis, not proof of generalization to new merchants or another college.

### 7. Evaluation and results

Accuracy is the fraction of correct labels. Precision measures predicted-class correctness; recall measures actual-class coverage; F1 balances both. Macro-F1 gives every class equal weight and therefore exposes weak minority-category performance. Support-weighted averages and per-class tables supply complementary context. Confusion matrices reveal frequently confused pairs. Training time includes CV and refit; latency is batch prediction time per record, not an online service guarantee. ROC-AUC is omitted because class-prediction metrics meet the project's comparison needs without additional probability assumptions.

<!-- RESULTS -->

**Results pending:** no verified dataset has been trained. All four benchmark rows—Logistic Regression, Multinomial Naive Bayes, Linear SVM and Decision Tree—have pending accuracy, macro-F1, weighted-F1 and timing values. There is no measured winner. Confusion matrices, per-class charts, a comparison plot and tree visualization will be generated from actual predictions by the evaluation command. Do not use scores from synthetic unit tests in this report.

| Model | CV macro-F1 | Test macro-F1 | Accuracy | Weighted-F1 | Training time |
|---|---|---|---|---|---|
| Logistic Regression | Pending | Pending | Pending | Pending | Pending |
| Multinomial Naive Bayes | Pending | Pending | Pending | Pending | Pending |
| Linear SVM | Pending | Pending | Pending | Pending | Pending |
| Decision Tree | Pending | Pending | Pending | Pending | Pending |

<!-- END RESULTS -->

Discussion must be completed after execution: identify minority failures, inspect privacy-redacted error rows locally, compare primary and grouped results, and distinguish high within-source scores from external validity. A selected model's current strength is its training-CV ranking, not proven usefulness for college expenses. Compare tree and linear-model scores using the executed table; any difference must be interpreted within this evaluation's scope.

### 8. Inference, interface and persistence

The feature-mapping module validates a description and supplies only that string to the selected pipeline. The app displays its category and allows correction. LR/NB estimates and Tree leaf proportions are labelled uncalibrated; Tree estimates may be particularly extreme. SVM output is a margin. Fixed review heuristics use maximum probability below 0.60 or margin below 0.50; no known vocabulary always triggers review. These rules are not calibrated error guarantees. Recognized terms show vocabulary coverage rather than causal evidence.

The six Streamlit pages are Overview, Add & Import, Transactions & Analytics, Budget Coach, ML Lab and Methodology. A restrained teal/navy palette, readable tables, responsive columns, explicit empty states and Plotly charts support a live demonstration. Cached model resources avoid retraining on refresh. Imports validate all rows before committing and allow category edits. SQLite uses parameterized queries and integer minor units for money. Corrections preserve the original suggested label for inspection and never trigger retraining.

### 9. Budget arithmetic

With recorded spending S, monthly budget B and D remaining days including today, remaining balance is max(0, B−S), overspend is max(0, S−B), and safe daily spending is remaining/D when D is positive. The pace estimate is S/elapsed days multiplied by month length. Spending drift compares S with an evenly allocated budget to date. The current day is counted in both elapsed and remaining periods. Past months have no remaining daily allowance; future months have no observed pace estimate. Leap years are handled by the standard calendar library.

These calculations depend on complete recorded expenses. Irregular fees, rent or purchases can make extrapolation inaccurate. Category budgets are independent of the overall limit; exceeding either creates a visible alert. No part of this arithmetic is described as a trained financial forecast.

### 10. Reproduction, testing and deployment

Create an isolated environment, install pinned dependencies, follow `data/README.md`, run `python -m src.train`, `python -m src.evaluate`, then `streamlit run app.py`. Run `pytest -q` for schema, cleaning, leakage, all four classifiers, persistence, uncertainty, database, import/export, budgets and Streamlit smoke tests. Unit fixtures are synthetic and confined to temporary directories. Re-running evaluation uses saved models and verifies the saved test checksum; it never retrains.

Completed runs store all four pipelines, CV candidates, environment versions, the selected-model criterion, split row IDs and generated plots. `python -m src.report` assembles this draft with actual result artifacts when present. Local deployment binds to localhost. There is no claimed public deployment or hosted URL. See `docs/verification.md` for commands actually executed and their outcomes.

### 11. Limitations, privacy and ethics

Data access is the present experimental blocker. Source-domain vocabulary, label noise, class imbalance and ambiguous mixed purchases can limit performance. Exact de-duplication does not remove every semantic near-duplicate. A consented campus validation set would be valuable but has not been collected. It must remain separate from training and model selection.

Users should omit identifiers and credentials. Personal records, processed descriptions, fitted models and generated vocabulary figures are ignored by Git. CSV exports neutralize formula-like text; descriptions are withheld from shared error reports. Local storage is not encrypted and this demo has no authentication, so it should not be exposed publicly with personal data. A model is a suggestion tool, and corrections remain under user control.

### 12. References

The report builder appends the verified bibliography from `docs/literature_review.md`, preserving stable source links and distinguishing publication years from later arXiv deposits.
