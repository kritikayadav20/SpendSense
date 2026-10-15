# ◈ SpendSense Campus

**ML-Powered Expense Categorization and Student Budget Coach**

A local college mini-project combining four reproducible TF-IDF classifiers with a Streamlit expense ledger and transparent budget arithmetic. Students can record purchases, inspect spending patterns, correct categories and calculate daily spending room.

**Current state:** a reproducible **synthetic demonstration dataset** has been generated at the user's request, and all four classifiers have been trained and evaluated. The active model is **Linear SVM**, selected by training-only CV macro-F1. Categorization and the ML Lab now work. Scores measure the authored generator, not real-world student transactions. See [executed results](reports/results.md), [synthetic dataset documentation](docs/synthetic_dataset.md), and [verification](docs/verification.md).

## Quick start

Python 3.11 is recommended for a new installation; the pinned stack also supports the available Python 3.9 environment used during this implementation. Use Python 3.9–3.12 with these pins; do not assume Python 3.13 wheel support.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`. The application binds to localhost and disables Streamlit usage telemetry. No paid API or account is required. Keep the terminal running. For the already-created environment, `.venv/bin/streamlit run app.py` also works.

`requirements-lock.txt` records the full resolved environment used for verification (Python 3.9, macOS ARM64). The main requirements file pins direct dependencies; the lock file additionally records transitive versions for reproducing this environment.

## run it on your PC

Choose **one** guide below for your operating system. Run each command in order and wait for it to finish before continuing. If a command fails, fix that error before running the next step.

**Before starting:** install **Python 3.11** and **Git**, then open a new terminal. You need internet access to download the project and install its dependencies. The commands use the virtual environment's Python directly, so you do **not** need to activate it or change PowerShell's execution policy.

A fresh GitHub checkout does not contain this computer's trained models, generated reports, virtual environment, or personal database. The steps below recreate the synthetic dataset, train all four models, and launch the complete app locally.

### Windows — PowerShell

**Step 1 — Check Python and Git.**

Open **PowerShell** and run:

```powershell
py -3.11 --version
git --version
```

The first command should print `Python 3.11.x`. If either command is not found, finish installing that program and reopen PowerShell.

**Step 2 — Download the project.**

On this GitHub repository, click **Code → HTTPS** and copy the clone URL. Run the following commands; paste the copied URL when prompted:

```powershell
$repoUrl = Read-Host "Paste this repository's HTTPS clone URL"
git clone "$repoUrl" spendsense-campus
Set-Location spendsense-campus
Get-Item app.py, requirements.txt
```

The final command should list both files. Keep this terminal in the `spendsense-campus` folder for every remaining step. If you already downloaded and extracted the ZIP, open PowerShell inside the folder containing `app.py` and start at step 3.

**Step 3 — Create the virtual environment.**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe --version
```

**Step 4 — Install and check dependencies.**

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

Wait for installation to finish. The last command should print `No broken requirements found`.

**Step 5 — Generate and inspect the synthetic dataset.**

```powershell
.\.venv\Scripts\python.exe -m src.generate_synthetic --activate
.\.venv\Scripts\python.exe -m src.data_loader data/raw/spendsense_synthetic_v1_seed42.csv --description-column description --target-column category
```

The generator creates the CSV and active source manifest. The inspection command prints the actual shape, missing values, checksum, and category counts.

**Step 6 — Train all four models.**

```powershell
.\.venv\Scripts\python.exe -m src.train
```

Wait until the command finishes and prints `Completed ...; CV-selected model: ...`. This creates the saved models and makes the selected model available to the app.

**Step 7 — Generate evaluation results and the academic report.**

```powershell
.\.venv\Scripts\python.exe -m src.evaluate
.\.venv\Scripts\python.exe -m src.report
```

Evaluation prints the model comparison and saves its tables and figures. The assembled report is saved as `reports/submission.md`.

**Step 8 — Run the automated tests.**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The tests should finish with all tests passing. They use temporary fixtures and databases.

**Step 9 — Start the app.**

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Keep PowerShell open and open **http://127.0.0.1:8501** in your browser. Continue with **Try the running app** below.

### macOS / Linux — Terminal

**Step 1 — Check Python and Git.**

Open **Terminal** and run:

```sh
python3.11 --version
git --version
```

The first command should print `Python 3.11.x`. If either command is not found, finish installing that program and reopen Terminal. On Linux, also install your distribution's virtual-environment support package for Python 3.11 if `venv` is unavailable.

**Step 2 — Download the project.**

On this GitHub repository, click **Code → HTTPS** and copy the clone URL. Run the following commands; paste the copied URL when prompted:

```sh
printf 'Paste this repository’s HTTPS clone URL: '
read -r repo_url
git clone "$repo_url" spendsense-campus
cd spendsense-campus
ls app.py requirements.txt
```

The final command should list both files. Keep this terminal in the `spendsense-campus` folder for every remaining step. If you already downloaded and extracted the ZIP, open Terminal inside the folder containing `app.py` and start at step 3.

**Step 3 — Create the virtual environment.**

```sh
python3.11 -m venv .venv
./.venv/bin/python --version
```

**Step 4 — Install and check dependencies.**

```sh
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m pip check
```

Wait for installation to finish. The last command should print `No broken requirements found`.

**Step 5 — Generate and inspect the synthetic dataset.**

```sh
./.venv/bin/python -m src.generate_synthetic --activate
./.venv/bin/python -m src.data_loader data/raw/spendsense_synthetic_v1_seed42.csv --description-column description --target-column category
```

The generator creates the CSV and active source manifest. The inspection command prints the actual shape, missing values, checksum, and category counts.

**Step 6 — Train all four models.**

```sh
./.venv/bin/python -m src.train
```

Wait until the command finishes and prints `Completed ...; CV-selected model: ...`. This creates the saved models and makes the selected model available to the app.

**Step 7 — Generate evaluation results and the academic report.**

```sh
./.venv/bin/python -m src.evaluate
./.venv/bin/python -m src.report
```

Evaluation prints the model comparison and saves its tables and figures. The assembled report is saved as `reports/submission.md`.

**Step 8 — Run the automated tests.**

```sh
./.venv/bin/python -m pytest -q
```

The tests should finish with all tests passing. They use temporary fixtures and databases.

**Step 9 — Start the app.**

```sh
./.venv/bin/python -m streamlit run app.py
```

Keep Terminal open and open **http://127.0.0.1:8501** in your browser.

### Try the running app

1. Confirm the sidebar says **Synthetic demo model ready**.
2. Open **Add & Import**. Enter `UPI payment at cafe for lunch`, an amount such as `120`, and today's date.
3. Click **Suggest category**, review the suggestion, check **I have checked the category**, and click **Save expense**.
4. Open **Overview** or **Transactions & Analytics** to see the saved expense. A fresh installation has an empty personal ledger; charts appear after adding or importing expenses.
5. Open **Budget Coach** and save a monthly budget such as `3000`. Select the same reporting month as your expense to see the remaining budget and daily spending room.
6. Open **ML Lab** to inspect the four-model comparison, confusion matrices, grouped-template evaluation, and Decision Tree visualization.

The training dataset is synthetic and separate from your ledger. Benchmark scores measure generated examples, not real-world student-expense accuracy. Expenses and budgets you enter persist locally in `data/spendsense.sqlite3`.

### Stop and restart the app

**Stop:** press **Ctrl+C** in the terminal running Streamlit. Closing the browser alone does not stop the server or delete your records.

**Restart:** open a terminal inside the same project folder containing `app.py`, then run the command for your operating system:

**Windows — PowerShell:**

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

**macOS / Linux:**

```sh
./.venv/bin/python -m streamlit run app.py
```

Open **http://127.0.0.1:8501** again. You do not need to reinstall dependencies, regenerate data, or retrain for an ordinary restart. The app loads the saved model. GitHub stores the project code; these commands run the live application on your computer.

### Troubleshooting commands

**Dependency download timed out:** check your internet connection and retry from the project folder.

Windows — PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --timeout 120
```

macOS / Linux:

```sh
./.venv/bin/python -m pip install -r requirements.txt --timeout 120
```

**Port 8501 is already in use:** start on port 8502 instead, then open **http://127.0.0.1:8502**.

Windows — PowerShell:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

macOS / Linux:

```sh
./.venv/bin/python -m streamlit run app.py --server.port 8502
```

**Model setup is pending:** complete steps 5–7 in your operating system's guide, check for command errors, then refresh the browser.

**Missing module:** repeat step 4 using the explicit `.venv` Python commands shown above.

**Missing `app.py` or `requirements.txt`:** your terminal is in the wrong folder. Open it in the extracted/cloned project folder before running commands.

**No spending charts:** add an expense and select its reporting month. Model training does not insert expenses into your personal ledger.

## Features

- Overview with selected-month totals, category spending, recent activity and readiness status.
- Manual entry, validated CSV previews, automatic categories after training, and correction before saving.
- Search/date/category filters, daily/monthly charts, review flags and CSV export.
- SQLite transactions and monthly/category budget settings, using integer paise.
- Daily safe-to-spend, explicit overspend and pace-based month-end estimates.
- ML Lab with executed metrics only, four confusion matrices, per-class scores, tree visualization and source/split metadata.
- Methodology, dataset/model cards, verified literature review and report draft.

Use one currency (INR). CSV columns are `date,description,amount,category` with optional `needs_review`; category may be omitted only when a trained model is available. Dates use YYYY-MM-DD and cannot be in the future. Amounts must be positive. A blank header template can be downloaded in the app. Imported identical date/normalized-description/amount combinations are skipped, including after correction; genuine identical purchases can be entered manually. Category spelling is case-sensitive. Exports prefix formula-like text with an apostrophe for spreadsheet safety; deliberate unescaping is needed for a lossless re-import of those unusual strings.

## Architecture

```text
Verified raw CSV + source manifest
    → schema audit → conservative cleaning → common stratified holdout
    → training-only CV [TF-IDF → one of four classifiers]
    → CV-selected pipeline + all four artifacts → fixed-holdout reports
                                                  ↓
Streamlit → feature mapping → cached predictor → human review
    ↕
SQLite ledger/budgets → calendar arithmetic → Plotly analytics
```

`src/data_loader.py`, `preprocessing.py`, `train.py`, `evaluate.py` and `predict.py` separate data and model concerns. `database.py`, `importer.py` and `budget.py` are independent of the UI. `visualization.py` produces Matplotlib report figures. `app.py` is the six-page presentation layer. No ordinary page refresh performs training.

## Dataset setup

The active source is the locally authored **SpendSense synthetic campus corpus**, explicitly requested by the user. It contains fictional descriptions and generator-assigned category labels. Its source, taxonomy, generation rules, known ambiguities and permission record are documented in [docs/synthetic_dataset.md](docs/synthetic_dataset.md). The original Kaggle candidate has not been acquired or verified; these examples are not attributed to Kaggle.

Reproduce the current demonstration:

```sh
python -m src.generate_synthetic --activate
python -m src.data_loader data/raw/spendsense_synthetic_v1_seed42.csv --description-column description --target-column category
python -m src.train
python -m src.evaluate
python -m src.report
```

The first command writes the deterministic CSV and source manifest, and explicitly selects it as `data/dataset.json`. It does not add synthetic expenses to your personal ledger. The raw CSV and generator are included for reproducibility; private raw/processed records and fitted artifacts remain ignored by Git. A fresh checkout needs to run these commands to generate local models.

To replace this with a legitimate documented real dataset, follow [data/README.md](data/README.md), preserve its original CSV, inspect actual headers and complete a new source manifest. `python -m src.train --manifest path/to/manifest.json --seed 42` supports another verified source. `python -m src.evaluate --run RUN_ID` re-evaluates fixed saved models without tuning. Do not change the recipe or hyperparameters to chase a previously inspected test score.

## Models and evaluation

| Model | Tuning on training CV | Inference output |
|---|---|---|
| Logistic Regression | C: 0.5, 2 | Uncalibrated class probability |
| Multinomial Naive Bayes | alpha: 0.1, 0.5, 1 | Uncalibrated class probability |
| Linear SVM (LinearSVC) | C: 0.5, 2 | Decision score/margin, never probability |
| Decision Tree | depth: 12, 30; leaf size: 2, 5; pruning alpha: 0, 0.001 | Leaf probability; may be poorly calibrated |

Each pipeline fits its own TF-IDF vocabulary only inside training folds. The common policy uses unigrams/bigrams, sublinear TF and up to 20,000 features. Exact descriptions are de-duplicated before an approximately 80/20 stratified split. A separate grouped-template evaluation is attempted where class support permits.

**Selection:** highest mean training CV **macro-F1**, not training accuracy or test ranking. Macro-F1 weights each category equally. Precision measures correctness of predicted labels; recall measures coverage of true labels; F1 is their harmonic mean. Accuracy measures overall correct predictions; weighted averages use class support. Per-class tables/confusion matrices expose hidden failures. Latency is batch elapsed time divided by records, and training time includes tuning/refit. ROC-AUC is not reported.

Executed synthetic benchmark results are available in `reports/results.md`, `reports/metrics/<run>/comparison.csv` / `.json`, or the ML Lab. Detailed artifacts include per-class scores, all confusion matrices, privacy-redacted error rows, dataset profile, template sensitivity results and readable tree rules. The active pointer is published only after all four primary model reports succeed.

Confidence rules are documented heuristics: probability < 0.60, SVM margin < 0.50, or no known vocabulary requires review. They are not calibrated guarantees and are not comparable across models. The actual tree and linear-model measurements can be compared in the ML Lab; their ranking applies to this synthetic experiment.

## Budget logic

`remaining = max(0, B - S)`; `overspend = max(0, S - B)`; `safe_per_day = remaining / D`, where D includes today. Projection is recorded spending divided by elapsed days times month length. Past months have no remaining allowance; future months have no pace projection. Missing budgets are distinct from zero budgets. Category limits are independent and their sum may exceed the overall limit, which the UI flags. Missing or irregular expenses can distort the estimate. These are calculations, not ML forecasts or financial advice.

## Tests and live demonstration

```sh
pytest -q
python -m pip check
```

Tests cover all four estimators and the complete artifact pipeline using separate in-memory fixtures saved only under temporary test directories. The authorized demonstration corpus is generated independently by `src.generate_synthetic`; test fixture scores never populate the ML Lab. Streamlit AppTest covers all six empty-state pages, manual entry and populated analytics/budget pages.

For a viva, follow [docs/demo.md](docs/demo.md). The strongest demonstration adds your own non-sensitive expense, corrects its category, imports a small user-provided CSV, configures a budget, explains overspend and shows the ML Lab status honestly. The current trained categorization and benchmark use the explicitly labelled synthetic demonstration corpus. Real student-expense accuracy remains unmeasured.

## Privacy and limitations

The app runs locally with no bank credentials, cloud database or automatic training from personal entries. SQLite, raw/processed data, model files, generated vocabulary figures and secrets are excluded from Git. SQLite is not encrypted and there is no multi-user authentication. Do not expose this demo to the internet with personal records. Load only trusted locally generated Joblib artifacts.

Source labels may not cover campus purchases. Vocabulary drift, class imbalance, mixed purchases and near-duplicates limit validity. Description-disjoint scores do not prove unseen-merchant or college generalization. No consented campus validation or public deployment has been performed. Dataset permission and provenance must be documented independently of the code.

## Documentation and references

- [Academic report with executed results](reports/submission.md), [report template](docs/report.md), [literature review](docs/literature_review.md)
- [Methodology](docs/methodology.md), [dataset card](docs/dataset_card.md), [model card](docs/model_card.md)
- [Verified sources and papers](docs/literature_review.md) include transaction categorization, explainability and class-imbalance research from 2022–2024 plus a directly relevant earlier SVM paper.
- [Streamlit](https://docs.streamlit.io/) · [scikit-learn](https://scikit-learn.org/1.5/) · [SQLite](https://www.sqlite.org/docs.html)
