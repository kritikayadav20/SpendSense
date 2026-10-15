# About SpendSense Campus

A simple guide to what this project is, how it works, and what the four ML models do.

---

## What is this project?

**SpendSense Campus** is a college mini-project that helps students:

1. **Categorize expenses** — type a short purchase description (like “UPI payment at cafe for lunch”), and the app suggests a category (Food and Dining, Travel, Education, etc.).
2. **Track spending** — save expenses in a local ledger, view charts, and search past transactions.
3. **Budget coaching** — set a monthly budget and see remaining money, overspend, and a simple “safe to spend per day” number.

Only the **category suggestion** uses machine learning. Budget numbers are plain arithmetic (not ML forecasts). Everything runs **locally** on your computer — no bank login, no cloud API, no paid services.

---

## The big idea in one sentence

> We turn expense text into numbers (TF-IDF), train four classifiers on the same data, pick the best one, then use it in a Streamlit app to suggest categories while students manage a simple budget.

---

## Dataset (what we train on)

We use a **synthetic (fake but realistic) campus expense dataset** created for this project — not real bank data.

| Item | Simple meaning |
|------|----------------|
| File | `spendsense_synthetic_v1_seed42.csv` |
| Size | About **2,320** short expense descriptions |
| Input | `description` (the text of the purchase) |
| Output / label | `category` (one of 10 spending types) |
| Examples of categories | Food and Dining, Groceries, Transportation, Shopping, Entertainment, Bills and Utilities, Health, Travel, Subscriptions, Education |

**Important:** Scores measure how well models fit this generated demo data. They do **not** prove accuracy on real student spending.

---

## What is TF-IDF? (shared by all 4 models)

Before any model can learn, text must become numbers.

**TF-IDF** (Term Frequency–Inverse Document Frequency) is a simple text feature method:

- It looks at **words and word pairs** (unigrams and bigrams) in each description.
- Words that appear often in *one* expense but are rarer overall get higher weight.
- Example idea: in “cafe lunch sandwich”, words like *cafe* and *lunch* help point toward **Food and Dining**.

All four models use the **same TF-IDF style features**. Each model then learns a different way to map those features to a category.

---

## Four models — what they are and where they are used

All four are trained and compared in the **ML Lab**. Only **one** becomes the **active model** used when you click **Suggest category** in the app. Selection uses training cross-validation **macro-F1** (fair average across all categories). On the current run, **Linear SVM** is the active model.

### 1. Logistic Regression — TF-IDF features

| | |
|--|--|
| **What it is** | A classic linear classifier. It learns a weight for each word feature and combines them to score each category. |
| **Simple analogy** | Like giving each word a “vote strength” toward Food, Travel, Education, etc., then picking the category with the strongest total vote. |
| **Where it is used** | Trained in `src/train.py`, scored in `src/evaluate.py`, shown in **ML Lab**. Not the active suggester unless it wins CV selection. |
| **What it outputs** | A category plus **probabilities** (useful for “how sure?” — but still uncalibrated). |
| **Why we include it** | Strong baseline for text; easy to understand and compare. |

### 2. Multinomial Naive Bayes — TF-IDF features

| | |
|--|--|
| **What it is** | A fast probability model that assumes word features are independent given the category (the “naive” part). |
| **Simple analogy** | It asks: “Given these words, which category usually produces text like this?” using smoothed word counts. |
| **Where it is used** | Same training/evaluation pipeline; compared in **ML Lab**. |
| **What it outputs** | Category + probabilities (can be overconfident). |
| **Why we include it** | Very common for text classification; quick to train; good teaching contrast with SVM and trees. |

### 3. Linear SVM — TF-IDF features

| | |
|--|--|
| **What it is** | A **Support Vector Machine** with a linear kernel (`LinearSVC`). It finds a hyperplane that best separates categories in the TF-IDF space. |
| **Simple analogy** | Draw the widest possible “gap” between groups of expenses; new text falls on one side of the gap. |
| **Where it is used** | Trained like the others; currently the **active model** for **Suggest category** in Add & Import (selected by best training CV macro-F1). |
| **What it outputs** | Category + a **decision margin/score** (not a true probability). |
| **Why we include it** | Often strong on high-dimensional text; good state-of-the-art classic baseline for this kind of task. |

### 4. Decision Tree — simple, interpretable classification

| | |
|--|--|
| **What it is** | A tree of yes/no rules on TF-IDF features (e.g. “does this text contain word X?”). |
| **Simple analogy** | A flowchart: follow branches until you reach a leaf that names a category. |
| **Where it is used** | Trained and evaluated with the others; **ML Lab** shows a tree visualization and readable rules. |
| **What it outputs** | Category + leaf-based probabilities (can look very sure even when wrong). |
| **Why we include it** | Easy to explain in a viva; shows that “interpretable” does not always mean “best on sparse text.” |

### Quick comparison

| Model | Easy idea | Confidence style | Role in project |
|-------|-----------|------------------|-----------------|
| Logistic Regression | Weighted word votes | Probabilities | Compared in ML Lab |
| Multinomial Naive Bayes | Word-likelihood by class | Probabilities | Compared in ML Lab |
| Linear SVM | Separating hyperplane | Margin / score | **Active category suggester** (current run) |
| Decision Tree | If–then flowchart | Leaf probabilities | Compared + visualized in ML Lab |

---

## Project flow (how everything works)

Think of the project in **two tracks**:

### Track A — Machine learning pipeline (offline / once)

```text
1. Generate synthetic CSV          →  src/generate_synthetic.py
2. Verify & load with manifest     →  src/data_loader.py  (+ data/dataset.json)
3. Clean text & remove bad rows    →  src/preprocessing.py
4. Split train / test (~80/20)     →  same cleaned unique descriptions, seed 42
5. Train 4 TF-IDF + classifier pipelines  →  src/train.py
6. Pick best by training CV macro-F1      →  usually Linear SVM
7. Evaluate all 4 on the same holdout     →  src/evaluate.py
8. Build academic report + figures        →  src/report.py, src/visualization.py
9. Save models (.joblib) for the app      →  models/<run>/
```

**Key rule:** Test data is never used to tune hyperparameters. Tuning happens only on the training side (cross-validation).

### Track B — Student app (what you use day to day)

```text
Streamlit UI (app.py)
    │
    ├─ Add & Import
    │     type description → predict.py (active model) → suggest category
    │     human checks / corrects → save to SQLite
    │
    ├─ Overview / Transactions & Analytics
    │     charts, filters, export CSV
    │
    ├─ Budget Coach
    │     budget.py math: remaining, overspend, safe-per-day
    │     (no ML here)
    │
    ├─ ML Lab
    │     show four-model comparison, confusion matrices, tree plot
    │
    └─ Methodology
          read the docs inside the app
```

Personal expenses live in local SQLite (`data/spendsense.sqlite3`). They are **not** automatically used to retrain the models.

---

## How things are implemented (simple map)

| Piece | File(s) | Job in plain words |
|-------|---------|--------------------|
| Web UI | `app.py` | Six pages for ledger, budget, and ML Lab |
| Fake training data | `src/generate_synthetic.py` | Builds the demo CSV + source info |
| Data checks | `src/data_loader.py` | Reads CSV, checks schema/hash/manifest |
| Cleaning | `src/preprocessing.py` | Normalize text, drop bad/duplicate labels |
| Training | `src/train.py` | Fit four TF-IDF pipelines, pick winner |
| Evaluation | `src/evaluate.py` | Same holdout scores for all four models |
| Predictions | `src/predict.py` | Suggest category for new text in the app |
| Database | `src/database.py` | Save expenses and budgets locally |
| CSV import | `src/importer.py` | Validate and import user expense files |
| Budget math | `src/budget.py` | Remaining / overspend / daily room |
| Plots | `src/visualization.py` | Confusion matrices and tree figures |
| Report | `src/report.py` | Assembles the written submission draft |

---

## App pages (what each screen is for)

1. **Overview** — Month totals, readiness status, quick picture of spending.
2. **Add & Import** — Enter one expense or import a CSV; use **Suggest category** from the active model.
3. **Transactions & Analytics** — Search, filter, edit categories, charts, export.
4. **Budget Coach** — Set monthly (and optional category) budgets; see remaining and pace.
5. **ML Lab** — Four-model comparison and performance analytics (this is where you study all models).
6. **Methodology** — Project docs: how training, data, and limits were designed.

---

## Budget logic (no ML)

If `B` = monthly budget, `S` = spending so far, `D` = days left including today:

- **Remaining** = max(0, B − S)
- **Overspend** = max(0, S − B)
- **Safe per day** = remaining ÷ D

This is transparent calendar math for students — not a prediction model.

---

## What success looks like for this mini-project

- Four reproducible TF-IDF classifiers trained and compared fairly.
- One selected model suggests categories in the live app.
- Students can correct the suggestion before saving.
- Budget coach explains remaining money with clear formulas.
- ML Lab shows honest metrics on the **synthetic** demo — with the limitation that real-world accuracy is still unproven.

---

## One-line takeaway

**SpendSense Campus** = text classifiers that guess expense categories + a local student ledger + simple budget arithmetic, with all four classic models compared so you can see *which* approach works best on this demo data and *why* Linear SVM was chosen as the active suggester.
