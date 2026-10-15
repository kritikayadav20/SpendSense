# Current verification — authorized synthetic dataset

The user explicitly requested generation and use of synthetic data after the initial delivery. That demonstration workflow is now executed and active.

- Generated and inspected `spendsense_synthetic_v1_seed42.csv`: 2320 rows, 2 columns, 10 authored categories, zero missing fields; cleaning retained all rows.
- Trained all four required TF-IDF pipelines with the fixed training-only CV search. Shared primary holdout: 464 records; training: 1856.
- Active run: `20261010T121905Z-f76a69e2`. Selected model: **Linear SVM** by CV macro-F1 (0.9574).
- Actual synthetic holdout macro-F1: **0.9453**; accuracy: **0.9440**. Independent grouped-template macro-F1: **0.8878**.
- Current tests: **37 passed in 2.86s**. Tests now isolate empty-state UI coverage from the active project model, verify generator determinism/schema/checksums, and assert synthetic notices.
- All six app pages and the actual active model's category suggestion were smoke-tested using a temporary ledger. Evidence: `reports/verification/active_model_smoke.json`.
- The running localhost health endpoint returned `ok`. Model readiness, predictions, ML Lab, grouped results and generated report now use the synthetic demonstration run.
- Existing personal ledger records and budgets were not seeded or changed by dataset generation/training.

Scores are computed, not fabricated, but measure an authored generator with repeated vocabulary and generator-assigned labels. Real student-expense accuracy, independent campus validation and Kaggle provenance remain unverified. The original real-data experiment is still unperformed; the newly authorized synthetic demonstration is complete.

Reproduce with `python -m src.generate_synthetic --activate`, `python -m src.train`, `python -m src.evaluate`, and `python -m src.report`. See `docs/synthetic_dataset.md` for the authored taxonomy and limitations.

---

## Historical verification before the synthetic-data request

The following records the initial delivery and its then-existing dataset blocker; its pending states are historical.

# Executed verification

Session date: **2026-10-10**. Environment: macOS ARM64, Python **3.9.6**, isolated `.venv`. Python 3.11 is recommended for new installations; the installed versions were exercised on the available compatible interpreter. Direct versions are pinned in `requirements.txt`; resolved transitive versions are recorded in `requirements-lock.txt`.

| Check | Observed outcome |
|---|---|
| Repository/data inspection | Initially empty directory; no existing code, dataset or applicable AGENTS.md found |
| Candidate source browsing | Kaggle data page exposed no readable file listing/schema; actual dataset contents and permissions remain unverified |
| Dependency installation | Completed after retrying a timed-out download |
| `python -m pip check` | No broken requirements found |
| Python syntax compilation | Passed; macOS's default bytecode cache required using a temporary cache path |
| `pytest -q` | **35 passed in 2.80 seconds** on the final run |
| Four-model fixture training | All required estimators fitted successfully; CV selection, shared holdout, grouped sensitivity, metrics, figures, persistence and reload exercised in temporary test directories |
| Streamlit AppTest | Six empty-state pages, manual save, populated overview/analytics/budgets, trained ML Lab and category suggestion passed |
| `python -m src.train` | Expected exit 2: verified `data/dataset.json` missing; no production model created |
| `python -m src.evaluate` | Expected exit 2: no completed real-data benchmark |
| `python -m src.report` | Generated `reports/submission.md` with explicit pending results and verified bibliography |
| Streamlit server | Started on `http://127.0.0.1:8501`; localhost binding required sandbox approval |
| Local health endpoint | `/_stcore/health` returned `ok` after sandbox-approved localhost access |

The first test run found a Streamlit magic-rendering issue caused by a conditional expression returning a UI element. Replacing it with explicit control flow fixed both empty and trained app states. A compatible pyparsing version was pinned to remove upstream Matplotlib deprecation warnings. The final test run has no warning summary. Pip reports that its user cache is not writable in the sandbox; package compatibility checks still succeed.

Raw test output: `reports/verification/pytest.txt`. Dependency check output: `reports/verification/dependencies.txt`. No fixture models or fixture performance values are published as project results. All synthetic training data originates inside tests and is written only to temporary test locations when exercising file-based interfaces.

## Completed implementation

- Six-page Streamlit app with local ledger, validated imports, category suggestions/corrections, analytics, exports, budgets, honest readiness and ML Lab states.
- Four exact required TF-IDF pipelines, training-only CV, CV-based selection, shared immutable holdout, diagnostics and optional-by-feasibility independent grouped sensitivity evaluation.
- Model-appropriate uncalibrated probabilities or SVM margins, vocabulary checks and documented review heuristics.
- SQLite parameterized operations, integer monetary storage, batch import validation and duplicate handling.
- Calendar-aware budgets, overspending, daily spending room and transparent pace estimates.
- Dataset/source manifest validation, reproducible cleaning audits, versioned local artifacts, report/figure generation and documentation.
- Dataset/model cards, methodology, literature review, academic report draft, README and viva walkthrough.

## Remaining blocker and limits

**A permitted, documented, labelled dataset must be supplied and verified.** Follow `data/README.md`, inspect actual fields and source taxonomy, complete the source manifest, then execute training/evaluation. Until then there are no measured real-data model scores, selected deployment model, genuine-data confusion matrices or campus-generalization claims. The overall experimental definition of done is therefore not yet met.

A public deployment, consented campus validation, visual browser screenshot review and a typeset PDF page-count check were not performed. The deliverable includes a Markdown report draft sized for approximately ten pages; verify its final page count after typesetting. The local server check and Streamlit interaction tests are completed; no hosted deployment URL is claimed.
