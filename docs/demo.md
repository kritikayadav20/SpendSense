# Five-minute viva demonstration

1. Activate `.venv`, run `streamlit run app.py` and open the local URL. Explain that data/model readiness is explicit. Do not present fixture scores as research results.
2. Overview: select the current month. Show the recorded total and empty-state guidance, or your own existing ledger.
3. Add & Import: enter a non-sensitive purchase and a personal category. The active model uses the labelled synthetic demonstration corpus. Request a category, explain probability versus SVM margin, check/correct it, and save.
4. Transactions & Analytics: search the description, change its category, inspect charts, and download the local CSV. Import a user-supplied valid CSV twice to show duplicate handling. Do not claim manually typed demonstration data is benchmark data.
5. Budget Coach: set a monthly budget, add a category limit, explain the formula and show the explicit overspend if recorded spending exceeds the limit. Change reporting month to discuss past/future handling.
6. ML Lab: show the synthetic-data notice, training CV selection, the identical holdout, all four metrics, a minority-class confusion and the tree visualization. Compare primary and grouped-template scores; explain why neither establishes accuracy on real expenses.
7. Methodology: explain why macro-F1 is primary, why TF-IDF lives inside Pipeline, why test labels never guide tuning and why new-merchant generalization is unproven.

Useful viva questions: Why can NB accept TF-IDF? Why is an SVM score not a probability? Why can a tree return 100% and still be wrong? How can duplicated descriptions inflate a random-split score? Why does a pace estimate fail after a rent payment? What does source verification prove, and what does it not prove?

No public hosting is required. Stop the local server with Ctrl+C. Back up the local SQLite file privately if needed; never commit personal data.
