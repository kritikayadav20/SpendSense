# Model card

**Status:** all four required classifiers have been trained on the explicitly authorized, locally authored synthetic demonstration corpus. The active model is **Linear SVM**, selected using training-only CV macro-F1. Run: `20261010T121905Z-f76a69e2`. The demonstration dataset is separate from temporary unit-test fixtures; no real-data validation is claimed.

**Intended use:** suggest one source-defined category for a short transaction description, with human correction. Not intended for lending, financial advice, fraud detection or autonomous banking decisions.

**Inputs:** one validated string (1–1,000 characters with letter/number content), mapped to the fitted TF-IDF pipeline. Amount/date are excluded. **Outputs:** source class label, uncalibrated probabilities when supported or SVM decision margin, review reasons, recognized terms. The class order is read from the fitted estimator.

**Models:** Logistic Regression, Multinomial NB, LinearSVC and DecisionTreeClassifier only. See methodology for fixed search spaces. Selection uses training CV macro-F1, not holdout performance. The complete trained pipeline is persisted with Joblib. Environment versions, source hash, seeds, row-level exclusions, split membership and CV candidate scores are recorded with each immutable run. A new active pointer is published only after successful primary reports.

**Performance:** actual synthetic-corpus measurements are available. Consult the ML Lab and `reports/metrics/<run>/comparison.json`; confusion matrices and per-class charts use actual predictions. These scores measure the limited generator, not real expenses. High random/description-disjoint scores cannot establish unseen-merchant or different-college generalization. The grouped template sensitivity analysis is limited to a simple text proxy.

**Review policy:** probability < 0.60 or SVM margin < 0.50; no recognized vocabulary always triggers review. Values are model-dependent heuristics. Tree leaf proportions may be misleadingly extreme. No calibration claim is made, and the review flag is not a prediction interval or error guarantee.

**Limitations:** source taxonomy may omit campus expenses; ambiguous mixed purchases have only one label; misspellings, unseen vocabulary and domain shift can reduce accuracy. Source class imbalance, labels and synthetic provenance affect validity. Recognized terms are only an inspection aid. Tree plots expose learned vocabulary and should be reviewed before publication.

**Update policy:** obtaining a new verified source and explicitly rerunning training creates a new benchmark run. Do not repeatedly change parameters to improve a viewed test score; use a new untouched final set for a revised experiment. Local ledger corrections are not ingested by the trainer.
