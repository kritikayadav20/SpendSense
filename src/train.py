"""Train exactly four pipelines; select by training CV macro-F1, then evaluate once."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import time
import uuid
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src.data_loader import ROOT, load_verified_dataset
from src.preprocessing import clean_dataset, normalize_text, split_dataset

MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "multinomial_nb": "Multinomial Naive Bayes",
    "linear_svm": "Linear SVM",
    "decision_tree": "Decision Tree",
}


def model_specs(seed=42):
    return {
        "logistic_regression": (LogisticRegression(max_iter=4000, random_state=seed), {"classifier__C": [0.5, 2.0]}),
        "multinomial_nb": (MultinomialNB(), {"classifier__alpha": [0.1, 0.5, 1.0]}),
        "linear_svm": (LinearSVC(C=1.0, dual="auto", max_iter=10000, random_state=seed), {"classifier__C": [0.5, 2.0]}),
        "decision_tree": (DecisionTreeClassifier(random_state=seed), {
            "classifier__max_depth": [12, 30], "classifier__min_samples_leaf": [2, 5],
            "classifier__ccp_alpha": [0.0, 0.001]}),
    }


def template_key(text):
    """Conservative proxy for repeated templates; not a merchant identity detector."""
    return re.sub(r"\W+", " ", re.sub(r"\d+", " NUM ", text.casefold())).strip()


def fit_models(train, seed=42, groups=None):
    folds = min(5, int(train.category.value_counts().min()))
    if groups is None:
        cv = list(StratifiedKFold(folds, shuffle=True, random_state=seed).split(train.description, train.category))
    else:
        folds = min(folds, int(train.assign(group=groups).groupby("category").group.nunique().min()))
        if folds < 2:
            raise ValueError("Too few training template groups for independent grouped CV.")
        cv = list(StratifiedGroupKFold(folds, shuffle=True, random_state=seed).split(train.description, train.category, groups))
    if any(set(train.iloc[a].category) != set(train.category) or set(train.iloc[b].category) != set(train.category) for a, b in cv):
        raise ValueError("A CV training or validation fold lacks a class. More independent descriptions are required.")
    fitted, summaries = {}, {}
    for name, (classifier, grid) in model_specs(seed).items():
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(preprocessor=normalize_text, lowercase=False,
                                     strip_accents=None, ngram_range=(1, 2), max_features=20000,
                                     token_pattern=r"(?u)\b\w+\b", sublinear_tf=True)),
            ("classifier", classifier),
        ])
        # Shared case policy: preserve case in features; casefold only for duplicate detection.
        search = GridSearchCV(pipeline, grid, scoring="f1_macro", cv=cv,
                              n_jobs=1, refit=True, error_score="raise", return_train_score=False)
        start = time.perf_counter()
        retries = 0
        while True:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    search.fit(train.description, train.category)
                break
            except ConvergenceWarning:
                if retries >= 1 or not hasattr(classifier, "max_iter"):
                    raise RuntimeError(f"{name} failed to converge; no completed benchmark published.")
                pipeline.set_params(classifier__max_iter=classifier.max_iter * 5)
                retries += 1
        fitted[name] = search.best_estimator_
        summaries[name] = {
            "display_name": MODEL_NAMES[name], "cv_macro_f1": float(search.best_score_),
            "best_params": search.best_params_, "training_seconds": time.perf_counter() - start,
            "cv_folds": folds, "convergence_retries": retries,
            "candidates": [{"params": params, "mean_cv_macro_f1": float(mean), "std_cv_macro_f1": float(std)}
                           for params, mean, std in zip(search.cv_results_["params"],
                           search.cv_results_["mean_test_score"], search.cv_results_["std_test_score"])],
        }
    return fitted, summaries


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def train_benchmark(manifest_path, root=ROOT, seed=42):
    from src.evaluate import evaluate_run, metrics_for_model
    frame, manifest, profile = load_verified_dataset(manifest_path)
    cleaned, cleaning = clean_dataset(frame, manifest["description_column"], manifest["target_column"])
    train, test = split_dataset(cleaned, seed)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    root = Path(root)
    model_dir = root / "models" / run_id
    processed = root / "data" / "processed" / run_id
    model_dir.mkdir(parents=True)
    processed.mkdir(parents=True)
    fitted, summaries = fit_models(train, seed)
    selected = max(summaries, key=lambda name: summaries[name]["cv_macro_f1"])
    for name, model in fitted.items():
        joblib.dump(model, model_dir / f"{name}.joblib")
    train.to_csv(processed / "train.csv", index=False)
    test.to_csv(processed / "test.csv", index=False)
    cleaned.to_csv(processed / "cleaned.csv", index=False)
    from hashlib import sha256
    template_overlap = set(train.description.map(template_key)) & set(test.description.map(template_key))
    metadata = {
        "run_id": run_id, "created_utc": datetime.now(timezone.utc).isoformat(), "seed": seed,
        "dataset": manifest, "profile": profile, "cleaning": cleaning,
        "selected_model": selected, "selection_criterion": "Highest mean training CV macro-F1; ties use declared model order. Test set never selects a model.",
        "models": summaries, "classes": sorted(cleaned.category.unique().tolist()),
        "split": {"train_rows": len(train), "test_rows": len(test), "test_fraction": len(test) / len(cleaned),
                  "description_overlap": 0, "template_overlap_groups": len(template_overlap),
                  "train_source_rows": train.source_row.tolist(), "test_source_rows": test.source_row.tolist(),
                  "train_classes": train.category.value_counts().to_dict(), "test_classes": test.category.value_counts().to_dict()},
        "test_sha256": sha256((processed / "test.csv").read_bytes()).hexdigest(),
        "versions": {p: importlib.metadata.version(p) for p in ["scikit-learn", "pandas", "numpy", "scipy", "joblib"]},
        "python_version": platform.python_version(),
    }
    # Separate sensitivity analysis. Hyperparameters are re-tuned using only this split's training groups.
    groups = cleaned.description.map(template_key)
    if groups.nunique() == len(cleaned):
        metadata["grouped_evaluation"] = {"status": "not_applicable", "reason": "All template groups are unique; primary holdout is already description-disjoint."}
    else:
        grouped_result = {"status": "unavailable", "reason": "No valid five-fold template partition with all categories and sufficient CV groups."}
        try:
            splitter = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
            for tr, te in splitter.split(cleaned.description, cleaned.category, groups):
                group_train, group_test = cleaned.iloc[tr].reset_index(drop=True), cleaned.iloc[te].reset_index(drop=True)
                if set(group_train.category) != set(cleaned.category) or set(group_test.category) != set(cleaned.category):
                    continue
                train_groups = group_train.description.map(template_key)
                if group_train.assign(group=train_groups).groupby("category").group.nunique().min() < 2:
                    continue
                group_models, group_info = fit_models(group_train, seed, train_groups)
                grouped_result = {"status": "completed", "train_rows": len(tr), "test_rows": len(te),
                    "train_source_rows": group_train.source_row.tolist(), "test_source_rows": group_test.source_row.tolist(),
                    "group_overlap": 0, "grouping": "casefold, replace digit runs, collapse punctuation; not merchant identity",
                    "models": {name: {"metrics": metrics_for_model(model, group_test)[0], "tuning": group_info[name]}
                               for name, model in group_models.items()}}
                break
        except ValueError as exc:
            grouped_result["reason"] = str(exc)
        metadata["grouped_evaluation"] = grouped_result
    write_json(model_dir / "metadata.json", metadata)
    evaluate_run(root, run_id)
    # Publish only after all four models, metadata and reports succeed.
    pointer = root / "models" / "active.json"
    temp = pointer.with_suffix(".tmp")
    write_json(temp, {"run_id": run_id})
    temp.replace(pointer)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "dataset.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        result = train_benchmark(args.manifest, seed=args.seed)
        print(f"Completed {result['run_id']}; CV-selected model: {result['selected_model']}")
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Training blocked: {exc}\n")


if __name__ == "__main__":
    main()
