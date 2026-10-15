"""Evaluate fixed fitted models on the saved shared holdout. Never retrains."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from src.data_loader import ROOT
from src.predict import load_model


def active_run(root=ROOT):
    pointer = Path(root) / "models" / "active.json"
    if not pointer.exists():
        raise FileNotFoundError("No completed benchmark. Follow data/README.md and run python -m src.train.")
    return json.loads(pointer.read_text())["run_id"]


def metrics_for_model(model, test):
    start = time.perf_counter()
    predicted = model.predict(test.description)
    elapsed = time.perf_counter() - start
    metrics = {"accuracy": float(accuracy_score(test.category, predicted)),
               "prediction_ms_per_record": elapsed * 1000 / len(test)}
    for average in ["macro", "weighted"]:
        p, r, f, _ = precision_recall_fscore_support(test.category, predicted, average=average, zero_division=0)
        metrics.update({f"{average}_precision": float(p), f"{average}_recall": float(r), f"{average}_f1": float(f)})
    return metrics, predicted


def evaluate_run(root=ROOT, run_id=None):
    from src.train import MODEL_NAMES, write_json
    from src.visualization import save_benchmark_figures
    root = Path(root)
    run_id = run_id or active_run(root)
    model_dir = root / "models" / run_id
    metadata = json.loads((model_dir / "metadata.json").read_text())
    test_path = root / "data" / "processed" / run_id / "test.csv"
    if hashlib.sha256(test_path.read_bytes()).hexdigest() != metadata["test_sha256"]:
        raise ValueError("Holdout checksum mismatch. Refusing evaluation on changed records.")
    test = pd.read_csv(test_path, dtype={"description": str, "category": str}, keep_default_na=False)
    metrics_dir, figures_dir = root / "reports" / "metrics" / run_id, root / "reports" / "figures" / run_id
    metrics_dir.mkdir(parents=True, exist_ok=True)
    rows, details, models = [], {}, {}
    classes = metadata["classes"]
    for name in MODEL_NAMES:
        model = load_model(model_dir / f"{name}.joblib")
        models[name] = model
        metrics, predicted = metrics_for_model(model, test)
        row = {"model": name, "display_name": MODEL_NAMES[name], "provenance": metadata["dataset"]["provenance"], **metrics,
               "cv_macro_f1": metadata["models"][name]["cv_macro_f1"],
               "training_seconds": metadata["models"][name]["training_seconds"]}
        rows.append(row)
        report = classification_report(test.category, predicted, labels=classes, output_dict=True, zero_division=0)
        details[name] = {"per_class": {c: report[c] for c in classes},
                         "confusion_matrix": confusion_matrix(test.category, predicted, labels=classes).tolist(),
                         "classes": classes}
        wrong = test.category.to_numpy() != predicted
        # No descriptions or user identifiers in shared diagnostic outputs.
        errors = pd.DataFrame({"source_row": test.source_row[wrong], "actual": test.category[wrong],
                               "predicted": predicted[wrong], "description": "[withheld for privacy]"})
        errors.head(50).to_csv(metrics_dir / f"{name}_errors.csv", index=False)
        pd.DataFrame(details[name]["per_class"]).T.to_csv(metrics_dir / f"{name}_per_class.csv")
    comparison = pd.DataFrame(rows)
    comparison.to_csv(metrics_dir / "comparison.csv", index=False)
    write_json(metrics_dir / "comparison.json", rows)
    write_json(metrics_dir / "details.json", details)
    write_json(metrics_dir / "dataset_profile.json", {**metadata["profile"], "cleaning": metadata["cleaning"]})
    save_benchmark_figures(comparison, details, models, figures_dir)
    selected = comparison.set_index("model").loc[metadata["selected_model"]]
    provenance_note = ("**Synthetic demonstration only.** These scores measure generated examples, not real-world accuracy. Repeated vocabulary and authored label rules can make this task easier than real transactions."
                       if metadata["dataset"]["provenance"] == "synthetic" else f"Data provenance: {metadata['dataset']['provenance']}.")
    lines = ["# Executed benchmark", "", provenance_note, "", f"Run: `{run_id}`. Source: {metadata['dataset']['source_url']}", "",
        f"Selected using training CV macro-F1: **{selected.display_name}**. Held-out macro-F1: {selected.macro_f1:.4f}; accuracy: {selected.accuracy:.4f}.", "",
        "| Model | CV macro-F1 | Test macro-F1 | Accuracy | Training seconds |",
        "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['display_name']} | {row['cv_macro_f1']:.4f} | {row['macro_f1']:.4f} | {row['accuracy']:.4f} | {row['training_seconds']:.3f} |")
    lines += ["", "Training time includes grid search and refit. Latency is batch wall time per record, not a production SLA.",
              "ROC-AUC is omitted: the required primary comparison uses class predictions and macro-F1, with no probability calibration.",
              "Selection strength: highest training CV macro-F1 under this split. Limits: uncalibrated scores, source-domain dependence, and no evidence of unseen-merchant or campus generalization.", "",
              f"![Comparison](figures/{run_id}/comparison.png)", ""]
    for name in MODEL_NAMES:
        lines += [f"![{MODEL_NAMES[name]} confusion matrix](figures/{run_id}/{name}_confusion.png)",
                  f"![{MODEL_NAMES[name]} per-class F1](figures/{run_id}/{name}_per_class.png)"]
    lines += [f"![Decision tree](figures/{run_id}/decision_tree.png)", "",
              "Misclassification examples: local metrics files contain source row, actual and predicted class; descriptions are withheld. Raw rows can be examined privately."]
    (root / "reports" / "results.md").write_text("\n".join(lines))
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run")
    args = parser.parse_args()
    try:
        print(evaluate_run(run_id=args.run).to_string(index=False))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Evaluation blocked: {exc}\n")


if __name__ == "__main__":
    main()
