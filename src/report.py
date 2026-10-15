"""Assemble a submission draft using executed artifacts, or explicit pending results."""
import json
from pathlib import Path

import pandas as pd

from src.data_loader import ROOT
from src.evaluate import active_run


def build_report(root=ROOT, docs_root=None):
    root = Path(root)
    docs_root = Path(docs_root) if docs_root else root / "docs"
    draft = (docs_root / "report.md").read_text()
    try:
        run = active_run(root)
    except FileNotFoundError:
        run = None
    if run:
        meta = json.loads((root / "models" / run / "metadata.json").read_text())
        metrics_dir = root / "reports" / "metrics" / run
        comparison = pd.read_csv(metrics_dir / "comparison.csv")
        detail = json.loads((metrics_dir / "details.json").read_text())
        best = comparison.set_index("model").loc[meta["selected_model"]]
        per_class = detail[meta["selected_model"]]["per_class"]
        weakest = min(per_class, key=lambda c: per_class[c]["f1-score"])
        rows = [f"Executed run: `{run}`. Verified source manifest: {meta['dataset']['source_url']}. Provenance: **{meta['dataset']['provenance']}**.", "",
                f"Retained {meta['cleaning']['retained_rows']} of {meta['profile']['records']} records. Training: {meta['split']['train_rows']}; test: {meta['split']['test_rows']}. Test description overlap: {meta['split']['description_overlap']}.", "",
                "| Model | CV macro-F1 | Test macro-F1 | Accuracy | Weighted-F1 | Training s |",
                "|---|---:|---:|---:|---:|---:|"]
        for row in comparison.itertuples():
            rows.append(f"| {row.display_name} | {row.cv_macro_f1:.4f} | {row.macro_f1:.4f} | {row.accuracy:.4f} | {row.weighted_f1:.4f} | {row.training_seconds:.3f} |")
        grouped = meta.get("grouped_evaluation", {})
        if grouped.get("status") == "completed":
            rows += ["", "Independent template-group sensitivity evaluation (same generator/domain; not external validation):", "",
                     "| Model | Grouped test macro-F1 | Grouped test accuracy |", "|---|---:|---:|"]
            for name, result in grouped["models"].items():
                metrics = result["metrics"]
                rows.append(f"| {meta['models'][name]['display_name']} | {metrics['macro_f1']:.4f} | {metrics['accuracy']:.4f} |")
        rows += ["", f"**{best.display_name}** was selected by training CV macro-F1 ({best.cv_macro_f1:.4f}). Its held-out macro-F1 is {best.macro_f1:.4f}. Its lowest-F1 class is **{weakest}** ({per_class[weakest]['f1-score']:.4f}, support {per_class[weakest]['support']:.0f}). This ranking is within the source domain, not external validation.", "",
                 f"Template-group sensitivity status: **{meta['grouped_evaluation']['status']}**. " + meta["grouped_evaluation"].get("reason", "The independent split's complete results and row IDs are stored in run metadata and displayed in ML Lab."), "",
                 f"![Accuracy and macro-F1](figures/{run}/comparison.png)",
                 f"![All four confusion matrices](figures/{run}/confusion_panel.png)",
                 f"![Selected model per-class F1](figures/{run}/{meta['selected_model']}_per_class.png)",
                 f"![Tree first three levels](figures/{run}/decision_tree.png)"]
        start, rest = draft.split("<!-- RESULTS -->", 1)
        _, end = rest.split("<!-- END RESULTS -->", 1)
        draft = start + "\n\n" + "\n".join(rows) + "\n\n" + end
        draft = draft.replace("**Experimental status: pending verified dataset.**", f"**Experimental status: completed run `{run}`.**")
        draft = draft.replace("At this stage a verified labelled dataset has not been acquired, so model results and real-world usefulness remain unmeasured.",
                              "The results section contains executed source-domain measurements; real-world campus usefulness remains unmeasured.")
        draft = draft.replace("Data access is the present experimental blocker.", "The executed dataset's permission and provenance are recorded in the run manifest.")
        draft = draft.replace("Discussion must be completed after execution:", "For final academic discussion:")
        before_dataset, remainder = draft.split("### 4. Dataset and provenance", 1)
        _, after_dataset = remainder.split("### 5. Cleaning and TF-IDF representation", 1)
        source = meta["dataset"]
        if source["provenance"] == "synthetic":
            draft = draft.replace("**Experimental status: completed run", "**Experimental status: completed SYNTHETIC-data demonstration run")
            draft = draft.replace("The results section contains executed source-domain measurements; real-world campus usefulness remains unmeasured.",
                                  "The user explicitly authorized a locally generated synthetic dataset. The results section measures this artificial corpus only; real-world campus accuracy remains unmeasured. Template vocabulary and generator-assigned labels limit external validity.")
            # Carry the provenance notice through the section reconstruction below.
            before_dataset = draft.split("### 4. Dataset and provenance", 1)[0]
        profile = meta["profile"]
        dataset_section = (
            f"### 4. Dataset and provenance\n\n"
            f"Source: {source['source_url']}. Original file: `{profile['filename']}`; "
            f"{profile['records']} records and {profile['columns']} columns. "
            f"Mapped input: `{source['description_column']}`; target: `{source['target_column']}`. "
            f"The cleaned experiment contains {len(meta['classes'])} source-defined categories.\n\n"
            f"License/permission: {source['license']}. Evidence: {source['license_evidence']}. "
            f"Provenance: **{source['provenance']}**. {source['provenance_notes']}\n\n"
            f"Missing text: {profile['missing_values'][source['description_column']]}; "
            f"missing target: {profile['missing_values'][source['target_column']]}. "
            f"Full headers, blank-value counts, class distributions and exclusions are in "
            f"`metrics/{run}/dataset_profile.json`. The source manifest records the taxonomy "
            f"evidence, field definitions and SHA-256. Verification is a human attestation, "
            f"not automated proof of origin. Only description text enters the model.\n\n"
        )
        draft = before_dataset + dataset_section + "### 5. Cleaning and TF-IDF representation" + after_dataset
    # Historical acquisition account remains factual even after a later source is verified.
    draft += "\n\n" + (docs_root / "literature_review.md").read_text(encoding="utf-8")
    destination = root / "reports" / "submission.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(draft, encoding="utf-8")
    return destination


if __name__ == "__main__":
    print(build_report())
