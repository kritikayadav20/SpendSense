"""Static academic figures; application charts use Plotly."""
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spendsense-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import plot_tree, export_text


def save_benchmark_figures(comparison, details, models, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    fig, ax = plt.subplots(figsize=(9, 4))
    comparison.set_index("display_name")[["macro_f1", "accuracy"]].plot.bar(ax=ax, color=["#147D73", "#687CBA"])
    ax.set(ylim=(0, 1.05), ylabel="Held-out score", xlabel="")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(directory / "comparison.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(15, 13))
    for ax, (name, detail) in zip(axes.flat, details.items()):
        labels = detail["classes"]
        ax.imshow(detail["confusion_matrix"], cmap="BuGn")
        ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels,
               xlabel="Predicted", ylabel="Actual", title=name.replace("_", " ").title())
        ax.tick_params(labelsize=7)
        plt.setp(ax.get_xticklabels(), rotation=65, ha="right")
    fig.tight_layout()
    fig.savefig(directory / "confusion_panel.png", dpi=140)
    plt.close(fig)
    for name, detail in details.items():
        labels = detail["classes"]
        size = min(24, max(6, len(labels) * 0.45))
        fig, ax = plt.subplots(figsize=(size, size))
        values = np.array(detail["confusion_matrix"])
        image = ax.imshow(values, cmap="BuGn")
        ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels,
               xlabel="Predicted", ylabel="Actual", title=name.replace("_", " ").title())
        plt.setp(ax.get_xticklabels(), rotation=70, ha="right")
        if len(labels) <= 20:
            for i in range(len(labels)):
                for j in range(len(labels)):
                    ax.text(j, i, str(values[i, j]), ha="center", va="center",
                            color="white" if values[i, j] > values.max() / 2 else "#182B45", fontsize=8)
        fig.colorbar(image, ax=ax, shrink=0.7)
        fig.tight_layout()
        fig.savefig(directory / f"{name}_confusion.png", dpi=140)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.3)))
        ax.barh(labels, [detail["per_class"][c]["f1-score"] for c in labels], color="#147D73")
        ax.set(xlim=(0, 1), xlabel="Per-class F1", title=name.replace("_", " ").title())
        fig.tight_layout()
        fig.savefig(directory / f"{name}_per_class.png", dpi=140)
        plt.close(fig)
    tree = models["decision_tree"]
    features = tree.named_steps["tfidf"].get_feature_names_out()
    fig, ax = plt.subplots(figsize=(18, 8))
    plot_tree(tree.named_steps["classifier"], feature_names=features, class_names=tree.classes_.tolist(),
              max_depth=2, filled=True, fontsize=8, ax=ax)
    ax.set_title("Decision Tree · first three levels · thresholds are TF-IDF weights")
    fig.tight_layout()
    fig.savefig(directory / "decision_tree.png", dpi=140)
    plt.close(fig)
    (directory / "decision_tree_rules.txt").write_text(
        export_text(tree.named_steps["classifier"], feature_names=list(features), max_depth=4),
        encoding="utf-8",
    )
