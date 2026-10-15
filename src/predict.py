"""Streamlit-independent feature mapping, inference and uncertainty handling."""
from pathlib import Path

import joblib
import numpy as np

from src.preprocessing import validate_description


def load_model(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Model unavailable. Train using a verified dataset first.")
    # Joblib uses pickle. Only load artifacts generated locally by this project.
    model = joblib.load(path)
    if not hasattr(model, "named_steps") or "tfidf" not in model.named_steps:
        raise ValueError("Expected a fitted TF-IDF classification pipeline.")
    return model


def map_features(description):
    """Only text enters the model. Amount and date are bookkeeping fields."""
    return [validate_description(description)]


def predict_expense(model, description):
    features = map_features(description)
    matrix = model.named_steps["tfidf"].transform(features)
    category = str(model.predict(features)[0])
    result = {"category": category, "needs_review": False, "probability": None,
              "decision_margin": None, "reasons": [], "classes": [str(x) for x in model.classes_]}
    if matrix.nnz == 0:
        result["reasons"].append("No known vocabulary; prediction is not supported by recognized words.")
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        result["probability"] = float(probabilities.max())
        result["class_probabilities"] = dict(zip(result["classes"], map(float, probabilities)))
        result["score_label"] = "Uncalibrated class probability estimate"
        result["explanation"] = "Probability estimates are not calibrated or a guarantee of correctness. Review threshold: < 0.60."
        if model.named_steps["classifier"].__class__.__name__ == "DecisionTreeClassifier":
            result["explanation"] += " Tree leaf proportions may be especially poorly calibrated, even at 100%."
        if result["probability"] < 0.6:
            result["reasons"].append("Maximum probability below the heuristic 0.60 review threshold.")
    else:
        scores = np.asarray(model.decision_function(features))
        if scores.ndim == 1:
            margin = float(abs(scores[0]))
            ranking = [-float(scores[0]), float(scores[0])]
        else:
            ranking = scores[0].tolist()
            ordered = sorted(ranking)
            margin = float(ordered[-1] - ordered[-2])
        result["decision_margin"] = margin
        result["class_decision_scores"] = dict(zip(result["classes"], ranking))
        result["score_label"] = "Decision margin (not probability)"
        result["explanation"] = "SVM: top-two score gap for multiclass, absolute distance from boundary for binary. Heuristic review threshold: < 0.50. Scores are not probabilities."
        if margin < 0.5:
            result["reasons"].append("Decision margin below the heuristic 0.50 review threshold.")
    result["needs_review"] = bool(result["reasons"])
    # Recognized terms are evidence of feature coverage, not causal explanations.
    vocab = model.named_steps["tfidf"].get_feature_names_out()
    indices = matrix.indices[np.argsort(matrix.data)[::-1]][:8]
    result["recognized_terms"] = vocab[indices].tolist()
    return result
