"""Conservative normalization and auditable, description-disjoint splitting."""
import re
import unicodedata

import pandas as pd
from sklearn.model_selection import train_test_split


def normalize_text(value):
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


def validate_description(value):
    text = normalize_text(value)
    if not text or not any(ch.isalnum() for ch in text):
        raise ValueError("Enter a description containing letters or numbers.")
    if len(text) > 1000:
        raise ValueError("Description must be at most 1,000 characters.")
    return text


def clean_dataset(frame, description_column, target_column):
    from src.data_loader import validate_schema
    validate_schema(frame, description_column, target_column)
    data = pd.DataFrame({
        "description": frame[description_column].map(normalize_text),
        "category": frame[target_column].map(normalize_text),
        "source_row": range(2, len(frame) + 2),
    })
    data["description_key"] = data.description.str.casefold()
    invalid = (data.description.eq("") | data.category.eq("") |
               ~data.description.str.contains(r"\w", regex=True) | data.description.str.len().gt(1000))
    exclusions = [{"source_row": int(row.source_row), "reason": "missing_or_invalid_text_or_label"}
                  for row in data[invalid].itertuples()]
    data = data[~invalid].copy()
    conflicts = data.groupby("description_key").category.nunique()
    conflict_mask = data.description_key.isin(conflicts[conflicts > 1].index)
    exclusions.extend({"source_row": int(row.source_row), "reason": "conflicting_labels"}
                      for row in data[conflict_mask].itertuples())
    data = data[~conflict_mask].copy()
    duplicates = data.duplicated("description_key", keep="first")
    exclusions.extend({"source_row": int(row.source_row), "reason": "duplicate_description"}
                      for row in data[duplicates].itertuples())
    data = data[~duplicates].reset_index(drop=True)
    audit = {"raw_rows": len(frame), "retained_rows": len(data),
             "exclusion_counts": pd.Series([x["reason"] for x in exclusions], dtype=str).value_counts().to_dict(),
             "exclusions": exclusions, "class_distribution": data.category.value_counts().to_dict(),
             "removed_categories": sorted(set(frame[target_column].map(normalize_text)) - set(data.category) - {""}),
             "policy": "NFKC + whitespace; casefold key; remove all label conflicts, keep first exact normalized description."}
    return data, audit


def split_dataset(data, seed=42):
    counts = data.category.value_counts()
    if len(counts) < 2 or counts.min() < 5:
        raise ValueError("Need at least two categories and five unique descriptions per category after cleaning for holdout and CV. Collect data; do not duplicate rows.")
    n_test = max(len(counts), round(len(data) * 0.2))
    train, test = train_test_split(data, test_size=n_test, stratify=data.category, random_state=seed)
    if set(train.description_key) & set(test.description_key):
        raise ValueError("Description leakage detected. Run clean_dataset before splitting.")
    if train.category.value_counts().min() < 2:
        raise ValueError("Insufficient training examples for stratified cross-validation.")
    return train.reset_index(drop=True), test.reset_index(drop=True)
