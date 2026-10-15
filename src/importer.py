"""CSV preview uses the same validation and inference as manual input."""
import pandas as pd

from src.database import validate_transaction
from src.predict import predict_expense


def prepare_import(source, model=None, model_name=None):
    frame = pd.read_csv(source, dtype=str, keep_default_na=False)
    missing = {"date", "description", "amount"} - set(frame.columns)
    if missing:
        raise ValueError(f"CSV requires columns date, description, amount; missing {sorted(missing)}.")
    if frame.empty or len(frame) > 10000:
        raise ValueError("Import between 1 and 10,000 rows per file.")
    rows, errors = [], []
    for index, row in frame.iterrows():
        record = row.to_dict()
        record["category"] = record.get("category", "").strip()
        record["needs_review"] = str(record.get("needs_review", "")).lower() in {"true", "1", "yes"}
        try:
            if not record["category"]:
                if model is None:
                    raise ValueError("Category is required until a trained model is available.")
                prediction = predict_expense(model, record["description"])
                record.update(category=prediction["category"], predicted_category=prediction["category"],
                              needs_review=prediction["needs_review"], model_name=model_name)
            valid = validate_transaction(record)
            rows.append({**record, "description": valid["description"], "amount": valid["amount_cents"] / 100,
                         "category": valid["category"]})
        except (ValueError, TypeError) as exc:
            errors.append({"csv_row": index + 2, "error": str(exc)})
    return rows, errors
