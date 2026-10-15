"""Inspect original CSVs and require a human-verified source manifest before training."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def validate_schema(frame, description_column, target_column):
    if description_column == target_column:
        raise ValueError("Description and target must be different columns.")
    missing = {description_column, target_column} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}; actual headers: {list(frame.columns)}")
    if frame.empty:
        raise ValueError("Dataset contains no records.")


def read_csv(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}. Follow data/README.md; no benchmark data is bundled.")
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])


def inspect_csv(path, description_column=None, target_column=None):
    frame = read_csv(path)
    result = {"filename": Path(path).name, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
              "records": len(frame), "columns": len(frame.columns), "headers": list(frame.columns),
              "missing_values": frame.isna().sum().to_dict(),
              "blank_values": {c: int(frame[c].fillna("").str.strip().eq("").sum()) for c in frame}}
    if description_column and target_column:
        validate_schema(frame, description_column, target_column)
        result["target_column"] = target_column
        result["description_column"] = description_column
        result["raw_class_distribution"] = frame[target_column].fillna("<missing>").value_counts().to_dict()
    return result


def load_verified_dataset(manifest_path):
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Verified dataset manifest missing: {path}. Follow data/README.md.")
    manifest = json.loads(path.read_text())
    required = ["file", "description_column", "target_column", "source_url", "license", "license_evidence",
                "provenance", "provenance_notes", "taxonomy", "taxonomy_evidence", "feature_definitions", "sha256"]
    for key in required:
        if not manifest.get(key):
            raise ValueError(f"Dataset manifest requires {key}.")
    if manifest.get("verified") is not True:
        raise ValueError("Verify source, license, taxonomy and schema, then set verified=true in the manifest.")
    if manifest["provenance"] not in {"real", "synthetic", "uncertain"}:
        raise ValueError("provenance must be real, synthetic, or uncertain.")
    if str(manifest["license"]).lower() in {"unknown", "pending", "unverified"}:
        raise ValueError("Document a license or explicit permission before training.")
    if not isinstance(manifest["taxonomy"], list) or len(manifest["taxonomy"]) < 2:
        raise ValueError("taxonomy must list the documented category labels.")
    csv_path = (path.parent / manifest["file"]).resolve()
    profile = inspect_csv(csv_path, manifest["description_column"], manifest["target_column"])
    if profile["sha256"] != manifest["sha256"]:
        raise ValueError("Dataset checksum differs from the verified manifest; inspect it again.")
    frame = read_csv(csv_path)
    actual = set(frame[manifest["target_column"]].dropna().str.strip()) - {""}
    if actual - set(manifest["taxonomy"]):
        raise ValueError(f"Undocumented target labels: {sorted(actual - set(manifest['taxonomy']))}")
    return frame, manifest, profile


def main():
    parser = argparse.ArgumentParser(description="Inspect a local labelled CSV without training or displaying descriptions.")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--description-column")
    parser.add_argument("--target-column")
    args = parser.parse_args()
    try:
        print(json.dumps(inspect_csv(args.csv, args.description_column, args.target_column), indent=2))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Dataset inspection blocked: {exc}\n")


if __name__ == "__main__":
    main()
