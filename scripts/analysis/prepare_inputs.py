"""Prepare analysis inputs from the deposited workbooks and frozen features."""

from pathlib import Path
import shutil

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "analysis_outputs" / "reproduction"

SHEETS = [
    ("database/mycoextract.xlsx", "Records", "dataset/mycoextract_primary_database_v20260530_reviewed.csv"),
    ("validation/extraction_validation.xlsx", "Record_scores", "validation/record_evaluation/per_paper_ledger.csv"),
    ("validation/extraction_validation.xlsx", "Document_scores", "validation/document_evaluation/per_publication_document_ledger.csv"),
    ("validation/extraction_validation.xlsx", "Six_fields", "validation/field_and_source_audit/six_field_scores.csv"),
    ("validation/extraction_validation.xlsx", "Numerical_source_checks", "validation/field_and_source_audit/numerical_source_audit.csv"),
    ("modeling/kinetic_modeling.xlsx", "Nested_predictions", "reviewer_revision_20261007/nested_all_targets_20261010_oof_predictions.csv"),
]


def main():
    for workbook, sheet, relative_output in SHEETS:
        destination = OUT / relative_output
        destination.parent.mkdir(parents=True, exist_ok=True)
        pd.read_excel(DATA / workbook, sheet_name=sheet).to_csv(
            destination, index=False, encoding="utf-8-sig",
        )
    # Preserve full input precision for regression tree split selection.
    measurements = OUT / "data" / "kinetic_modeling_dataset.csv"
    measurements.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DATA / "modeling" / measurements.name, measurements)

    features = OUT / "features" / "feature_matrix.npz"
    features.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DATA / "modeling" / "feature_matrix.npz", features)
    print(f"Prepared analysis inputs in {OUT}")


if __name__ == "__main__":
    main()
