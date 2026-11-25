"""Reproducible reviewer checks on the frozen MycoExtract release.

This script does not alter the published source data or refit the original model
selection procedure. It evaluates a prespecified Ridge regressor under stricter
group splits and prepares percentage counts for the composition figure.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parents[2] / "analysis_outputs" / "reproduction"
OUT = BASE
DATA = BASE / "data" / "kinetic_modeling_dataset.csv"
FEATURES = BASE / "features" / "feature_matrix.npz"
DATABASE = BASE / "dataset" / "mycoextract_primary_database_v20260530_reviewed.csv"


def clean_sequence(value: str) -> str:
    return re.sub(r"\s+", "", str(value)).upper()


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[b] = a


def sequence_clusters(sequences: list[str]) -> dict[str, int]:
    """Single-linkage components at >=80% identity and shorter-sequence coverage."""
    unique = list(dict.fromkeys(sequences))
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -0.5
    uf = UnionFind(len(unique))
    for i, left in enumerate(unique):
        for j in range(i + 1, len(unique)):
            right = unique[j]
            aln = aligner.align(left, right)[0]
            counts = aln.counts()
            shorter = min(len(left), len(right))
            aligned = sum(e - s for s, e in aln.aligned[0])
            if counts.identities / shorter >= 0.8 and aligned / shorter >= 0.8:
                uf.union(i, j)
    roots = {uf.find(i): n for n, i in enumerate(sorted({uf.find(i) for i in range(len(unique))}))}
    return {s: roots[uf.find(i)] for i, s in enumerate(unique)}


def scores(y: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "n": len(y),
        "R2": float(r2_score(y, pred)) if len(y) >= 2 else np.nan,
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "MAE": float(mean_absolute_error(y, pred)),
    }


def model_analysis() -> dict:
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    feat = np.load(FEATURES)
    X = feat["X"]
    edim = int(feat["esm_dim"])
    assert len(df) == len(X) == 72
    seq = df.sequence.map(clean_sequence)
    clusters = sequence_clusters(seq.tolist())
    df["sequence_cluster_80"] = seq.map(clusters)
    df["sequence_clean"] = seq
    df["substrate_group"] = df.substrate_smiles_canonical
    df[["source_record_id", "enzyme_name", "substrate", "sequence_cluster_80", "substrate_group"]].to_csv(
        OUT / "model_group_assignments.csv", index=False, encoding="utf-8-sig"
    )
    split_rows, fold_rows, oof_rows = [], [], []
    targets = ["log10_kcat_s", "log10_kcat_Km_M_s", "log10_Km_uM"]
    for target in targets:
        mask = df[target].notna().to_numpy()
        yt = df.loc[mask, target].to_numpy(float)
        Xt = X[mask]
        for scheme, group_col in [
            ("exact_sequence", "sequence_clean"),
            ("sequence_cluster_80", "sequence_cluster_80"),
            ("unseen_substrate", "substrate_group"),
        ]:
            groups = df.loc[mask, group_col].to_numpy()
            splits = list(GroupKFold(n_splits=min(5, len(set(groups)))).split(Xt, yt, groups))
            for feature_name, subset in [
                ("enzyme_only", Xt[:, :edim]),
                ("substrate_only", Xt[:, edim:]),
                ("combined", Xt),
                ("mean_baseline", Xt),
            ]:
                pred = np.full(len(yt), np.nan)
                for fold, (train, test) in enumerate(splits):
                    model = (DummyRegressor(strategy="mean") if feature_name == "mean_baseline"
                             else make_pipeline(StandardScaler(), Ridge(alpha=10.0)))
                    model.fit(subset[train], yt[train])
                    pred[test] = model.predict(subset[test])
                    fold_rows.append({"target": target, "scheme": scheme, "feature_set": feature_name,
                                      "fold": fold + 1, "train_n": len(train), "test_n": len(test),
                                      **scores(yt[test], pred[test])})
                    for idx in test:
                        oof_rows.append({"target": target, "scheme": scheme, "feature_set": feature_name,
                                         "fold": fold + 1, "row": int(np.flatnonzero(mask)[idx]),
                                         "y_true": yt[idx], "y_pred": pred[idx]})
                split_rows.append({"target": target, "scheme": scheme, "feature_set": feature_name,
                                   "n_groups": len(set(groups)), **scores(yt, pred)})
    metrics = pd.DataFrame(split_rows)
    metrics.to_csv(OUT / "model_grouped_cv_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(OUT / "model_fold_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(oof_rows).to_csv(OUT / "model_oof_predictions.csv", index=False, encoding="utf-8-sig")
    return {"n_measurements": len(df), "unique_sequences": seq.nunique(),
            "sequence_clusters_80": len(set(clusters.values())),
            "unique_substrates": df.substrate_group.nunique(), "metrics": metrics.to_dict("records")}


def database_composition() -> dict:
    df = pd.read_csv(DATABASE, encoding="utf-8-sig")
    columns = df.columns.tolist()
    (OUT / "database_columns.txt").write_text("\n".join(columns), encoding="utf-8")
    rows = []
    for category, col in [
        ("substrate", "normalized_substrate_full_name"),
        ("enzyme", "normalized_enzyme_full_name"),
        ("organism", "normalized_organism_full_name"),
        ("measurement_type", "measurement_type"),
    ]:
        counts = df[col].fillna("Unspecified").replace("", "Unspecified").value_counts()
        for rank, (name, count) in enumerate(counts.items(), start=1):
            rows.append({"category": category, "rank": rank, "name": name,
                         "count": int(count), "denominator": len(df),
                         "percent": 100 * count / len(df)})
    pd.DataFrame(rows).to_csv(OUT / "database_composition_percentages.csv", index=False, encoding="utf-8-sig")
    return {"records": len(df), "papers": df.pdf_file.nunique(),
            "record_types": df.measurement_type.value_counts(dropna=False).to_dict(),
            "columns": columns}


if __name__ == "__main__":
    summary = {"model": model_analysis(), "database": database_composition()}
    (OUT / "reanalysis_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({"model": {k: v for k, v in summary["model"].items() if k != "metrics"},
                      "database": {k: v for k, v in summary["database"].items() if k != "columns"}}, indent=2))
