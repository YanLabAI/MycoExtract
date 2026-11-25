"""Paired leave-one-cluster-and-substrate-out sensitivity analysis.

For each observed (80%-identity sequence cluster, canonical substrate) pair,
test its records and train only on records sharing neither the cluster nor the
substrate. Each record receives one prediction; cross-arm records are excluded
from that fold. This is a retrospective sensitivity analysis, not an
independent/prospective validation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from reanalyse_existing_data import DATA, FEATURES, clean_sequence, sequence_clusters

HERE = Path(__file__).resolve().parents[2] / "analysis_outputs" / "reproduction"


def main() -> None:
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    features = np.load(FEATURES)
    matrix = features["X"]
    enzyme_dim = int(features["esm_dim"])
    assert len(df) == len(matrix) == 72
    clean = df.sequence.map(clean_sequence)
    clusters = sequence_clusters(clean.tolist())
    df["cluster_80"] = clean.map(clusters)
    df["substrate_group"] = df.substrate_smiles_canonical

    predictions = []
    folds = []
    summaries = []
    for target in ("log10_kcat_s", "log10_kcat_Km_M_s", "log10_Km_uM"):
        keep = df[target].notna().to_numpy()
        d = df.loc[keep].reset_index(drop=True)
        x = matrix[keep]
        y = d[target].to_numpy(float)
        cluster = d.cluster_80.to_numpy()
        substrate = d.substrate_group.to_numpy()
        pairs = d[["cluster_80", "substrate_group"]].drop_duplicates()
        subsets = {
            "enzyme_only": x[:, :enzyme_dim],
            "substrate_only": x[:, enzyme_dim:],
            "combined": x,
            "mean_baseline": x,
        }
        for feature_set, xx in subsets.items():
            pred = np.full(len(d), np.nan)
            for pair_id, (held_cluster, held_substrate) in enumerate(pairs.itertuples(index=False, name=None)):
                test = (cluster == held_cluster) & (substrate == held_substrate)
                train = (cluster != held_cluster) & (substrate != held_substrate)
                cross_arm = ~(train | test)
                assert test.sum() > 0 and train.sum() > 1
                assert not set(cluster[train]).intersection(set(cluster[test]))
                assert not set(substrate[train]).intersection(set(substrate[test]))
                model = (DummyRegressor(strategy="mean") if feature_set == "mean_baseline"
                         else make_pipeline(StandardScaler(), Ridge(alpha=10.0)))
                model.fit(xx[train], y[train])
                pred[test] = model.predict(xx[test])
                folds.append({
                    "target": target, "feature_set": feature_set, "pair_id": pair_id,
                    "cluster_80": held_cluster, "substrate_group": held_substrate,
                    "n_train": int(train.sum()), "n_test": int(test.sum()),
                    "n_excluded_cross_arm": int(cross_arm.sum()),
                    "test_R2": float(r2_score(y[test], pred[test])) if test.sum() >= 2 else np.nan,
                    "test_RMSE": float(np.sqrt(mean_squared_error(y[test], pred[test]))),
                })
            assert np.isfinite(pred).all()
            summaries.append({
                "target": target, "feature_set": feature_set, "n": len(d),
                "n_pairs": len(pairs), "pooled_R2": float(r2_score(y, pred)),
                "pooled_RMSE": float(np.sqrt(mean_squared_error(y, pred))),
                "pooled_MAE": float(mean_absolute_error(y, pred)),
            })
            for row_index, row in d.iterrows():
                predictions.append({
                    "target": target, "feature_set": feature_set,
                    "source_record_id": row.source_record_id,
                    "cluster_80": row.cluster_80,
                    "substrate_group": row.substrate_group,
                    "observed": y[row_index], "predicted": pred[row_index],
                })

    prefix = HERE / "joint_unseen_cluster_substrate_20261010"
    pd.DataFrame(predictions).to_csv(str(prefix) + "_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(folds).to_csv(str(prefix) + "_folds.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(summaries)
    summary.to_csv(str(prefix) + "_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print("Fold test/train ranges by target:")
    print(pd.DataFrame(folds).query("feature_set == 'combined'").groupby("target")[["n_train", "n_test", "n_excluded_cross_arm"]].agg(["min", "max"]).to_string())


if __name__ == "__main__":
    main()
