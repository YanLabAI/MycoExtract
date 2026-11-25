"""Nested exact-sequence-grouped model selection for all three kinetic targets.

Eight fixed candidate regressors are compared by pooled RMSE in four inner
GroupKFold folds. The selected model is refit on each outer training set and
scored on one of five held-out exact-sequence groups. No hyperparameters are
tuned beyond choosing one of the eight prespecified configurations.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from train_kinetics_model import get_models

HERE = Path(__file__).resolve().parents[2] / "analysis_outputs" / "reproduction"
MODEL_DIR = HERE

TARGETS = ["log10_kcat_s", "log10_kcat_Km_M_s", "log10_Km_uM"]
STEM = HERE / "nested_all_targets_20261010"


def score(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    rho = float(spearmanr(y, prediction).statistic) if len(y) > 1 else np.nan
    return {
        "R2": float(r2_score(y, prediction)) if len(y) > 1 else np.nan,
        "RMSE": float(np.sqrt(mean_squared_error(y, prediction))),
        "MAE": float(mean_absolute_error(y, prediction)),
        "Spearman_rho": rho,
    }


def fit_model(name: str, x: np.ndarray, y: np.ndarray):
    model = get_models()[name]
    if name in ("RandomForest", "XGBoost"):
        model.set_params(n_jobs=1)
    model.fit(x, y)
    return model


def main() -> None:
    df = pd.read_csv(MODEL_DIR / "data" / "kinetic_modeling_dataset.csv", encoding="utf-8-sig")
    matrix = np.load(MODEL_DIR / "features" / "feature_matrix.npz")["X"]
    assert len(df) == len(matrix) == 72
    all_candidates = list(get_models())
    assert len(all_candidates) == 8
    inner_rows, outer_rows, prediction_rows, summary_rows = [], [], [], []

    for target in TARGETS:
        valid = df[target].notna().to_numpy()
        original_row = np.flatnonzero(valid)
        x = matrix[valid]
        y = df.loc[valid, target].to_numpy(float)
        groups = df.loc[valid, "sequence"].map(lambda s: re.sub(r"\s+", "", s).upper()).to_numpy()
        outer_splits = GroupKFold(n_splits=5).split(x, y, groups)
        pred = np.full(len(y), np.nan)
        for outer_fold, (train, test) in enumerate(outer_splits, start=1):
            assert not set(groups[train]).intersection(set(groups[test]))
            candidates = []
            inner_splits = list(GroupKFold(n_splits=4).split(x[train], y[train], groups[train]))
            for name in all_candidates:
                inner_pred = np.full(len(train), np.nan)
                for inner_train, inner_test in inner_splits:
                    model = fit_model(name, x[train][inner_train], y[train][inner_train])
                    inner_pred[inner_test] = model.predict(x[train][inner_test])
                assert np.isfinite(inner_pred).all()
                metrics = score(y[train], inner_pred)
                candidates.append((metrics["RMSE"], name))
                inner_rows.append({"target": target, "outer_fold": outer_fold,
                                   "candidate": name, "n_inner_pool": len(train),
                                   **{f"inner_pooled_{k}": v for k, v in metrics.items()}})
            best_rmse, best_name = min(candidates)
            selected = fit_model(best_name, x[train], y[train])
            pred[test] = selected.predict(x[test])
            outer_metrics = score(y[test], pred[test])
            outer_rows.append({"target": target, "outer_fold": outer_fold,
                               "selected_model": best_name, "inner_selected_RMSE": best_rmse,
                               "train_n": len(train), "test_n": len(test),
                               "train_groups": len(set(groups[train])), "test_groups": len(set(groups[test])),
                               **outer_metrics})
            for i in test:
                prediction_rows.append({"target": target, "outer_fold": outer_fold,
                                        "row": int(original_row[i]),
                                        "source_record_id": df.iloc[original_row[i]].source_record_id,
                                        "selected_model": best_name, "y_true": y[i], "y_pred": pred[i]})
            print(f"{target} fold {outer_fold}: selected {best_name}, test n={len(test)}", flush=True)
        assert np.isfinite(pred).all()
        pooled = score(y, pred)
        folds = pd.DataFrame(outer_rows).query("target == @target")
        summary_rows.append({
            "target": target, "n": len(y), "n_exact_sequence_groups": len(set(groups)),
            "outer_folds": 5, "inner_folds": 4,
            "selected_models": ";".join(folds.selected_model.tolist()),
            **{f"pooled_{k}": v for k, v in pooled.items()},
            **{f"fold_{k}_mean": folds[k].mean() for k in pooled},
            **{f"fold_{k}_sd": folds[k].std(ddof=1) for k in pooled},
            **{f"fold_{k}_min": folds[k].min() for k in pooled},
            **{f"fold_{k}_max": folds[k].max() for k in pooled},
        })

    pd.DataFrame(inner_rows).to_csv(str(STEM) + "_inner_candidates.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(outer_rows).to_csv(str(STEM) + "_outer_folds.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(prediction_rows).to_csv(str(STEM) + "_oof_predictions.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(str(STEM) + "_summary.csv", index=False, encoding="utf-8-sig")
    (Path(str(STEM) + "_method.json")).write_text(json.dumps({
        "data_rows": 72,
        "targets": TARGETS,
        "group": "exact amino-acid sequence",
        "outer": "5-fold GroupKFold",
        "inner": "4-fold GroupKFold within each outer training set",
        "selection": "minimum pooled inner RMSE among eight fixed regressors",
        "uncertainty": "descriptive spread across five outer folds, not a confidence interval",
        "features": "frozen enzyme ESM-2 and substrate RDKit feature matrix",
    }, indent=2), encoding="utf-8")
    print(summary[["target", "n", "pooled_R2", "pooled_RMSE", "pooled_MAE", "pooled_Spearman_rho"]].to_string(index=False))


if __name__ == "__main__":
    main()
