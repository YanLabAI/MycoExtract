"""Predict enzyme kinetic parameters from (enzyme sequence + substrate SMILES).

Design (data-scarcity / transfer-learning regime; see paper Methods):
  - Enzyme   : frozen ESM-2 (esm2_t12_35M_UR50D) mean-pooled embedding (480-d).
  - Substrate: RDKit Morgan fingerprint (1024-bit, r=2) + physicochemical descriptors.
  - Regressor: XGBoost / RandomForest / Ridge (compared).
  - Targets  : log10(kcat/Km [M^-1 s^-1]), log10(kcat [s^-1]), log10(Km [uM]).
  - CV       : enzyme-grouped K-fold (no enzyme shared across folds) = honest estimate,
               plus random K-fold for contrast (optimistic, leakage-prone).

Outputs (modeling/):
  features/esm2_embeddings.npy, features/feature_matrix.npz
  results/cv_metrics.csv, results/per_fold_predictions.csv, results/summary.json
  results/fig_pred_vs_true_<target>.png
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parents[2] / "analysis_outputs" / "reproduction"
DATA = HERE / "data" / "kinetic_modeling_dataset.csv"
FEAT = HERE / "features"
RES = HERE / "results"

TARGETS = {
    "log10_kcat_Km_M_s": "log10 kcat/Km (M^-1 s^-1)",
    "log10_kcat_s": "log10 kcat (s^-1)",
    "log10_Km_uM": "log10 Km (uM)",
}
ESM_MODEL = "facebook/esm2_t12_35M_UR50D"
N_SPLITS = 5
SEED = 42


# ---------- enzyme features: ESM-2 (fair-esm, pure PyTorch) -----------------
def compute_esm_embeddings(sequences: list[str]) -> np.ndarray:
    import torch
    import esm

    cache = FEAT / "esm2_embeddings.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        cached = {k: d[k] for k in d.files}
        if set(sequences) <= set(cached.keys()):
            return np.vstack([cached[s] for s in sequences])

    # esm2_t12_35M_UR50D: 12 layers, 480-d embedding (small, downloads once)
    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    bc = alphabet.get_batch_converter()
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    REPR_LAYER = 12

    uniq = sorted(set(sequences))
    emb = {}
    with torch.no_grad():
        for i, seq in enumerate(uniq, 1):
            s = seq[:1022]
            _, _, toks = bc([("p", s)])
            toks = toks.to(device)
            out = model(toks, repr_layers=[REPR_LAYER], return_contacts=False)
            rep = out["representations"][REPR_LAYER][0]  # (L+2, d)
            vec = rep[1:len(s) + 1].mean(0).cpu().numpy()  # drop BOS/EOS, mean-pool
            emb[seq] = vec.astype(np.float32)
            print(f"  ESM {i}/{len(uniq)}", flush=True)
    np.savez(cache, **emb)
    return np.vstack([emb[s] for s in sequences])


# ---------- substrate features: RDKit --------------------------------------
def compute_mol_features(smiles: list[str]) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    desc_fns = [Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA,
                Descriptors.NumHAcceptors, Descriptors.NumHDonors,
                Descriptors.NumRotatableBonds, Descriptors.RingCount,
                Descriptors.FractionCSP3, Descriptors.NumAromaticRings,
                Descriptors.HeavyAtomCount]
    rows = []
    for smi in smiles:
        m = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024)
        arr = np.zeros((1024,), dtype=np.float32)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        desc = np.array([fn(m) for fn in desc_fns], dtype=np.float32)
        rows.append(np.concatenate([arr, desc]))
    return np.vstack(rows)


def metrics(y_true, y_pred):
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from scipy.stats import spearmanr, pearsonr
    return {
        "n": int(len(y_true)),
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Pearson": float(pearsonr(y_true, y_pred)[0]),
        "Spearman": float(spearmanr(y_true, y_pred)[0]),
    }


def get_models():
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.dummy import DummyRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    return {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "ElasticNet": make_pipeline(StandardScaler(), ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)),
        "kNN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5, weights="distance")),
        "SVR (RBF)": make_pipeline(StandardScaler(), SVR(C=10.0, gamma="scale")),
        "RandomForest": RandomForestRegressor(n_estimators=400, max_depth=None,
                                              min_samples_leaf=2, random_state=SEED, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=300, max_depth=2,
                                                      learning_rate=0.05, subsample=0.8, random_state=SEED),
        "XGBoost": xgb.XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1),
    }


def run_cv(X, y, groups, scheme):
    from sklearn.model_selection import GroupKFold, KFold
    preds_all, fold_metrics = [], []
    for name, model in get_models().items():
        oof = np.full(len(y), np.nan)
        if scheme == "grouped":
            splitter = GroupKFold(n_splits=min(N_SPLITS, len(np.unique(groups))))
            split = splitter.split(X, y, groups)
        else:
            splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
            split = splitter.split(X, y)
        for fold, (tr, te) in enumerate(split):
            model.fit(X[tr], y[tr])
            p = model.predict(X[te])
            oof[te] = p
            for i, idx in enumerate(te):
                preds_all.append({"model": name, "scheme": scheme, "fold": fold,
                                  "row": int(idx), "y_true": float(y[idx]), "y_pred": float(p[i])})
        m = metrics(y, oof)
        m.update({"model": name, "scheme": scheme})
        fold_metrics.append(m)
    return fold_metrics, preds_all


def plot_pred(df_pred, target, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    best = df_pred[(df_pred.scheme == "grouped") & (df_pred.model != "Mean baseline")]
    if best.empty:
        return
    def safe_corr(g):
        if g.y_pred.std() == 0:
            return -1.0
        return np.corrcoef(g.y_true, g.y_pred)[0, 1]
    model = best.groupby("model").apply(safe_corr).idxmax()
    g = best[best.model == model]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(g.y_true, g.y_pred, s=22, alpha=0.7, edgecolor="k", linewidth=0.3)
    lo = min(g.y_true.min(), g.y_pred.min()); hi = max(g.y_true.max(), g.y_pred.max())
    ax.plot([lo, hi], [lo, hi], "--", color="grey", linewidth=1)
    ax.set_xlabel(f"Measured {label}"); ax.set_ylabel(f"Predicted {label}")
    ax.set_title(f"{model} (enzyme-grouped CV)")
    fig.tight_layout()
    fig.savefig(RES / f"fig_pred_vs_true_{target}.png", dpi=300)
    plt.close(fig)


def main():
    FEAT.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    print(f"Loaded {len(df)} measurements; computing features...", flush=True)
    Xe = compute_esm_embeddings(df["sequence"].tolist())
    Xm = compute_mol_features(df["substrate_smiles_canonical"].tolist())
    X = np.hstack([Xe, Xm]).astype(np.float32)
    np.savez_compressed(FEAT / "feature_matrix.npz", X=X, esm_dim=Xe.shape[1], mol_dim=Xm.shape[1])
    print(f"Feature matrix: {X.shape} (ESM {Xe.shape[1]} + mol {Xm.shape[1]})", flush=True)

    groups = df["sequence"].astype("category").cat.codes.to_numpy()
    all_metrics, all_preds, summary = [], [], {}
    for target, label in TARGETS.items():
        mask = df[target].notna().to_numpy()
        y = df.loc[mask, target].to_numpy(dtype=float)
        Xt, gt = X[mask], groups[mask]
        print(f"\n=== {target}: n={len(y)}, unique enzymes={len(np.unique(gt))} ===", flush=True)
        rows, preds = [], []
        for scheme in ("grouped", "random"):
            fm, pa = run_cv(Xt, y, gt, scheme)
            rows += fm
            preds += pa
        for r in rows:
            r["target"] = target
            print(f"  [{r['scheme']:7}] {r['model']:12} R2={r['R2']:.3f} RMSE={r['RMSE']:.3f} Spearman={r['Spearman']:.3f}", flush=True)
        dfp = pd.DataFrame(preds); dfp["target"] = target
        for p in preds:
            p["target"] = target
        all_metrics += rows; all_preds += preds
        plot_pred(dfp, target, label)
        grouped = [r for r in rows if r["scheme"] == "grouped" and np.isfinite(r["Spearman"])
                   and r["model"] != "Mean baseline"]
        best = max(grouped, key=lambda r: r["Spearman"])
        summary[target] = {"n": int(len(y)), "n_enzymes": int(len(np.unique(gt))),
                            "best_model_grouped": best["model"],
                            "grouped_R2": best["R2"], "grouped_Spearman": best["Spearman"],
                            "grouped_RMSE": best["RMSE"]}

    pd.DataFrame(all_metrics).to_csv(RES / "cv_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_preds).to_csv(RES / "per_fold_predictions.csv", index=False, encoding="utf-8-sig")

    # ---- explicit enzyme-grouped held-out test split (train vs test) ----
    from sklearn.model_selection import GroupShuffleSplit
    ho_preds, ho_metrics = [], []
    for target in TARGETS:
        mask = df[target].notna().to_numpy()
        y = df.loc[mask, target].to_numpy(dtype=float)
        Xt, gt = X[mask], groups[mask]
        name = summary[target]["best_model_grouped"]
        model = get_models()[name]
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED).split(Xt, y, gt))
        model.fit(Xt[tr], y[tr])
        ptr, pte = model.predict(Xt[tr]), model.predict(Xt[te])
        for idx, yt, yp in zip(tr, y[tr], ptr):
            ho_preds.append({"target": target, "model": name, "split": "train",
                             "y_true": float(yt), "y_pred": float(yp)})
        for idx, yt, yp in zip(te, y[te], pte):
            ho_preds.append({"target": target, "model": name, "split": "test",
                             "y_true": float(yt), "y_pred": float(yp)})
        mtr, mte = metrics(y[tr], ptr), metrics(y[te], pte)
        mtr.update({"target": target, "model": name, "split": "train",
                    "n_enzymes_train": int(len(np.unique(gt[tr])))})
        mte.update({"target": target, "model": name, "split": "test",
                    "n_enzymes_test": int(len(np.unique(gt[te])))})
        ho_metrics += [mtr, mte]
        summary[target].update({"heldout_test_R2": mte["R2"], "heldout_test_RMSE": mte["RMSE"],
                                "heldout_test_MAE": mte["MAE"], "heldout_test_Spearman": mte["Spearman"],
                                "heldout_train_R2": mtr["R2"], "heldout_n_test": int(len(te)),
                                "heldout_n_train": int(len(tr))})
        print(f"  [held-out] {target}: train R2={mtr['R2']:.3f} (n={len(tr)}) | "
              f"test R2={mte['R2']:.3f} RMSE={mte['RMSE']:.3f} Spearman={mte['Spearman']:.3f} (n={len(te)})", flush=True)
    pd.DataFrame(ho_preds).to_csv(RES / "heldout_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ho_metrics).to_csv(RES / "heldout_metrics.csv", index=False, encoding="utf-8-sig")

    (RES / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSaved results to", RES, flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
