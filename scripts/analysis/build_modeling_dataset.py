"""Build the kinetic-parameter modeling dataset from the v20260530 reviewed gold.

Inputs : dataset/mycoextract_primary_database_v20260530_reviewed.csv
Outputs: modeling/data/kinetic_modeling_dataset.csv  (one row per usable measurement)
         modeling/data/kinetic_modeling_dataset_summary.json

Standardization
---------------
SMILES   : RDKit canonical SMILES (isomeric). Rows with unparseable SMILES dropped.
Km       -> micromolar (uM)
kcat     -> per second (s^-1)
kcat/Km  -> per molar per second (M^-1 s^-1)
Targets  : log10 of each standardized value (kinetic params span orders of magnitude).

Only rows with a sequence AND a canonical SMILES AND at least one kinetic target are kept.
"""
from __future__ import annotations
import json
import math
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parents[2] / "analysis_outputs" / "reproduction"
GOLD = HERE / "dataset" / "mycoextract_primary_database_v20260530_reviewed.csv"
OUT_DIR = HERE / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def s(v):
    if v is None:
        return ""
    t = str(v).strip()
    return "" if t.lower() in {"nan", "none", "null", ""} else t


def to_float(v):
    try:
        return float(str(v).strip())
    except (ValueError, TypeError):
        return None


def canon_smiles(smi):
    smi = s(smi)
    if not smi:
        return None, None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None, None
    return Chem.MolToSmiles(m, isomericSmiles=True), Descriptors.MolWt(m)


# ---- unit converters -> canonical unit -------------------------------------
def km_to_uM(val, unit, mw):
    """Return Km in micromolar."""
    u = s(unit).lower().replace("μ", "u")
    if val is None:
        return None
    if u in {"um", "umol/l", "umol/kg", "µm"}:
        return val
    if u in {"mm", "mmol/l"}:
        return val * 1e3
    if u in {"nm"}:
        return val * 1e-3
    if u in {"m", "mol/l"}:
        return val * 1e6
    if u in {"ug/ml", "mg/l"}:  # mass conc -> molar needs MW (g/mol): uM = ug/mL / MW *1000
        if mw and mw > 0:
            return val / mw * 1e3
        return None
    return None


def kcat_to_s(val, unit):
    u = s(unit).lower().replace("⁻", "^-").replace("¹", "1")
    if val is None:
        return None
    if u in {"s^-1", "s-1", "1/s", "/s"}:
        return val
    if u in {"min^-1", "min-1", "1/min", "/min"}:
        return val / 60.0
    if u in {"h^-1", "h-1"}:
        return val / 3600.0
    return None


def kkm_to_M_s(val, unit):
    u = s(unit).lower().replace("⁻", "^-").replace("¹", "1").replace("μ", "u").replace(" ", "")
    if val is None:
        return None
    if u in {"m^-1s^-1", "m-1s-1", "1/(ms)"}:
        return val
    if u in {"mm^-1s^-1", "l/(smmol)", "l/(mmols)", "mm-1s-1"}:
        return val * 1e3
    if u in {"um^-1s^-1", "um-1s-1"}:
        return val * 1e6
    if u in {"m^-1min^-1", "m-1min-1"}:
        return val / 60.0
    return None


def main():
    df = pd.read_csv(GOLD, encoding="utf-8-sig", dtype=str)
    rows = []
    dropped = {"no_sequence": 0, "no_smiles": 0, "bad_smiles": 0, "no_target": 0, "unconvertible_units": 0}

    for _, r in df.iterrows():
        seq = s(r.get("sequence"))
        if not seq:
            dropped["no_sequence"] += 1
            continue
        smi_raw = s(r.get("substrate_smiles"))
        if not smi_raw:
            dropped["no_smiles"] += 1
            continue
        smi, mw = canon_smiles(smi_raw)
        if smi is None:
            dropped["bad_smiles"] += 1
            continue

        km = km_to_uM(to_float(r.get("Km_value")), r.get("Km_unit"), mw)
        kcat = kcat_to_s(to_float(r.get("kcat_value")), r.get("kcat_unit"))
        kkm = kkm_to_M_s(to_float(r.get("kcat_Km_value")), r.get("kcat_Km_unit"))

        if km is None and kcat is None and kkm is None:
            # had raw values but none convertible, or simply no kinetic value
            if any(s(r.get(c)) for c in ("Km_value", "kcat_value", "kcat_Km_value")):
                dropped["unconvertible_units"] += 1
            else:
                dropped["no_target"] += 1
            continue

        rows.append({
            "source_record_id": s(r.get("source_record_id")),
            "doi": s(r.get("doi")),
            "enzyme_name": s(r.get("normalized_enzyme_full_name")) or s(r.get("enzyme_name")),
            "organism": s(r.get("normalized_organism_full_name")) or s(r.get("organism")),
            "mutations": s(r.get("mutations")),
            "uniprot_id": s(r.get("uniprot_id")),
            "sequence": seq,
            "seq_len": len(seq),
            "substrate": s(r.get("normalized_substrate_full_name")) or s(r.get("substrate")),
            "substrate_smiles_canonical": smi,
            "substrate_mw": round(mw, 3) if mw else None,
            "Km_uM": km,
            "kcat_s": kcat,
            "kcat_Km_M_s": kkm,
            "log10_Km_uM": math.log10(km) if km and km > 0 else None,
            "log10_kcat_s": math.log10(kcat) if kcat and kcat > 0 else None,
            "log10_kcat_Km_M_s": math.log10(kkm) if kkm and kkm > 0 else None,
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "kinetic_modeling_dataset.csv", index=False, encoding="utf-8-sig")

    summary = {
        "n_measurements": len(out),
        "n_unique_sequences": int(out["sequence"].nunique()) if len(out) else 0,
        "n_unique_substrates": int(out["substrate_smiles_canonical"].nunique()) if len(out) else 0,
        "n_with_Km": int(out["Km_uM"].notna().sum()) if len(out) else 0,
        "n_with_kcat": int(out["kcat_s"].notna().sum()) if len(out) else 0,
        "n_with_kcat_Km": int(out["kcat_Km_M_s"].notna().sum()) if len(out) else 0,
        "dropped": dropped,
    }
    (OUT_DIR / "kinetic_modeling_dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
