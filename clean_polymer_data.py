#!/usr/bin/env python3
"""
clean_polymer_data.py
=====================
Clean and validate polymer property data (Tg, Tm, density) from Properties.csv
BEFORE feature engineering.

What it does, in order
----------------------
1. SELECT      keep only the exact property_ids tg / tm / density, scalar values only
2. UNITS       Tg, Tm -> Kelvin ; density -> g/cm^3 ; cross-checks converted values
3. STRUCTURE   RDKit: parse, exactly two [*] endpoints, single fragment, standardize,
               canonicalize (so the same polymer written two ways becomes one entry)
4. RANGE       drop physically impossible values, flag unusual ones for review
5. DUPLICATES  one row per polymer: close values are averaged, moderate disagreement
               -> median (flagged), large disagreement -> dropped (or median, your choice)
6. OUTPUT      data_tg.csv, data_tm.csv, data_density.csv, data_all.csv (+ reports/)

Optional: --kaggle train.csv merges the Kaggle Open Polymer Prediction file (SIMULATED
values, Tg in degC). Its SMILES are already p-SMILES (the * marks the endpoints), so no
conversion is needed; they go through the same validation as everything else. Every output
row carries an `origin` column (properties_csv / kaggle_simulated / mixed).

Nothing is silently deleted: every removed row goes to reports/rejected_rows.csv
with a reason, and every judgment call goes to a report you can inspect.

Install:  pip install rdkit pandas numpy matplotlib
Run:      python clean_polymer_data.py --properties Properties.csv --outdir cleaned
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
except ImportError:
    sys.exit("RDKit is required.  Install it with:  pip install rdkit   (or conda install -c conda-forge rdkit)")

RDLogger.DisableLog("rdApp.*")

# --------------------------------------------------------------------------- #
# Settings per property. Edit these if your guide / domain knowledge says so.
#   hard  : outside this range -> impossible, row is REMOVED
#   soft  : outside this range -> unusual, polymer is KEPT but listed for review
#   close : values within this spread are treated as agreeing -> mean
#   drop  : spread above this = real conflict -> dropped (see --large-conflict)
# --------------------------------------------------------------------------- #
PROPS = {
    "tg":      dict(col="Tg_K",          hard=(100, 1000), soft=(150, 800), close=10.0, drop=50.0),
    "tm":      dict(col="Tm_K",          hard=(150, 1100), soft=(200, 800), close=10.0, drop=50.0),
    "density": dict(col="density_g_cm3", hard=(0.5, 3.5),  soft=(0.7, 2.6), close=0.02, drop=0.10),
}

NEEDED_COLS = ["record_id", "source_collection", "property_id", "record_kind", "value_kind",
               "numeric_value", "standard_value", "unit_raw", "psmiles", "reference"]

LOG_LINES = []


def log(msg=""):
    print(msg)
    LOG_LINES.append(str(msg))


# --------------------------------------------------------------------------- #
# Optional Kaggle file -> same long format as Properties.csv
# --------------------------------------------------------------------------- #
KAGGLE = "kaggle_opp2025"


def load_kaggle(path):
    k = pd.read_csv(path)
    if "SMILES" not in k.columns:
        sys.exit("--kaggle file needs a SMILES column (expected the Kaggle train.csv)")
    if "id" not in k.columns:
        k["id"] = range(len(k))
    parts = []
    base = dict(source_collection=KAGGLE, record_kind="property", value_kind="scalar", reference="")
    if "Tg" in k.columns:  # Kaggle Tg is in degC
        t = k[["id", "SMILES", "Tg"]].dropna(subset=["Tg"])
        parts.append(pd.DataFrame({**base, "record_id": "kaggle_tg_" + t.id.astype(str), "property_id": "tg",
                                   "numeric_value": t.Tg.values, "standard_value": (t.Tg + 273.15).values,
                                   "unit_raw": "°C", "psmiles": t.SMILES.values}))
    if "Density" in k.columns:  # g/cm3
        d = k[["id", "SMILES", "Density"]].dropna(subset=["Density"])
        parts.append(pd.DataFrame({**base, "record_id": "kaggle_density_" + d.id.astype(str), "property_id": "density",
                                   "numeric_value": d.Density.values, "standard_value": np.nan,
                                   "unit_raw": "g/cm3", "psmiles": d.SMILES.values}))
    return pd.concat(parts, ignore_index=True)


def origin_of(sources):
    sim = [p.startswith("kaggle") for p in sources.split(";")]
    return "kaggle_simulated" if all(sim) else ("mixed" if any(sim) else "properties_csv")


# --------------------------------------------------------------------------- #
# Step 3 helpers: structure validation + standardization (cached)
# --------------------------------------------------------------------------- #
_STRUCT_CACHE = {}


def standardize_psmiles(raw, keep_stereo=False):
    """Return (canonical_p_smiles_with_[*], None) or (None, reason)."""
    key = (raw, keep_stereo)
    if key in _STRUCT_CACHE:
        return _STRUCT_CACHE[key]
    res = _standardize(raw, keep_stereo)
    _STRUCT_CACHE[key] = res
    return res


def _standardize(raw, keep_stereo):
    if not isinstance(raw, str) or not raw.strip():
        return None, "missing_psmiles"
    mol = Chem.MolFromSmiles(raw.strip())
    if mol is None:
        return None, "rdkit_parse_failed"
    if len(Chem.GetMolFrags(mol)) > 1:
        return None, "multiple_fragments"
    dummies = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) != 2:
        return None, f"endpoints_{len(dummies)}_not_2"
    if any(a.GetDegree() != 1 for a in dummies):
        return None, "endpoint_not_terminal"
    if mol.GetNumAtoms() - 2 < 1:
        return None, "no_real_atoms"
    try:
        mol = rdMolStandardize.Cleanup(mol)  # normalize functional groups, reionize
    except Exception:
        return None, "standardization_failed"
    for a in mol.GetAtoms():  # make [1*], [*:1] etc. identical to [*]
        if a.GetAtomicNum() == 0:
            a.SetIsotope(0)
            a.SetAtomMapNum(0)
    out = Chem.MolToSmiles(mol, isomericSmiles=keep_stereo)
    out = re.sub(r"(?<!\[)\*(?!\])", "[*]", out)  # write endpoints uniformly as [*]
    chk = Chem.MolFromSmiles(out)  # round-trip safety check
    if chk is None or sum(1 for a in chk.GetAtoms() if a.GetAtomicNum() == 0) != 2:
        return None, "roundtrip_failed"
    return out, None


# --------------------------------------------------------------------------- #
# Step 2 helpers: units
# --------------------------------------------------------------------------- #
C_UNITS = {"°c", "degc", "℃", "c", "celsius", "°celsius"}
K_UNITS = {"k", "kelvin"}


def _unit_key(u):
    return str(u).strip().lower().replace(" ", "")


def temp_to_kelvin(value, unit):
    """Convert only when the unit is unambiguous, else NaN."""
    if pd.isna(value) or pd.isna(unit):
        return np.nan
    u = _unit_key(unit)
    if u in K_UNITS:
        return float(value)
    if u in C_UNITS:
        return float(value) + 273.15
    return np.nan


def density_factor(unit):
    """Multiplier to convert to g/cm^3. NaN = unit not understood."""
    if pd.isna(unit):
        return 1.0, True  # legacy rows have no unit; assumed g/cm^3 (flagged)
    raw = str(unit).strip()
    if re.search(r"^Mg\s*/?\s*m", raw):  # Mg/m^3 (capital M = megagram) == g/cm^3
        return 1.0, False
    u = raw.lower().replace("³", "3").replace("^", "").replace(" ", "").replace("−", "-")
    if re.search(r"^kg/?m", u):
        return 0.001, False
    if re.search(r"^g/?cm|^g/?cc", u):
        return 1.0, False
    return np.nan, False


def source_tag(collection, reference):
    """Readable source label. 'legacy' rows are split by the paper they cite."""
    if collection != "legacy":
        return collection
    cit = ""
    if isinstance(reference, str):
        try:
            cit = json.loads(reference).get("citation", "") or ""
        except Exception:
            cit = reference
    for key in ("PolyMetriX", "PolyGraphMT", "PolyNC"):
        if key in cit:
            return f"legacy:{key}"
    return "legacy:other"


# --------------------------------------------------------------------------- #
# Per-property pipeline
# --------------------------------------------------------------------------- #
def process_property(df, pid, cfg, args, rejected, suspects, conflicts):
    col = cfg["col"]
    funnel = {}
    d = df[df.property_id == pid].copy()
    funnel["1_rows_with_property_id"] = len(d)

    def reject(sub, reason):
        if len(sub):
            r = pd.DataFrame({
                "record_id": sub.record_id, "property_id": pid, "psmiles_raw": sub.psmiles,
                "value": sub.get("value", np.nan), "source": sub.get("source", np.nan), "reason": reason})
            rejected.append(r)

    # --- scalar only -------------------------------------------------------- #
    bad = d[d.value_kind != "scalar"]
    reject(bad.assign(reason=None), "not_scalar_(range/text/approximate)")
    d = d[d.value_kind == "scalar"].copy()
    funnel["2_scalar_values"] = len(d)

    d["source"] = [source_tag(c, r) for c, r in zip(d.source_collection, d.reference)]
    if args.exclude_source:
        ex = d.source.apply(lambda s: any(x.lower() in s.lower() for x in args.exclude_source))
        reject(d[ex], "excluded_by_--exclude-source")
        d = d[~ex].copy()
    funnel["3_after_source_exclusion"] = len(d)

    # --- units -------------------------------------------------------------- #
    if pid in ("tg", "tm"):
        # our own conversion of the raw number, used as a cross-check and as a fallback
        conv = pd.Series([temp_to_kelvin(v, u) for v, u in zip(d.numeric_value, d.unit_raw)], index=d.index)
        d["value"] = d.standard_value.fillna(conv)  # preferred: the file's own Kelvin value
        mism = d.standard_value.notna() & conv.notna() & ((d.standard_value - conv).abs() > 1.0)
        reject(d[mism], "unit_conversion_mismatch")
        d = d[~mism].copy()
        n_cel = d.unit_raw.map(lambda u: _unit_key(u) in C_UNITS if pd.notna(u) else False).sum()
        log(f"[{pid}] Celsius rows converted to Kelvin: {n_cel}")
    else:
        fac = d.unit_raw.map(density_factor)
        d["factor"] = [f[0] for f in fac]
        d["assumed_unit"] = [f[1] for f in fac]
        reject(d[d.factor.isna()], "unit_unresolved")
        d = d[d.factor.notna()].copy()
        d["value"] = d.numeric_value * d.factor
        log(f"[{pid}] rows with no unit (assumed g/cm3): {int(d.assumed_unit.sum())};  "
            f"rows converted from kg/m3: {int((d.factor == 0.001).sum())}")
    reject(d[d.value.isna()], "value_missing")
    d = d[d.value.notna()].copy()
    funnel["4_valid_value_and_unit"] = len(d)

    # --- impossible values -------------------------------------------------- #
    lo, hi = cfg["hard"]
    imp = (d.value < lo) | (d.value > hi)
    reject(d[imp], f"impossible_value_outside_{lo}-{hi}")
    d = d[~imp].copy()
    funnel["5_physically_possible"] = len(d)

    # --- structure ---------------------------------------------------------- #
    uniq = d.psmiles.dropna().unique()
    mapping = {s: standardize_psmiles(s, args.keep_stereo) for s in uniq}
    canon, why = [], []
    for s in d.psmiles:
        c, w = mapping.get(s, (None, "missing_psmiles"))
        canon.append(c)
        why.append(w)
    d["canon"], d["why"] = canon, why
    badstruct = d[d.canon.isna()]
    for reason, sub in badstruct.groupby("why"):
        reject(sub, f"structure_{reason}")
    d = d[d.canon.notna()].copy()
    funnel["6_valid_structure"] = len(d)

    # --- Kaggle rows: by default never override a polymer that already has a value -- #
    if args.kaggle and args.kaggle_mode == "fill-gaps":
        is_k = d.source_collection == KAGGLE
        clash = is_k & d.canon.isin(set(d.loc[~is_k, "canon"]))
        reject(d[clash], "kaggle_skipped_value_already_in_properties_csv")
        d = d[~clash].copy()
        log(f"[{pid}] Kaggle rows kept: {int((d.source_collection == KAGGLE).sum())} | "
            f"skipped (polymer already in Properties.csv): {int(clash.sum())}")

    # --- exact duplicate records (same polymer, value, source) -------------- #
    before = len(d)
    d = d.drop_duplicates(subset=["canon", "source", "value"]).copy()
    funnel["7_after_exact_duplicate_removal"] = len(d)
    log(f"[{pid}] exact duplicate records removed: {before - len(d)}")

    # --- aggregate per polymer --------------------------------------------- #
    g = d.groupby("canon")
    agg = g.value.agg(n_measurements="count", vmin="min", vmax="max", mean="mean",
                      median="median", std=lambda x: x.std(ddof=0)).reset_index()
    agg["spread"] = agg.vmax - agg.vmin
    agg["sources"] = g.source.agg(lambda s: ";".join(sorted(set(s)))).values
    agg["origin"] = agg.sources.map(origin_of)
    funnel["8_unique_polymers"] = len(agg)

    def decide(r):
        if r.n_measurements == 1:
            return r["mean"], "single"
        if r.spread <= cfg["close"]:
            return r["mean"], "averaged_close"
        if r.spread <= cfg["drop"]:
            return r["median"], "median_moderate_conflict"
        if args.large_conflict == "median":
            return r["median"], "median_large_conflict"
        return np.nan, "dropped_large_conflict"

    res = agg.apply(decide, axis=1, result_type="expand")
    agg["value"], agg["status"] = res[0], res[1]
    conf = agg[agg.status.str.contains("conflict")].copy()
    if len(conf):
        raw_vals = d.groupby("canon").value.apply(lambda s: ", ".join(f"{x:.4g}" for x in sorted(s)))
        conf["all_values"] = conf.canon.map(raw_vals)
        conf.insert(0, "property_id", pid)
        conflicts.append(conf)
    log(f"[{pid}] polymers: {len(agg)} | multi-measurement: {(agg.n_measurements > 1).sum()} | "
        f"moderate conflict (median): {(agg.status == 'median_moderate_conflict').sum()} | "
        f"large conflict: {agg.status.str.contains('large').sum()}")

    final = agg[agg.value.notna()].copy()
    final["value"] = final.value.round(4)  # remove floating-point noise (e.g. 373.1499999999)
    funnel["9_final_polymers"] = len(final)

    # --- unusual values (kept, but listed for manual review) ---------------- #
    slo, shi = cfg["soft"]
    med = final.value.median()
    mad = (final.value - med).abs().median() * 1.4826
    z = (final.value - med) / mad if mad > 0 else pd.Series(0.0, index=final.index)
    flag_soft = (final.value < slo) | (final.value > shi)
    flag_z = z.abs() > 4
    sus = final[flag_soft | flag_z].copy()
    if len(sus):
        sus["reason"] = np.where(flag_soft[sus.index], f"outside_typical_range_{slo}-{shi}", "robust_z>4")
        sus.insert(0, "property_id", pid)
        suspects.append(sus[["property_id", "canon", "value", "n_measurements", "sources", "reason"]])
    log(f"[{pid}] unusual values flagged for review: {len(sus)}")

    final = final.rename(columns={"canon": "p_smiles", "value": col})
    return final, funnel


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--properties", required=True, help="path to Properties.csv")
    ap.add_argument("--outdir", default="cleaned")
    ap.add_argument("--keep-stereo", action="store_true",
                    help="keep stereochemistry in canonical SMILES (default: remove, so stereo variants merge)")
    ap.add_argument("--large-conflict", choices=["drop", "median"], default="drop",
                    help="what to do when measurements of one polymer disagree by more than the 'drop' threshold")
    ap.add_argument("--exclude-source", nargs="*", default=[],
                    help="skip rows whose source label contains any of these words, e.g. legacy:PolyGraphMT")
    ap.add_argument("--kaggle", default=None,
                    help="optional: Kaggle Open Polymer Prediction train.csv (simulated data, Tg in degC)")
    ap.add_argument("--kaggle-mode", choices=["fill-gaps", "all"], default="fill-gaps",
                    help="fill-gaps (default): use Kaggle values only for polymers with no Properties.csv value. "
                         "all: use every Kaggle value (conflicts are then resolved by the duplicate rules)")
    args = ap.parse_args()

    out = Path(args.outdir)
    (out / "reports").mkdir(parents=True, exist_ok=True)

    header = pd.read_csv(args.properties, nrows=0).columns
    missing = [c for c in NEEDED_COLS if c not in header]
    if missing:
        sys.exit(f"Properties.csv is missing expected columns: {missing}")
    df = pd.read_csv(args.properties, usecols=NEEDED_COLS, low_memory=False)
    for c in ("numeric_value", "standard_value"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    log(f"Loaded {len(df):,} rows from {args.properties}")
    df = df[(df.record_kind == "property") & df.property_id.isin(PROPS)].copy()
    log(f"Rows with property_id in {list(PROPS)}: {len(df):,}")
    if args.kaggle:
        kg = load_kaggle(args.kaggle)
        df = pd.concat([df, kg[df.columns]], ignore_index=True)
        log(f"Added {len(kg):,} Kaggle rows from {args.kaggle} (mode: {args.kaggle_mode})")
    log("")

    rejected, suspects, conflicts, finals, funnels = [], [], [], {}, {}
    for pid, cfg in PROPS.items():
        finals[pid], funnels[pid] = process_property(df, pid, cfg, args, rejected, suspects, conflicts)
        log("")

    # --- cross-property sanity: Tm should be above Tg ----------------------- #
    both = finals["tg"].merge(finals["tm"], on="p_smiles")
    odd = both[(both.Tm_K <= both.Tg_K) | (both.Tm_K / both.Tg_K > 2.5)]
    if len(odd):
        suspects.append(pd.DataFrame({
            "property_id": "tg_vs_tm", "canon": odd.p_smiles, "value": odd.Tm_K / odd.Tg_K,
            "n_measurements": np.nan, "sources": "", "reason": "Tm<=Tg or Tm/Tg>2.5 (check both values)"}))
    log(f"Tg/Tm consistency: {len(odd)} of {len(both)} polymers with both values look suspicious")

    # --- write the four requested datasets ---------------------------------- #
    def src(f):
        return f.sources

    tg = finals["tg"][["p_smiles", "Tg_K"]].assign(source=src(finals["tg"]), origin=finals["tg"].origin)
    tm = finals["tm"][["p_smiles", "Tm_K"]].assign(source=src(finals["tm"]), origin=finals["tm"].origin)
    de = finals["density"][["p_smiles", "density_g_cm3"]].assign(source=src(finals["density"]),
                                                                  origin=finals["density"].origin)
    tg.to_csv(out / "data_tg.csv", index=False)
    tm.to_csv(out / "data_tm.csv", index=False)
    de.to_csv(out / "data_density.csv", index=False)

    def part(pid):
        f = finals[pid]
        return f[["p_smiles", PROPS[pid]["col"], "origin"]].rename(columns={"origin": "origin_" + pid})

    allp = (part("tg").merge(part("tm"), on="p_smiles", how="outer")
            .merge(part("density"), on="p_smiles", how="outer"))
    allp.to_csv(out / "data_all.csv", index=False)  # blanks where a property is missing
    complete = allp.dropna()
    complete.to_csv(out / "data_all_complete.csv", index=False)  # only polymers with all three

    for pid, f in finals.items():
        f.drop(columns=["vmin", "vmax", "mean", "median"]).to_csv(out / "reports" / f"details_{pid}.csv", index=False)
    pd.concat(rejected, ignore_index=True).to_csv(out / "reports" / "rejected_rows.csv", index=False) if rejected else None
    if suspects:
        pd.concat(suspects, ignore_index=True).to_csv(out / "reports" / "suspect_values.csv", index=False)
    if conflicts:
        pd.concat(conflicts, ignore_index=True).to_csv(out / "reports" / "conflicting_measurements.csv", index=False)

    # --- distributions ------------------------------------------------------ #
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for a, (pid, cfg) in zip(ax, PROPS.items()):
            a.hist(finals[pid][cfg["col"]], bins=60)
            a.set_title(f"{cfg['col']}  (n={len(finals[pid])})")
        fig.tight_layout()
        fig.savefig(out / "reports" / "distributions.png", dpi=130)
    except Exception as e:  # plotting is optional
        log(f"(skipped distribution plot: {e})")

    # --- summary ------------------------------------------------------------ #
    log("\n=== CLEANING FUNNEL (rows / polymers left after each step) ===")
    ft = pd.DataFrame(funnels).fillna(0).astype(int)
    log(ft.to_string())
    log("\n=== OUTPUT ===")
    log(f"data_tg.csv:           {len(tg):>6} polymers")
    log(f"data_tm.csv:           {len(tm):>6} polymers")
    log(f"data_density.csv:      {len(de):>6} polymers")
    log(f"data_all.csv:          {len(allp):>6} polymers (blank where a property is missing)")
    log(f"data_all_complete.csv: {len(complete):>6} polymers (all three present)")
    (out / "reports" / "cleaning_summary.txt").write_text("\n".join(LOG_LINES), encoding="utf-8")
    log(f"\nReports written to {out / 'reports'}")


if __name__ == "__main__":
    main()