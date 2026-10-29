"""
polymer_features.py  --  the ONE shared featurization module
==============================================================
Used by training now and by the prediction API later. Never rewrite this logic
somewhere else: if training and prediction featurize differently, results are wrong.

Two feature families, both built from the repeat unit (p-SMILES):

  Family A  Morgan COUNT fingerprints  (radius 2, 2048 slots)
            -> "how many times does each local chemical pattern appear?"
            Computed on the p-SMILES as written, so the [*] endpoints take part
            (the model can see what the chain connects through).

  Family B  RDKit 2D descriptors (~210 numbers: weight, rings, polarity, ...)
            Computed on the repeat unit with each [*] replaced by hydrogen,
            because dummy atoms break several descriptors (e.g. zero mass, charges).

Nothing here is *fitted* on data except the two `fit_*` functions, and those must be
called on TRAINING rows only (this is how leakage is avoided).
"""
import json
from pathlib import Path

import numpy as np
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

CONFIG = {
    "fp_radius": 2,
    "fp_size": 2048,
    "fp_type": "morgan_counts",
    "descriptor_endpoint_handling": "replace [*] with [H]",
    "rdkit_version": rdkit.__version__,
}
DESC_NAMES = [name for name, _ in Descriptors._descList]
_GEN = None


def _generator():
    global _GEN
    if _GEN is None:
        _GEN = rdFingerprintGenerator.GetMorganGenerator(radius=CONFIG["fp_radius"], fpSize=CONFIG["fp_size"])
    return _GEN


# --------------------------------------------------------------------------- #
# Stateless featurization (safe to run on ALL polymers, nothing is learned)
# --------------------------------------------------------------------------- #
def fingerprint(p_smiles):
    """Morgan count vector (float32, length fp_size) or None if unparseable."""
    mol = Chem.MolFromSmiles(p_smiles)
    if mol is None:
        return None
    return np.asarray(_generator().GetCountFingerprintAsNumPy(mol), dtype=np.float32)


def descriptors(p_smiles):
    """RDKit descriptor vector (float64) on the H-capped repeat unit, or None."""
    mol = Chem.MolFromSmiles(p_smiles.replace("[*]", "[H]"))
    if mol is None:
        return None
    out = np.full(len(DESC_NAMES), np.nan)
    for i, (_, fn) in enumerate(Descriptors._descList):
        try:
            out[i] = float(fn(mol))
        except Exception:
            pass  # stays NaN; the column cleaner deals with it
    return out


def _featurize_chunk(chunk):
    res = []
    for p in chunk:
        fp, de = fingerprint(p), descriptors(p)
        res.append((fp, de))
    return res


def featurize(p_smiles_list, workers=1):
    """Return fp (n, fp_size), desc (n, n_desc), ok (n,) bool. Rows with ok=False must be excluded."""
    p_smiles_list = list(p_smiles_list)
    n = len(p_smiles_list)
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        size = max(1, n // (workers * 8))
        chunks = [p_smiles_list[i:i + size] for i in range(0, n, size)]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            parts = [r for part in ex.map(_featurize_chunk, chunks) for r in part]
    else:
        parts = _featurize_chunk(p_smiles_list)
    fp = np.zeros((n, CONFIG["fp_size"]), dtype=np.float32)
    de = np.full((n, len(DESC_NAMES)), np.nan)
    ok = np.zeros(n, dtype=bool)
    for i, (f, d) in enumerate(parts):
        if f is not None and d is not None:
            fp[i], de[i], ok[i] = f, d, True
    return fp, de, ok


# --------------------------------------------------------------------------- #
# Fitted cleaning (call on TRAINING rows only, then reuse the spec everywhere)
# --------------------------------------------------------------------------- #
def fit_columns(fp_train, desc_train):
    """Decide which columns to keep using training rows only. Returns a JSON-able spec."""
    fp_keep = np.where(fp_train.std(axis=0) > 0)[0]  # drop never-varying (e.g. all-zero) slots
    finite = np.isfinite(desc_train).all(axis=0)  # drop columns with any NaN / inf
    with np.errstate(all="ignore"):
        varies = np.nanstd(np.where(np.isfinite(desc_train), desc_train, np.nan), axis=0) > 0
    d_keep = np.where(finite & varies)[0]
    return {
        "config": CONFIG,
        "fp_keep": fp_keep.tolist(),
        "desc_keep_names": [DESC_NAMES[i] for i in d_keep],
        "desc_train_median": np.median(desc_train[:, d_keep], axis=0).tolist(),
    }


def apply_columns(spec, fp, desc):
    """Apply a saved spec to ANY rows (train / validation / test / new polymer from the API)."""
    fp_out = fp[:, spec["fp_keep"]]
    idx = [DESC_NAMES.index(nm) for nm in spec["desc_keep_names"]]
    d_out = desc[:, idx].astype(np.float64)
    bad = ~np.isfinite(d_out)  # a value that is NaN/inf in new data -> training median
    if bad.any():
        d_out[bad] = np.broadcast_to(np.asarray(spec["desc_train_median"]), d_out.shape)[bad]
    return fp_out, d_out


def assemble(which, fp_clean, desc_clean):
    """Experiment inputs:  A = fingerprints only | B = descriptors only | C = both."""
    which = which.upper()
    if which == "A":
        return fp_clean
    if which == "B":
        return desc_clean
    if which == "C":
        return np.hstack([fp_clean, desc_clean.astype(np.float32)])
    raise ValueError("which must be 'A', 'B' or 'C'")


def save_spec(spec, path):
    Path(path).write_text(json.dumps(spec, indent=1), encoding="utf-8")


def load_spec(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Loading the precomputed master table (written by build_features.py)
# --------------------------------------------------------------------------- #
def load_master(feature_dir, p_smiles_list):
    """Fetch raw fp/desc rows for the given polymers (in that order). Returns fp, desc, ok."""
    import pandas as pd
    feature_dir = Path(feature_dir)
    idx = pd.read_csv(feature_dir / "master_index.csv")
    pos = {p: i for i, p in enumerate(idx.p_smiles)}
    rows = [pos[p] for p in p_smiles_list]  # KeyError = polymer was not featurized
    fp = np.load(feature_dir / "fp_raw.npy", mmap_mode="r")
    de = np.load(feature_dir / "desc_raw.npy", mmap_mode="r")
    ok = idx.ok.values[rows]
    return np.asarray(fp[rows]), np.asarray(de[rows]), ok