#!/usr/bin/env python3
"""
build_features.py
=================
Compute the raw feature matrices ONCE for every polymer in your cleaned data files.

    python build_features.py --cleaned-dir cleaned --outdir features
    python build_features.py --cleaned-dir cleaned --outdir features --workers 4   # faster

Reads  : cleaned/data_tg.csv, data_tm.csv, data_density.csv, data_all.csv  (whichever exist)
Writes : features/fp_raw.npy          Morgan count fingerprints   (n_polymers x 2048)
         features/desc_raw.npy        RDKit descriptors, uncleaned (n_polymers x ~210)
         features/master_index.csv    p_smiles, ok   (row order of the two .npy files)
         features/feature_config.json settings + descriptor names (kept for the API later)
         features/failed.csv          polymers that could not be featurized (if any)

No column removal or scaling happens here. Those steps are FITTED on training data only,
so they wait until the train/test split exists (see polymer_features.fit_columns).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import polymer_features as pf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cleaned-dir", default="cleaned")
    ap.add_argument("--outdir", default="features")
    ap.add_argument("--workers", type=int, default=1, help="CPU processes (try 4)")
    args = ap.parse_args()

    cdir, out = Path(args.cleaned_dir), Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    names = ["data_tg.csv", "data_tm.csv", "data_density.csv", "data_all.csv"]
    found = [cdir / n for n in names if (cdir / n).exists()]
    if not found:
        sys.exit(f"No cleaned files found in {cdir}/. Run clean_polymer_data.py first.")
    smiles = pd.concat([pd.read_csv(f, usecols=["p_smiles"]) for f in found]).p_smiles
    smiles = sorted(set(smiles.dropna()))
    print(f"Files used: {[f.name for f in found]}")
    print(f"Unique polymers to featurize: {len(smiles):,}  (workers={args.workers})")

    t0 = time.time()
    fp, de, ok = pf.featurize(smiles, workers=args.workers)
    print(f"Featurized in {time.time() - t0:.0f} s")

    np.save(out / "fp_raw.npy", fp)
    np.save(out / "desc_raw.npy", de)
    pd.DataFrame({"p_smiles": smiles, "ok": ok}).to_csv(out / "master_index.csv", index=False)
    (out / "feature_config.json").write_text(
        json.dumps({"config": pf.CONFIG, "descriptor_names": pf.DESC_NAMES}, indent=1), encoding="utf-8")
    if (~ok).any():
        pd.DataFrame({"p_smiles": np.array(smiles)[~ok]}).to_csv(out / "failed.csv", index=False)

    # ---- sanity report ------------------------------------------------------ #
    print("\n=== SANITY CHECKS ===")
    print(f"fingerprints : {fp.shape}   descriptors : {de.shape}")
    print(f"featurization failed for {int((~ok).sum())} polymers" + ("  (see failed.csv)" if (~ok).any() else ""))
    f = fp[ok]
    print(f"fingerprint slots ever used: {int((f.sum(0) > 0).sum())} / {f.shape[1]};  "
          f"mean non-zero slots per polymer: {(f > 0).sum(1).mean():.1f};  max count: {int(f.max())}")
    d = de[ok]
    nan_cols = int(np.isnan(d).any(0).sum())
    inf_cols = int(np.isinf(d).any(0).sum())
    const_cols = int((np.nanstd(np.where(np.isfinite(d), d, np.nan), axis=0) == 0).sum())
    print(f"descriptor columns with NaN: {nan_cols} | with inf: {inf_cols} | constant: {const_cols}  "
          f"(these are removed later, using training rows only)")
    print("\nDone. Next step: build the cluster-based split, then call polymer_features.fit_columns on TRAIN rows.")


if __name__ == "__main__":
    main()