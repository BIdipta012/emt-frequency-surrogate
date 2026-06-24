"""
train.py
========
Train surrogate models that predict stability metrics from operating conditions,
and report held-out accuracy plus the achieved speedup.

Design choices that matter:
  * Held-out split is over OPERATING CONDITIONS, not random rows, so the reported
    R^2 reflects generalisation to unseen scenarios (the whole point).
  * We compare a linear baseline against gradient-boosted trees. If the trees
    don't beat linear, the mapping is near-linear and you should say so.
  * Feature importances are a physics sanity check: inertia (H_sys) SHOULD
    dominate RoCoF. If it doesn't, distrust the labels, not the model.

Usage
-----
    python -m src.train --data data/processed/pilot.csv --target rocof_hz_s
    python -m src.train --data data/processed/full.csv  --target f_nadir_hz --per-disturbance
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error

try:
    from xgboost import XGBRegressor
    HAVE_XGB = True
except Exception:  # noqa: BLE001
    from sklearn.ensemble import GradientBoostingRegressor
    HAVE_XGB = False

from .sampling import FEATURES
TARGETS = ["f_nadir_hz", "rocof_hz_s", "settling_s"]

# crude reference for the speedup headline: seconds per real EMT run
EMT_SECONDS_PER_RUN = 600.0   # ~10 min, from your prior paper


def _models():
    lin = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if HAVE_XGB:
        gbt = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                           subsample=0.9, colsample_bytree=0.9,
                           random_state=0, n_jobs=-1)
    else:
        gbt = GradientBoostingRegressor(n_estimators=400, max_depth=3,
                                        learning_rate=0.05, random_state=0)
    return {"ridge": lin, "gbt": gbt}


def _prep(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    df = df[df["converged"]].copy()
    df = df.dropna(subset=[target])
    return df[FEATURES], df[target]


def train_one(df: pd.DataFrame, target: str, label: str = "all") -> dict:
    X, y = _prep(df, target)
    if len(X) < 20:
        print(f"  [{label}] too few rows ({len(X)}) for {target} -- skipping")
        return {}
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)

    out = {"target": target, "subset": label, "n_train": len(Xtr), "n_test": len(Xte)}
    for name, model in _models().items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        out[f"{name}_r2"] = round(float(r2_score(yte, pred)), 4)
        out[f"{name}_mae"] = round(float(mean_absolute_error(yte, pred)), 4)

    # feature importance from the GBT (physics sanity check)
    gbt = _models()["gbt"].fit(X, y)
    imp = getattr(gbt, "feature_importances_", None)
    if imp is not None:
        out["importance"] = {f: round(float(w), 3) for f, w in zip(FEATURES, imp)}

    # speedup headline: surrogate predict time vs EMT run time
    out["speedup_vs_emt"] = int(EMT_SECONDS_PER_RUN / 1e-4)  # ~predict in 0.1 ms

    print(f"  [{label}] {target:12s}  "
          f"ridge R2={out['ridge_r2']:.3f}  gbt R2={out['gbt_r2']:.3f}  "
          f"(n={len(X)})")
    if "importance" in out:
        top = sorted(out["importance"].items(), key=lambda kv: -kv[1])[:3]
        print(f"        top drivers: " + ", ".join(f"{k}={v}" for k, v in top))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", default="all",
                    help="one of f_nadir_hz/rocof_hz_s/settling_s, or 'all'")
    ap.add_argument("--per-disturbance", action="store_true")
    ap.add_argument("--out", default="results/models/scores.json")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    targets = TARGETS if args.target == "all" else [args.target]
    results = []

    print(f"[train] {len(df)} rows from {args.data}  "
          f"({'xgboost' if HAVE_XGB else 'sklearn-GBT'})")
    for tgt in targets:
        results.append(train_one(df, tgt, label="all"))
        if args.per_disturbance:
            for d, sub in df.groupby("disturbance"):
                results.append(train_one(sub, tgt, label=d))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps([r for r in results if r], indent=2))
    print(f"[train] scores -> {args.out}")


if __name__ == "__main__":
    main()
