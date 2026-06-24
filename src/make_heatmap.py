"""
make_heatmap.py
===============
The downstream demonstration: use the trained surrogate to paint a dense map of
predicted RoCoF over the (inertia, load) operating plane -- thousands of
predictions that are instant for the surrogate but would take ~hours of EMT.
A RoCoF-threshold contour is overlaid to show the protection-relevant boundary.

This is the figure that turns "we fit a regressor" into "we built a screening
tool that does something EMT can't do cheaply".

It:
  1. trains a GBT on the chosen disturbance's RoCoF (default: gentrip, the
     strongest model),
  2. predicts over a fine h_scale x load_level grid (event params held fixed),
  3. times the prediction to report a measured speedup vs EMT,
  4. renders a publication-quality heatmap with threshold contours and the
     training points overlaid (to show the map stays in-domain),
  5. saves PNG + PDF to results/figures/.

IMPORTANT: only map a target the model predicts well, and stay INSIDE the
sampled ranges. Extrapolating a surrogate makes pretty-but-wrong figures.

Usage
-----
    python -m src.make_heatmap                      # gentrip RoCoF (default)
    python -m src.make_heatmap --disturbance islanding
    python -m src.make_heatmap --t-event 1.5 --i-gentrip 0 --grid 160
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from xgboost import XGBRegressor
    HAVE_XGB = True
except Exception:  # noqa: BLE001
    from sklearn.ensemble import GradientBoostingRegressor
    HAVE_XGB = False

from .sampling import FEATURES

# base GENROU inertia (s) of the four machines -> mean used to label the axis in
# physical units. (dyd.gen_H = [6.5, 6.5, 6.175, 6.175])
BASE_H_MEAN = 6.34
EMT_SECONDS_PER_RUN = 11.7   # measured per-run cost from your real campaign

# domain (must match sampling.CONTINUOUS)
H_RANGE = (0.5, 1.5)
LOAD_RANGE = (0.8, 1.2)


def _model():
    if HAVE_XGB:
        return XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05,
                            subsample=0.9, colsample_bytree=0.9,
                            random_state=0, n_jobs=-1)
    return GradientBoostingRegressor(n_estimators=500, max_depth=3,
                                     learning_rate=0.05, random_state=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/dataset_full.csv")
    ap.add_argument("--disturbance", default="gentrip")
    ap.add_argument("--target", default="rocof_hz_s")
    ap.add_argument("--t-event", type=float, default=1.0)
    ap.add_argument("--i-gentrip", type=int, default=0)
    ap.add_argument("--grid", type=int, default=160, help="grid points per axis")
    ap.add_argument("--thresholds", default="1.0,2.0",
                    help="RoCoF contour levels (Hz/s), comma-separated")
    args = ap.parse_args()

    # --- train on the chosen disturbance subset ---------------------------
    df = pd.read_csv(args.data)
    sub = df[(df["disturbance"] == args.disturbance) & (df["converged"])].dropna(
        subset=[args.target])
    if len(sub) < 30:
        raise SystemExit(f"too few rows ({len(sub)}) for {args.disturbance}")
    X, y = sub[FEATURES], sub[args.target]
    model = _model().fit(X, y)
    print(f"[heatmap] trained on {len(sub)} {args.disturbance} rows")

    # --- dense grid over (h_scale, load_level) ----------------------------
    n = args.grid
    hs = np.linspace(*H_RANGE, n)
    ld = np.linspace(*LOAD_RANGE, n)
    HS, LD = np.meshgrid(hs, ld)
    grid = pd.DataFrame({
        "h_scale": HS.ravel(),
        "load_level": LD.ravel(),
        "t_event": args.t_event,
        "i_gentrip": args.i_gentrip,
    })[FEATURES]

    t0 = time.perf_counter()
    Z = model.predict(grid).reshape(n, n)
    predict_s = time.perf_counter() - t0

    npts = n * n
    emt_s = npts * EMT_SECONDS_PER_RUN
    speedup = emt_s / predict_s if predict_s > 0 else float("inf")
    print(f"[heatmap] {npts} predictions in {predict_s*1e3:.1f} ms")
    print(f"[heatmap] equivalent EMT time ~ {emt_s/3600:.1f} h  ->  "
          f"speedup ~ {speedup:,.0f}x")

    # --- render -----------------------------------------------------------
    Hbar = BASE_H_MEAN * hs          # physical mean inertia (s) for the x-axis
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    pcm = ax.pcolormesh(Hbar, ld, Z, shading="auto", cmap="viridis")
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(f"Predicted max RoCoF (Hz/s)")

    levels = [float(t) for t in args.thresholds.split(",") if t.strip()]
    if levels:
        cs = ax.contour(Hbar, ld, Z, levels=levels, colors="white",
                        linewidths=1.6, linestyles="--")
        ax.clabel(cs, fmt="%.1f Hz/s", fontsize=8, colors="white")

    # overlay the actual training points (shows the map stays in-domain)
    ax.scatter(BASE_H_MEAN * sub["h_scale"], sub["load_level"],
               s=10, c="k", alpha=0.35, linewidths=0, label="EMT training runs")

    ax.set_xlabel("Mean generator inertia  H  (s)")
    ax.set_ylabel("Load level  (pu of base)")
    ax.set_title(f"Surrogate-predicted RoCoF — {args.disturbance} "
                 f"(t_event={args.t_event}s, gen {args.i_gentrip})")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    cap = (f"{npts:,} surrogate predictions in {predict_s*1e3:.0f} ms; "
           f"the same sweep by EMT ≈ {emt_s/3600:.0f} h "
           f"(~{speedup:,.0f}× speedup).")
    fig.text(0.5, -0.02, cap, ha="center", fontsize=8, style="italic")

    out = Path("results/figures")
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"heatmap_{args.disturbance}_{args.target}"
    fig.savefig(f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    print(f"[heatmap] wrote {stem}.png and {stem}.pdf")


if __name__ == "__main__":
    main()
