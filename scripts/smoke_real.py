"""
smoke_real.py
=============
Run ONE real ParaEMT case and check the trajectory is physically sane, BEFORE
committing to a batch. This is the validation gate for the whole Option-2 path.

It runs with USE_MOCK forced off, on a single case you pick, and reports:
  * did the run complete and stay finite
  * steady-state COI frequency (should sit ~60 Hz)
  * nadir / RoCoF / settling in real units
and writes the COI trace to results/figures/smoke_<disturbance>.csv so you can
plot it.

Usage
-----
    python -m scripts.smoke_real --disturbance baseline
    python -m scripts.smoke_real --disturbance gentrip --t-event 1.0
    python -m scripts.smoke_real --disturbance islanding --t-event 1.5

WHAT SUCCESS LOOKS LIKE
-----------------------
  baseline : COI frequency flat, ~60.000 Hz, nadir ~0, RoCoF ~0
  gentrip  : clear dip after t_event, nadir a few tenths of a Hz, RoCoF in a
             physically sane range (order 0.1-5 Hz/s), settling a few seconds
If baseline is NOT flat, the disturbance-flag logic in
paraemt_run._configure_disturbance needs adjustment (see its [VERIFY] note).
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np

import src.paraemt_driver as drv
from src.paraemt_run import init_case, run_emt_loop, extract_trajectory
from src.metrics import coi_frequency, compute_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disturbance", default="baseline",
                    choices=["baseline", "strong_grid", "heavy_load",
                             "islanding", "gentrip", "stepchange"])
    ap.add_argument("--t-event", type=float, default=1.0)
    ap.add_argument("--load-level", type=float, default=1.0)
    ap.add_argument("--h-scale", type=float, default=1.0,
                    help="inertia multiplier on dyd.gen_H (1.0=no change). "
                         "Lower => less inertia => deeper nadir / larger RoCoF.")
    ap.add_argument("--i-gentrip", type=int, default=0)
    args = ap.parse_args()

    params = {
        "disturbance": args.disturbance,
        "t_event": args.t_event,
        "load_level": args.load_level,
        "h_scale": args.h_scale,
        "i_gentrip": args.i_gentrip,
    }

    print(f"[smoke] running ONE real case: {params}")
    print(f"[smoke] ParaEMT dir: {drv.PARAEMT_DIR}")

    pfd, ini, dyd, emt = init_case(params, paraemt_dir=drv.PARAEMT_DIR,
                                   systemN=6, ts=drv.RAW_TS, Tlen=drv.T_END)
    print(f"[smoke] init OK: {len(pfd.gen_bus)} machines, "
          f"H={np.asarray(dyd.gen_H)}, ws={pfd.ws:.3f} rad/s")

    emt = run_emt_loop(pfd, dyd, ini, emt, ts=drv.RAW_TS,
                       Tlen=drv.T_END, DSrate=drv.DS_RATE)
    traj = extract_trajectory(emt, pfd, dyd)

    freq = coi_frequency(traj["speed_pu"], traj["H"], f0=drv.F0)
    m = compute_metrics(freq, dt=traj["dt"], f0=drv.F0, converged=traj["converged"])

    print("\n[smoke] ---- sanity report ----")
    print(f"  samples            : {len(freq)}")
    print(f"  converged          : {traj['converged']}")
    print(f"  steady-state freq  : {np.mean(freq[:50]):.4f} Hz  (expect ~60.000)")
    print(f"  final freq         : {np.mean(freq[-50:]):.4f} Hz")
    print(f"  per-unit speed mean: {np.mean(traj['speed_pu']):.5f}  (expect ~1.0)")
    for k, v in m.as_row().items():
        print(f"  {k:14s} = {v}")

    # dump the trace for plotting
    out = Path("results/figures") / f"smoke_{args.disturbance}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, np.column_stack([traj["t"], freq]),
               delimiter=",", header="t_s,coi_freq_hz", comments="")
    print(f"\n[smoke] COI trace -> {out}")
    print("[smoke] plot it (e.g. with: gnuplot / python-matplotlib) and eyeball "
          "the shape against the expectation in this script's docstring.")


if __name__ == "__main__":
    main()
