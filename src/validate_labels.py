"""
validate_labels.py
==================
THE KEYSTONE SANITY CHECK. Before trusting any label, confirm that the RoCoF the
pipeline extracts agrees with the analytic swing-equation estimate for a known
power imbalance. If these don't roughly match, the labels are wrong and the whole
surrogate would learn nonsense -- this is the fix for the prior paper's
non-physical numbers.

Analytic initial RoCoF for a sudden active-power imbalance dP (pu) on a system
with aggregate inertia H_sys (s) at nominal f0:

        RoCoF_0  =  dP * f0 / (2 * H_sys)      [Hz/s]

We run a clean load-step case through the SAME extraction path used for the
dataset, then compare.

Usage
-----
    python -m src.validate_labels
"""

from __future__ import annotations
import numpy as np

from .paraemt_driver import run_case, F0


def analytic_rocof(dP: float, H_sys: float, f0: float = F0) -> float:
    return abs(dP) * f0 / (2.0 * H_sys)


def check(dP: float = 0.10, H_sys: float = 5.0, tol_rel: float = 0.5) -> bool:
    params = dict(H_sys=H_sys, scr=8.0, load_level=1.0,
                  disturbance="heavy_load", dP=dP, t_event=1.5, seed=0)
    row = run_case(params)
    extracted = row["rocof_hz_s"]
    expected = analytic_rocof(dP, H_sys)
    rel = abs(extracted - expected) / expected if expected else float("inf")
    ok = rel <= tol_rel
    print(f"  dP={dP:.2f}  H={H_sys:.1f}  ->  "
          f"analytic={expected:.4f} Hz/s   extracted={extracted:.4f} Hz/s   "
          f"rel.err={rel:.0%}   {'PASS' if ok else 'CHECK'}")
    return ok


if __name__ == "__main__":
    print("[validate] swing-equation RoCoF cross-check")
    print("  (with the MOCK simulator the absolute match is approximate;")
    print("   with real ParaEMT this must agree within a sensible margin)")
    results = [check(dP, H) for dP in (0.05, 0.10, 0.20) for H in (3.0, 6.0)]
    print(f"[validate] {sum(results)}/{len(results)} within tolerance")
    print("  NOTE: trend matters most -- RoCoF should rise with dP and fall with H.")
