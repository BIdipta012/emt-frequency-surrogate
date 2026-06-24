"""
sampling.py  (Option 2: full re-init path)
==========================================
Design of experiments for the VALIDATED real-ParaEMT pipeline. Because every run
re-initialises from the bundled case (Option 2), the operating point itself can
vary per run -- not just the disturbance. Two continuous covariates apply to
EVERY run, and three disturbance classes set the transient event.

Continuous operating-point inputs (LHS samples these, used by all runs):
  * h_scale    : inertia multiplier on dyd.gen_H. DOMINANT driver of RoCoF/nadir.
                 Safe to vary (no power-flow effect). [VERIFY it moves the nadir.]
  * load_level : load (and matching generation) scaling. Approximate balance.

Event inputs (used by the relevant disturbance only):
  * t_event    : trip time (gentrip) / PLL-release time (islanding)
  * i_gentrip  : which generator trips (gentrip only), 0..NGEN-1

Disturbance classes (names MUST match paraemt_run._configure_disturbance):
  * baseline   : no event -- a quiet operating point at (h_scale, load_level)
  * islanding  : const-Z load + PLL release at t_event
  * gentrip    : trip machine i_gentrip at t_event (the severe case)

Note: 'heavy_load' is NOT a separate class -- a heavier load is just a higher
load_level under any disturbance, and there is no runtime load-STEP mechanism in
this ParaEMT main script, so a sustained heavier load alone produces no transient.
Per-disturbance models remain the right call.
"""

from __future__ import annotations
import numpy as np
from SALib.sample import latin

NGEN = 4  # Kundur two-area: machines 0..3

# (name, low, high) -- continuous space sampled by Latin Hypercube
CONTINUOUS = [
    ("h_scale",    0.5, 1.5),   # inertia multiplier (dominant RoCoF driver)
    ("load_level", 0.8, 1.2),   # load/gen scaling (approx-balanced)
    ("t_event",    0.5, 3.0),   # event instant (s)
    ("i_gen_f",    0.0, NGEN - 1e-3),  # continuous proxy -> int i_gentrip
]

DISTURBANCES = ["baseline", "islanding", "gentrip"]

# Canonical model-feature list (the columns ML trains on). Note i_gentrip, not
# the continuous i_gen_f proxy. Single source of truth -- train.py and
# active_learning.py import this so names never drift.
FEATURES = ["h_scale", "load_level", "t_event", "i_gentrip"]


def _problem():
    return {
        "num_vars": len(CONTINUOUS),
        "names": [c[0] for c in CONTINUOUS],
        "bounds": [[c[1], c[2]] for c in CONTINUOUS],
    }


def make_design(n_per_disturbance: int = 100, seed: int = 0,
                disturbances: list[str] | None = None) -> list[dict]:
    """n_per_disturbance LHS points per disturbance type."""
    disturbances = disturbances or DISTURBANCES
    problem = _problem()
    rng = np.random.default_rng(seed)
    design: list[dict] = []
    run_id = 0
    for d in disturbances:
        X = latin.sample(problem, n_per_disturbance, seed=int(rng.integers(1e9)))
        for row in X:
            vals = {name: float(v) for name, v in zip(problem["names"], row)}
            p = {
                "disturbance": d,
                "h_scale": vals["h_scale"],
                "load_level": vals["load_level"],
                "t_event": vals["t_event"],
                "i_gentrip": int(np.clip(int(vals["i_gen_f"]), 0, NGEN - 1)),
                "seed": run_id,
                "run_id": run_id,
            }
            # zero out event features the disturbance doesn't use (keeps dataset
            # columns consistent; the driver/runner ignore irrelevant ones)
            if d == "baseline":
                p["t_event"] = 0.0
                p["i_gentrip"] = 0
            elif d == "islanding":
                p["i_gentrip"] = 0
            # gentrip uses t_event and i_gentrip
            design.append(p)
            run_id += 1
    return design


if __name__ == "__main__":
    d = make_design(n_per_disturbance=3)
    print(f"{len(d)} runs total ({len(DISTURBANCES)} classes); first 6:")
    for p in d[:6]:
        print(" ", p)
