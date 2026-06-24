"""
paraemt_driver.py
=================
THE INTEGRATION POINT. Everything else in the project calls `run_case(params)`
and gets back a metrics dict. There are exactly two implementations:

    * USE_MOCK = True   -> swing-equation mock (works today, no ParaEMT needed)
    * USE_MOCK = False  -> real ParaEMT, resumed from the system-6 snapshot

Develop the whole pipeline with the mock. When your metrics and ML code are
solid, flip USE_MOCK to False and fill in the three TODO blocks in
`_run_paraemt`. Nothing else in the codebase changes.

----------------------------------------------------------------------
How the real ParaEMT path works (from the NREL ParaEMT_public structure):
  * The repo ships snapshot files; sim_snp_S6_50u_1pt.pkl is YOUR system 6 at a
    converged steady state. Resuming from it skips the slow power-flow + init,
    which is what makes a few hundred runs affordable.
  * main_step1_simulation.py drives a time loop on an EMT object (commonly `emt`,
    an EmtSimu instance from psutils.py). State lives in emt.x; saved/downsampled
    states accumulate during the loop; dyd/pfd hold dynamic & power-flow data
    (machine H lives in the dynamic data).
  * You apply a disturbance by modifying the object mid-loop (load change, fault
    bus admittance, PLL release) exactly as your existing scenario scripts do.

So the adapter is: load snapshot -> apply params -> run loop -> pull machine
speeds + H -> hand to metrics.compute_metrics. Keep this file the ONLY place
that imports ParaEMT.
----------------------------------------------------------------------
"""

from __future__ import annotations
import time
import numpy as np

from .metrics import coi_frequency, compute_metrics
from . import mock_simulator

# Flip to False once you've validated a single real run (see paraemt_run.py).
USE_MOCK = False

# Path to the ParaEMT checkout (use the STABLE /mnt path, not the symlink).
PARAEMT_DIR = "/mnt/BLACK/ACADEMIC/publications/ParaEMT/claudefix/files/emt-surrogate/ParaEMT_public-main"
SNAPSHOT_S6 = "sim_snp_S6_50u_1pt.pkl"   # unused in the re-init (Option 2) path

RAW_TS = 50e-6     # ParaEMT raw time step (50 us)
DS_RATE = 10       # down-sample: save every 10th step
DT = RAW_TS * DS_RATE   # 500 us post-downsample sampling step
T_END = 10.0
F0 = 60.0


def run_case(params: dict, save_trajectory: bool = False) -> dict:
    """
    Run one EMT case and return a flat dict of inputs + metric labels.

    This is the function the dataset generator calls in parallel. It must NEVER
    raise on a failed simulation -- it records converged=False and returns
    sentinel labels, because convergence-failure boundaries are themselves
    informative (they trace the stability limit).
    """
    t0 = time.time()
    try:
        if USE_MOCK:
            traj = mock_simulator.simulate(params, dt=DT, t_end=T_END, f0=F0,
                                           seed=params.get("seed"))
        else:
            traj = _run_paraemt(params)

        freq = coi_frequency(_to_pu_speed(traj["speed_pu"]), traj["H"], f0=F0)
        m = compute_metrics(freq, dt=traj["dt"], f0=F0,
                            converged=traj["converged"])
        row = {**params, **m.as_row()}
    except Exception as e:  # noqa: BLE001 - we want robustness over purity here
        row = {**params,
               "f_nadir_hz": np.nan, "f_zenith_hz": np.nan,
               "rocof_hz_s": np.nan, "settling_s": np.nan,
               "f_min_hz": np.nan, "f_max_hz": np.nan,
               "converged": False, "error": str(e)}

    row["runtime_s"] = round(time.time() - t0, 3)
    return row


def _to_pu_speed(speed: np.ndarray) -> np.ndarray:
    """
    Normalise the monitored speed to per-unit (nominal ~ 1.0).

    LABEL-REPAIR HOOK: in the prior paper the monitored state sat near ~377,
    which is electrical angular speed in rad/s (2*pi*60), NOT per-unit. If your
    real ParaEMT state is in rad/s, the heuristic below rescales it. VERIFY the
    actual units from the ParaEMT model definition and harden this.
    """
    speed = np.asarray(speed, dtype=float)
    if speed.ndim == 1:
        speed = speed[:, None]
    median = np.median(speed)
    if median > 50:                      # looks like rad/s (~377), not pu
        speed = speed / (2 * np.pi * F0)
    elif median > 1.5:                   # looks like Hz (~60)
        speed = speed / F0
    return speed


# ----------------------------------------------------------------------
# REAL ParaEMT adapter -- fill in the three TODO blocks, then set USE_MOCK=False
# ----------------------------------------------------------------------
def _run_paraemt(params: dict) -> dict:
    """
    Build system 6 fresh (license-free, no snapshot), apply load_level and the
    disturbance, run the loop, and return a trajectory dict with keys:
    't', 'speed_pu' (T,M), 'H' (M,), 'converged', 'dt'.

    All ParaEMT specifics live in paraemt_run; this just orchestrates the three
    stages. The state-identification, units, and inertia fixes are all inside
    paraemt_run.extract_trajectory.
    """
    from . import paraemt_run

    pfd, ini, dyd, emt = paraemt_run.init_case(
        params, paraemt_dir=PARAEMT_DIR, systemN=6, ts=RAW_TS, Tlen=T_END
    )
    emt = paraemt_run.run_emt_loop(pfd, dyd, ini, emt,
                                   ts=RAW_TS, Tlen=T_END, DSrate=DS_RATE)
    return paraemt_run.extract_trajectory(emt, pfd, dyd)


if __name__ == "__main__":
    demo = dict(H_sys=4.0, scr=3.0, load_level=1.1,
                disturbance="islanding", dP=0.15, t_event=1.5, seed=0)
    out = run_case(demo)
    for k, v in out.items():
        print(f"  {k:14s} = {v}")
