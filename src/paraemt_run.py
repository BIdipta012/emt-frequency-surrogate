"""
paraemt_run.py
==============
A CALLABLE refactor of ParaEMT's main_step1_simulation.py so it can be driven in
a parametric loop. The stock script does everything inside a single main() with
hardcoded disturbance settings and timing instrumentation; this splits it into
three functions the surrogate pipeline can call per case:

    init_case(params, ...)              -> (pfd, dyd, ini, emt)
    run_emt_loop(pfd, dyd, ini, emt...) -> emt   (populated state history)
    extract_trajectory(emt, pfd, dyd)   -> trajectory dict for metrics.py

This file is the ONLY place that imports ParaEMT. It is imported by
paraemt_driver._run_paraemt.

IMPORTANT — this is integration code that has NOT been run against real ParaEMT
in development; it was reconstructed from the source you traced. Treat the first
run as a VALIDATION run (one case, eyeball the trace), not a batch. The places
most likely to need adjustment are flagged with [VERIFY].

State map (verified from Lib_BW.py CombineX):
    GENROU machines, 18 states each, contiguous from index 0.
    per-machine offset 1 = rotor speed, in rad/s (init 1.0*pfd.ws, ws=2*pi*60).
    machine i speed index = dyd.gen_genrou_xi_st + i*dyd.gen_genrou_odr + 1.
    inertia: dyd.gen_H (pu on machine MVA base).
"""

from __future__ import annotations
import os
import sys
import numpy as np


def init_case(params, paraemt_dir, systemN=6, ts=50e-6, Tlen=10.0):
    """
    Build a fresh, license-free case from the bundled JSON/Excel (no snapshot,
    no PSS/E), apply the load_level scaling, configure the disturbance, and
    return the four ParaEMT objects ready to run.
    """
    # ParaEMT reads cases/ and models/ by RELATIVE path and does os.chdir, so we
    # must run from inside the ParaEMT directory.
    cwd0 = os.getcwd()
    sys.path.insert(0, paraemt_dir)
    os.chdir(paraemt_dir)
    try:
        from psutils import initialize_emt

        netMod = "lu"
        nparts = 2
        (pfd, ini, dyd, emt) = initialize_emt(
            ".", systemN, 1, 1, ts, Tlen, mode=netMod, nparts=nparts
        )
        emt.ts = ts
        emt.Tlen = Tlen

        _apply_inertia(dyd, params.get("h_scale", 1.0))
        _apply_load_level(pfd, params.get("load_level", 1.0))
        _configure_disturbance(emt, params, ts, Tlen)
        return pfd, ini, dyd, emt
    finally:
        os.chdir(cwd0)


def _apply_inertia(dyd, h_scale):
    """
    Scale every generator's inertia constant by h_scale.

    Unlike load, inertia has NO effect on the power-flow steady state -- it only
    changes the dynamic (swing) response. So this is a clean, safe knob with no
    rebalancing needed, and it is the dominant driver of RoCoF and nadir, which
    makes it the single most valuable input dimension. h_scale=1.0 is a no-op.

    [VERIFY] dyd.gen_H is consumed during the InitMac/CombineX init chain, which
    has ALREADY run by the time init_case calls this. For the GENROU swing
    equation the relevant inertia is read from dyd.gen_H at each step, so scaling
    it post-init should take effect -- but confirm on a smoke run that h_scale<1
    deepens the nadir and h_scale>1 shallows it. If it has no effect, the inertia
    was baked into a derived coefficient at init and h_scale must instead be
    applied to dyd.gen_H BEFORE Initialize() (i.e. inside init_case before the
    initialize_emt call, which needs a small restructure).
    """
    if abs(h_scale - 1.0) < 1e-9:
        return
    dyd.gen_H = np.asarray(dyd.gen_H, dtype=float) * h_scale


def _apply_load_level(pfd, load_level):
    """
    Scale load (and matching generation) to hold the case approximately balanced.

    The bundled JSON is PRE-SOLVED and ParaEMT does not re-run a power flow, so
    scaling load alone would create a power imbalance and a spurious t=0 jump.
    We therefore scale generation by the same factor. This is APPROXIMATE (it
    ignores how network losses change) but adequate for modest load swings, and
    must be stated as such in the paper's methods.

    When load_level == 1.0 we skip scaling entirely, guaranteeing a perfectly
    balanced steady state -- use this for the first validation run.
    """
    if abs(load_level - 1.0) < 1e-9:
        return
    pfd.load_P = pfd.load_P * load_level        # complex MVA array
    pfd.load_MW = pfd.load_MW * load_level
    if hasattr(pfd, "load_Mvar"):
        pfd.load_Mvar = pfd.load_Mvar * load_level
    # scale active generation to match; reactive is set by voltage control
    pfd.gen_MW = pfd.gen_MW * load_level
    if hasattr(pfd, "gen_S"):
        pfd.gen_S = pfd.gen_S * load_level


def _configure_disturbance(emt, params, ts, Tlen):
    """
    Map a params['disturbance'] onto ParaEMT's event flags.

    Disturbance classes available in THIS version of main_step1 (verified):
        baseline / strong_grid : no event (quiet reference)
        heavy_load             : sustained load step (applied at init via scaling)
        islanding              : const-Z load + PLL release at t_event
        gentrip                : generator trip at t_event (the severe case)

    NOTE: a true bus/line LLG fault is NOT exposed in this main script (the
    README mentions a newer fault function not present here). 'gentrip' is the
    clean, modeled severe disturbance and replaces the prior paper's LLG case.

    [VERIFY] To DISABLE an event we push its time beyond the horizon rather than
    zeroing its flag -- because the loop runs emt.Re_Init() every step when
    flag_gentrip==0, which is only meant to happen AFTER a real trip. Keeping
    flag_gentrip==1 with t_gentrip past the end keeps the normal updateIhis path.
    Confirm on the first baseline run that the trace is flat near 60 Hz.
    """
    beyond = Tlen + 100.0
    dist = params.get("disturbance", "baseline")

    # --- defaults: all events disabled, const-RLC load, PLL released at t=0 ---
    emt.t_sc = beyond
    emt.i_gen_sc = 0
    emt.flag_exc_gov = 1
    emt.dsp = 0.0
    emt.flag_sc = 1
    emt.t_gentrip = beyond
    emt.i_gentrip = 0
    emt.flag_gentrip = 1
    emt.flag_reinit = 1
    emt.t_release_f = float(params.get("t_release_f", 0.0))
    emt.loadmodel_option = 1

    if dist in ("baseline", "strong_grid", "heavy_load"):
        pass  # heavy_load is realised by _apply_load_level, no runtime event
    elif dist == "islanding":
        emt.loadmodel_option = 2                       # const-Z
        emt.t_release_f = float(params.get("t_event", 1.5))
    elif dist == "gentrip":
        emt.t_gentrip = float(params.get("t_event", 1.0))
        emt.i_gentrip = int(params.get("i_gentrip", 0))
    elif dist == "stepchange":
        emt.t_sc = float(params.get("t_event", 1.0))
        emt.dsp = float(params.get("dsp", -0.02))
        emt.flag_exc_gov = int(params.get("flag_exc_gov", 1))
    else:
        raise ValueError(f"unknown disturbance: {dist}")
    return emt


def run_emt_loop(pfd, dyd, ini, emt, ts=50e-6, Tlen=10.0, DSrate=10):
    """
    The time loop, lifted verbatim from main_step1_simulation.py (lines ~81-147)
    with timing instrumentation removed and no disk dump. Returns emt with its
    state history populated in emt.x / emt.t.
    """
    netMod = "lu"
    tn = 0
    tsave = 0
    while tn * ts < Tlen:
        tn = tn + 1
        emt.StepChange(dyd, ini, tn)
        emt.GenTrip(pfd, dyd, ini, tn, netMod)

        emt.predictX(pfd, dyd, emt.ts)

        emt.Igs = emt.Igs * 0
        emt.updateIg(pfd, dyd, ini)

        emt.Igi = emt.Igi * 0
        emt.Iibr = emt.Iibr * 0
        emt.updateIibr(pfd, dyd, ini)

        if emt.loadmodel_option != 1:
            emt.Il = emt.Il * 0
            emt.updateIl(pfd, dyd, tn)

        emt.solveV(ini)
        emt.BusMea(pfd, dyd, tn)

        emt.updateX(pfd, dyd, ini, tn)
        emt.updateXibr(pfd, dyd, ini, ts)
        if emt.loadmodel_option != 1:
            emt.updateXl(pfd, dyd, tn)

        emt.x_pred = {0: emt.x_pred[1], 1: emt.x_pred[2], 2: emt.x_pv_1}

        if np.mod(tn, DSrate) == 0:
            tsave = tsave + 1
            emt.t.append(tn * ts)
            emt.x[tsave] = emt.x_pv_1.copy()
            if len(pfd.ibr_bus) > 0:
                emt.x_ibr[tsave] = emt.x_ibr_pv_1.copy()
            if len(pfd.bus_num) > 0:
                emt.x_bus[tsave] = emt.x_bus_pv_1.copy()
            if len(pfd.load_bus) > 0:
                emt.x_load[tsave] = emt.x_load_pv_1.copy()
            emt.v[tsave] = emt.Vsol.copy()

        if (emt.flag_gentrip == 0) & (emt.flag_reinit == 1):
            emt.Re_Init(pfd, dyd, ini)
        else:
            emt.updateIhis(ini)

    return emt


def extract_trajectory(emt, pfd, dyd):
    """
    Pull the (T, 4) per-unit machine-speed array and inertia vector from a
    completed run, in the exact shape metrics.coi_frequency expects.
    """
    # flatten emt.x (dict keyed by save index) into (T, n_states)
    keys = sorted(k for k in emt.x.keys())
    x_arr = np.array([emt.x[k] for k in keys])          # (T, n_states)
    t_arr = np.asarray(emt.t)                            # (T,)

    xi_st = int(dyd.gen_genrou_xi_st)                    # 0
    odr = int(dyd.gen_genrou_odr)                        # 18
    ngen = len(pfd.gen_bus)                              # 4
    speed_idx = [xi_st + i * odr + 1 for i in range(ngen)]   # [1,19,37,55]

    speed_pu = x_arr[:, speed_idx] / pfd.ws             # rad/s -> per-unit
    H = np.asarray(dyd.gen_H, dtype=float)              # (ngen,) pu inertia

    converged = bool(np.all(np.isfinite(speed_pu)))
    dt = float(t_arr[1] - t_arr[0]) if len(t_arr) > 1 else 5e-4
    return {"t": t_arr, "speed_pu": speed_pu, "H": H,
            "converged": converged, "dt": dt}
