"""
mock_simulator.py  (Option 2 parameter set)
===========================================
Physics-based swing-equation stand-in for ParaEMT, kept in lock-step with the
REAL parameter names so the pipeline (sampling -> driver -> metrics -> ML) runs
identically whether USE_MOCK is True or False. Same knobs, same disturbance set:

    h_scale     : inertia multiplier (base H ~ 6.3 s)  -> effective H = BASE_H*h_scale
    load_level  : load scaling (affects governor/damping)
    disturbance : {baseline, islanding, gentrip}
    t_event     : event time (s)
    i_gentrip   : which machine trips (gentrip) -- in the mock, larger index = a
                  slightly bigger share of generation lost (cosmetic variety)

Crude on purpose: it only needs to give ML-shaped inputs/outputs and reproduce
the qualitative trends (lower inertia -> deeper nadir / larger RoCoF; gentrip
more severe than islanding). Real physics comes from ParaEMT.
"""

from __future__ import annotations
import numpy as np

BASE_H = 6.3   # ~ mean of the real dyd.gen_H = [6.5, 6.5, 6.175, 6.175]


def simulate(params: dict, dt: float = 5e-4, t_end: float = 10.0,
             f0: float = 60.0, seed: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    h_scale = float(params.get("h_scale", 1.0))
    load = float(params.get("load_level", 1.0))
    dist = params.get("disturbance", "baseline")
    t_event = float(params.get("t_event", 1.0))
    i_gentrip = int(params.get("i_gentrip", 0))

    H = BASE_H * h_scale
    w0 = 2 * np.pi * f0
    t = np.arange(0.0, t_end, dt)
    T = len(t)
    dw = np.zeros(T)

    droop = 0.05
    Kgov = load / droop
    D = 1.0 + 0.2 * load
    Heff = 2 * H / w0

    # disturbance power imbalance Pdist(t), pu
    Pdist = np.zeros(T)
    if dist == "gentrip":
        # lose ~1/4 of generation at t_event; larger i -> marginally larger loss
        frac = 0.22 + 0.02 * i_gentrip
        Pdist += frac * (t >= t_event)
    elif dist == "islanding":
        # grid support removed -> a swing both ways; model as a damped impulse
        amp = 0.18
        ev = (t >= t_event)
        Pdist += amp * np.sin(2 * np.pi * 0.7 * (t - t_event).clip(0)) * ev \
                 * np.exp(-(t - t_event).clip(0) / 3.0)
    # baseline: no imbalance

    diverged = False
    for k in range(1, T):
        restoring = Kgov * dw[k-1]
        ddw = (-Pdist[k-1] - restoring - D * dw[k-1]) / Heff
        dw[k] = dw[k-1] + dt * ddw
        if not np.isfinite(dw[k]) or abs(dw[k]) > 5.0:
            diverged = True
            dw[k:] = dw[k-1]
            break

    dw = dw + rng.normal(0, 2e-4, size=T)
    speed_pu = (1.0 + dw)[:, None]            # single aggregate machine (T,1)
    return {
        "t": t,
        "speed_pu": speed_pu,
        "H": np.array([H]),
        "converged": not diverged,
        "dt": dt,
    }
