"""
metrics.py
==========
Stability-metric extraction from an EMT trajectory.

This module is the KEYSTONE of the whole project. The surrogate model can only be
as trustworthy as the labels it learns from, so the physics fixes flagged in the
previous paper are implemented here, deliberately and in one place:

  1. Frequency is the CENTER-OF-INERTIA (COI) frequency of the machines, in Hz,
     not a single machine's rotor speed in unknown units.
  2. RoCoF is computed over a measurement WINDOW (grid-code style), not as a raw
     max finite-difference, which on 50 us EMT data is dominated by fault spikes
     and numerical noise.
  3. Settling time uses an envelope criterion with a clear "never settles" sentinel
     instead of silently returning 0.

Units throughout: frequency in Hz, RoCoF in Hz/s, time in s.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np


# ----------------------------------------------------------------------
# Frequency reconstruction
# ----------------------------------------------------------------------
def coi_frequency(speed_pu: np.ndarray, H: np.ndarray, f0: float = 60.0) -> np.ndarray:
    """
    Center-of-inertia frequency in Hz from per-unit machine speeds.

    Parameters
    ----------
    speed_pu : (T, M) array
        Per-unit rotor speed of each of M machines over T time steps.
        In per-unit, nominal is 1.0. (If your raw state is electrical angular
        speed in rad/s ~ 377, divide by 2*pi*f0 BEFORE calling this -- see
        paraemt_driver.extract_machine_speeds for the conversion point.)
    H : (M,) array
        Inertia constants (s) of each machine, on a common base.
    f0 : float
        Nominal frequency (Hz).

    Returns
    -------
    (T,) array of COI frequency in Hz.
    """
    speed_pu = np.asarray(speed_pu, dtype=float)
    H = np.asarray(H, dtype=float)
    if speed_pu.ndim == 1:
        speed_pu = speed_pu[:, None]
    if H.shape[0] != speed_pu.shape[1]:
        raise ValueError(f"H has {H.shape[0]} machines but speed has {speed_pu.shape[1]}")
    coi_pu = (speed_pu * H).sum(axis=1) / H.sum()
    return coi_pu * f0


# ----------------------------------------------------------------------
# Individual metrics
# ----------------------------------------------------------------------
def nadir_depth(freq_hz: np.ndarray, f0: float = 60.0) -> float:
    """Maximum downward excursion from nominal, in Hz (>= 0)."""
    return float(f0 - np.min(freq_hz))

def zenith_rise(freq_hz: np.ndarray, f0: float = 60.0) -> float:
    """Maximum upward excursion from nominal, in Hz (>= 0)."""
    return float(np.max(freq_hz) - f0)

def windowed_rocof(freq_hz: np.ndarray, dt: float, window_s: float = 0.1) -> float:
    """
    Maximum |RoCoF| in Hz/s, measured over a sliding window of `window_s`.

    Grid codes (e.g. measurement windows of 100-500 ms) define RoCoF over a
    window, NOT instantaneously. A raw point-to-point derivative of 50 us EMT
    data mostly measures noise and fault-instant electrical spikes -- that was
    the source of the implausible 377 "units" in the prior paper.

    RoCoF over the window is (f[t+w] - f[t]) / w.
    """
    freq_hz = np.asarray(freq_hz, dtype=float)
    w = max(1, int(round(window_s / dt)))
    if w >= len(freq_hz):
        # window longer than the trajectory: fall back to whole-trace slope
        return abs(float((freq_hz[-1] - freq_hz[0]) / ((len(freq_hz) - 1) * dt)))
    diffs = (freq_hz[w:] - freq_hz[:-w]) / (w * dt)
    return float(np.max(np.abs(diffs)))

def settling_time(freq_hz: np.ndarray, dt: float, band_hz: float = 0.02,
                  tail_frac: float = 0.05) -> float:
    """
    Time (s) after which frequency stays within +/- band_hz of its final value.

    final value = mean of the last `tail_frac` of the trace.
    Returns np.nan if it never settles within the horizon (an honest sentinel,
    unlike silently returning 0.0).
    """
    freq_hz = np.asarray(freq_hz, dtype=float)
    n = len(freq_hz)
    tail = max(1, int(tail_frac * n))
    f_final = float(np.mean(freq_hz[-tail:]))
    within = np.abs(freq_hz - f_final) <= band_hz
    # find the last index that is OUTSIDE the band; settle just after it
    outside = np.where(~within)[0]
    if outside.size == 0:
        return 0.0
    last_out = outside[-1]
    if last_out >= n - 1:
        return float("nan")  # still oscillating at the end -> did not settle
    return float((last_out + 1) * dt)


# ----------------------------------------------------------------------
# One-call bundle
# ----------------------------------------------------------------------
@dataclass
class StabilityMetrics:
    f_nadir_hz: float        # downward excursion (Hz)
    f_zenith_hz: float       # upward excursion (Hz)
    rocof_hz_s: float        # windowed max |RoCoF| (Hz/s)
    settling_s: float        # settling time (s), nan if never settles
    f_min_hz: float          # absolute minimum frequency (Hz)
    f_max_hz: float          # absolute maximum frequency (Hz)
    converged: bool          # did the EMT run complete cleanly

    def as_row(self) -> dict:
        return asdict(self)


def compute_metrics(freq_hz: np.ndarray, dt: float, f0: float = 60.0,
                    rocof_window_s: float = 0.1, converged: bool = True
                    ) -> StabilityMetrics:
    """Compute the full metric bundle from a 1-D COI frequency trace (Hz)."""
    freq_hz = np.asarray(freq_hz, dtype=float)
    return StabilityMetrics(
        f_nadir_hz=nadir_depth(freq_hz, f0),
        f_zenith_hz=zenith_rise(freq_hz, f0),
        rocof_hz_s=windowed_rocof(freq_hz, dt, rocof_window_s),
        settling_s=settling_time(freq_hz, dt),
        f_min_hz=float(np.min(freq_hz)),
        f_max_hz=float(np.max(freq_hz)),
        converged=bool(converged),
    )


if __name__ == "__main__":
    # tiny self-test on a synthetic underfrequency event
    dt = 5e-4
    t = np.arange(0, 10, dt)
    # step drop then exponential recovery toward 59.7 Hz, lightly damped ring
    f = 60 - 0.4 * (t > 1.5) * (1 - np.exp(-(t - 1.5).clip(0) / 2.0)) \
          + 0.05 * np.sin(2 * np.pi * 0.8 * t) * np.exp(-t / 4)
    m = compute_metrics(f, dt)
    for k, v in m.as_row().items():
        print(f"  {k:14s} = {v}")
