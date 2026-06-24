# EMT Stability-Metric Surrogate (ParaEMT)

A machine-learning surrogate that predicts power-system frequency-stability
metrics — frequency nadir, RoCoF, settling time — from operating conditions,
trained on ParaEMT electromagnetic-transient (EMT) simulations of the modified
Kundur two-area benchmark (ParaEMT *system 6*).

One EMT run takes ~10 minutes; the trained surrogate predicts in well under a
millisecond. That ~10⁵–10⁶× speedup is the point: it makes dense sensitivity
maps, protection-threshold sweeps, and battery-storage sizing tractable, none of
which are feasible with raw EMT.

## Why this design

- **Labels first.** The surrogate can only be as good as its labels. Frequency is
  reconstructed as **center-of-inertia frequency in Hz**, RoCoF is computed over a
  **measurement window** (grid-code style, not a raw finite difference), and
  settling time has an honest "never settled" sentinel. See `src/metrics.py`.
- **One integration point.** Everything calls `paraemt_driver.run_case(params)`.
  A physics-based **mock simulator** lets you build and test the entire pipeline
  with no ParaEMT installed; flip one flag to switch to the real thing.
- **Per-disturbance models.** Fault and no-fault cases live in different regimes;
  pooling them hurts accuracy. The sampler and trainer support a model per
  disturbance class.
- **Active learning (the novelty hook).** A Gaussian-process surrogate uses its
  own uncertainty to choose which operating point to simulate next, reaching
  target accuracy with fewer expensive EMT runs.

## Layout

```
emt-surrogate/
├── config.yaml              # parameter ranges & run settings (one place)
├── requirements.txt
├── src/
│   ├── metrics.py           # KEYSTONE: COI freq, windowed RoCoF, settling
│   ├── mock_simulator.py    # swing-equation stand-in (run the pipeline today)
│   ├── paraemt_driver.py    # THE integration point: run_case(params)->metrics
│   ├── sampling.py          # Latin Hypercube design of experiments
│   ├── generate_dataset.py  # parallel runs -> dataset.csv
│   ├── train.py             # baseline + gradient-boosted trees + importances
│   └── active_learning.py   # GP + uncertainty sampling
├── scripts/
│   └── run_pilot.sh         # end-to-end smoke test on the mock
├── data/{raw,processed}/    # trajectories / dataset.csv
└── results/{models,figures}/
```

## Quickstart (works immediately, mock simulator)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/run_pilot.sh
```

This validates labels, generates a pilot dataset, trains per-disturbance
surrogates, and runs the active-learning demo — all on the mock, in under a
minute.

## Wiring in ParaEMT

1. `git clone https://github.com/NREL/ParaEMT_public` and confirm you can run
   your system-6 case from its snapshot `sim_snp_S6_50u_1pt.pkl`.
2. In `src/paraemt_driver.py`, set `PARAEMT_DIR` and fill the three `TODO`
   blocks in `_run_paraemt`: (1) load the snapshot, (2) apply the sampled
   parameters & disturbance using your existing scenario code, (3) run the loop
   and return every machine's per-unit speed plus the inertia vector `H`.
3. **Identify the speed-state indices once** by inspecting the state ordering in
   `psutils.py`, and record them — this is the state-identification fix.
   Verify units with `src/validate_labels.py` (extracted RoCoF must track the
   analytic `dP·f0/(2·H)` estimate).
4. Set `USE_MOCK = False`. Nothing else changes.

## Reproducibility

Pin versions via `requirements.txt`, commit `config.yaml`, and deposit the final
dataset + code on Zenodo for a DOI. ParaEMT is BSD-licensed and free, so the
whole workflow is open-source end to end.

## Citing ParaEMT

M. Xiong et al., "ParaEMT: an open source, parallelizable, and HPC-compatible
EMT simulator for large-scale IBR-rich power grids," *IEEE Trans. Power Del.*,
vol. 39, no. 2, pp. 911–921, Apr. 2024.
