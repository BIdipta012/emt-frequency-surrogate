"""
active_learning.py
==================
The novelty hook: instead of sampling operating points blindly, use a Gaussian
Process surrogate's UNCERTAINTY to decide which point to simulate next. This
reaches a target accuracy with far fewer expensive EMT runs -- the publishable
story is "active learning cuts required EMT runs by X% versus space-filling LHS".

Uses scikit-learn's GaussianProcessRegressor (CPU, no extra heavy deps). For a
larger study you can swap in GPyTorch/BoTorch, but this is enough to demonstrate
and quantify the effect.

Strategy (pool-based active learning):
  1. Start from a small seed set of simulated points.
  2. Fit a GP. Over a large candidate POOL (cheap to enumerate), predict mean+std.
  3. Query the candidate with the highest predictive std (most informative).
  4. Simulate it, add to the training set, refit. Repeat.
  5. Track held-out R^2 vs number of simulations, against a random-sampling
     baseline. The gap is your result.

Usage
-----
    python -m src.active_learning --disturbance islanding --target rocof_hz_s \
        --seed-n 15 --queries 40 --pool 600
"""

from __future__ import annotations
import argparse
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from .sampling import make_design, FEATURES
from .paraemt_driver import run_case



def _gp():
    kernel = (ConstantKernel(1.0) * Matern(length_scale=np.ones(len(FEATURES)),
                                            nu=2.5)
              + WhiteKernel(noise_level=1e-3))
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                    n_restarts_optimizer=2, random_state=0)


def _simulate_pool(disturbance: str, n: int, seed: int):
    design = make_design(n_per_disturbance=n, seed=seed, disturbances=[disturbance])
    rows = [run_case(p) for p in design]
    import pandas as pd
    df = pd.DataFrame(rows)
    df = df[df["converged"]].dropna()
    return df


def run(disturbance: str, target: str, seed_n: int, queries: int,
        pool_n: int, test_n: int = 200) -> dict:
    print(f"[AL] disturbance={disturbance} target={target}")
    # candidate pool + held-out test set (simulated up front for the experiment;
    # in a real run you'd only simulate queried points)
    pool = _simulate_pool(disturbance, pool_n, seed=1)
    test = _simulate_pool(disturbance, test_n, seed=999)
    Xpool, ypool = pool[FEATURES].values, pool[target].values
    Xtest, ytest = test[FEATURES].values, test[target].values

    scaler = StandardScaler().fit(Xpool)
    Xpool_s, Xtest_s = scaler.transform(Xpool), scaler.transform(Xtest)

    rng = np.random.default_rng(0)

    def curve(strategy: str):
        idx = list(rng.choice(len(Xpool), size=seed_n, replace=False))
        remaining = set(range(len(Xpool))) - set(idx)
        scores = []
        for q in range(queries + 1):
            gp = _gp().fit(Xpool_s[idx], ypool[idx])
            pred = gp.predict(Xtest_s)
            scores.append((seed_n + q, float(r2_score(ytest, pred))))
            if not remaining:
                break
            rem = np.array(sorted(remaining))
            if strategy == "active":
                _, std = gp.predict(Xpool_s[rem], return_std=True)
                pick = rem[int(np.argmax(std))]          # most uncertain
            else:
                pick = rem[int(rng.integers(len(rem)))]  # random
            idx.append(int(pick))
            remaining.discard(int(pick))
        return scores

    active = curve("active")
    random = curve("random")

    # how many runs does each need to first reach R2>=0.9?
    def runs_to(target_r2, scores):
        for n, r in scores:
            if r >= target_r2:
                return n
        return None
    a90, r90 = runs_to(0.9, active), runs_to(0.9, random)
    print(f"  runs to reach R2>=0.90:  active={a90}  random={r90}")
    if a90 and r90:
        print(f"  active-learning saving: {100*(1 - a90/r90):.0f}% fewer runs")

    return {"active": active, "random": random, "a90": a90, "r90": r90}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disturbance", default="islanding")
    ap.add_argument("--target", default="rocof_hz_s")
    ap.add_argument("--seed-n", type=int, default=15)
    ap.add_argument("--queries", type=int, default=40)
    ap.add_argument("--pool", type=int, default=400)
    args = ap.parse_args()
    run(args.disturbance, args.target, args.seed_n, args.queries, args.pool)


if __name__ == "__main__":
    main()
