"""
generate_dataset.py
===================
Orchestrator: build the DoE, run every case (in parallel across your cores),
and write a tidy dataset.csv that the ML code consumes.

Robustness is deliberate: failed/diverged runs are KEPT (converged=False) rather
than dropped, because the convergence boundary often traces the real stability
limit. Filter them out at training time if you want, but keep the record.

Usage
-----
    python -m src.generate_dataset --n 50  --jobs 8 --out data/processed/pilot.csv
    python -m src.generate_dataset --n 400 --jobs 8 --out data/processed/full.csv
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from .sampling import make_design
from .paraemt_driver import run_case


def generate(n_per_disturbance: int, jobs: int, out_path: str,
             disturbances: list[str] | None = None, seed: int = 0) -> pd.DataFrame:
    design = make_design(n_per_disturbance, seed=seed, disturbances=disturbances)
    total = len(design)
    print(f"[gen] {total} cases ({n_per_disturbance}/disturbance) on {jobs} workers")

    rows, done, t0 = [], 0, time.time()
    # NOTE: with the real ParaEMT driver, each task is heavy and CPU-bound, so
    # ProcessPoolExecutor across physical cores is the right tool. With the mock
    # it's near-instant; jobs is then mostly irrelevant.
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futures = {ex.submit(run_case, p): p["run_id"] for p in design}
        for fut in as_completed(futures):
            rows.append(fut.result())
            done += 1
            if done % max(1, total // 20) == 0 or done == total:
                el = time.time() - t0
                rate = done / el if el else 0
                eta = (total - done) / rate if rate else 0
                print(f"  {done:>4}/{total}  ({100*done/total:4.0f}%)  "
                      f"elapsed {el:6.1f}s  eta {eta:6.1f}s")

    df = pd.DataFrame(rows).sort_values("run_id").reset_index(drop=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # quick health report
    n_ok = int(df["converged"].sum())
    print(f"[gen] wrote {out_path}: {len(df)} rows, "
          f"{n_ok} converged ({100*n_ok/len(df):.0f}%), "
          f"{len(df)-n_ok} failed/diverged")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50,
                    help="LHS points PER disturbance type")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", type=str, default="data/processed/dataset.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    generate(args.n, args.jobs, args.out, seed=args.seed)


if __name__ == "__main__":
    main()
