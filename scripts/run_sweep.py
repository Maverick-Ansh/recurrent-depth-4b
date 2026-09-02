"""Run a queue of training arms across the available GPUs, one process per GPU.

Sec. 5 of the reproduction notes: more than one process per GPU gave no
throughput gain and a large per-step slowdown, so this is a strict pool of
size = n_gpus. Runs are skippable -- `train_scratch.py` returns immediately if
its result json already exists -- so a killed sweep resumes instead of
restarting.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_queue(steps: int, lr: float, out: str):
    """(name, argv) pairs, ordered so the claims that matter land first."""
    two_seeds = ["rec", "fixed1", "fixedr"]          # C1, C2, C1-mechanism
    one_seed = ["noinject", "prenorm", "dets0", "addinject", "nonormparams", "fullbp"]
    q = []
    for arm in two_seeds:
        for seed in (0, 1):
            q.append((f"{arm}_s{seed}", arm, seed))
    for arm in one_seed:
        q.append((f"{arm}_s0", arm, 0))
    for seed in (0, 1):                               # FLOP-matched control, runs longest
        q.append((f"fixed1_flop_s{seed}", "fixed1_flop", seed))
    jobs = []
    for name, arm, seed in q:
        jobs.append((name, [sys.executable, "scripts/train_scratch.py",
                            "--arm", arm, "--seed", str(seed), "--steps", str(steps),
                            "--lr", str(lr), "--out", out]))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--out", default="results")
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--only", default="", help="comma-separated job names to run")
    args = ap.parse_args()

    gpus = [g.strip() for g in args.gpus.split(",")]
    os.makedirs(f"{HERE}/{args.logs}", exist_ok=True)
    os.makedirs(f"{HERE}/{args.out}", exist_ok=True)
    jobs = build_queue(args.steps, args.lr, args.out)
    if args.only:
        keep = set(args.only.split(","))
        jobs = [j for j in jobs if j[0] in keep]
    jobs = [j for j in jobs if not os.path.exists(f"{HERE}/{args.out}/{j[0]}.json")]
    print(f"{len(jobs)} jobs to run on {len(gpus)} gpus", flush=True)

    running: dict[str, tuple] = {}
    t0 = time.time()
    while jobs or running:
        for g in gpus:
            if g not in running and jobs:
                name, argv = jobs.pop(0)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, PYTHONUNBUFFERED="1")
                f = open(f"{HERE}/{args.logs}/{name}.log", "w")
                p = subprocess.Popen(argv, cwd=HERE, env=env, stdout=f,
                                     stderr=subprocess.STDOUT, start_new_session=True)
                running[g] = (name, p, f, time.time())
                print(f"[{time.time()-t0:7.0f}s] gpu{g} start {name}", flush=True)
        for g in list(running):
            name, p, f, ts = running[g]
            if p.poll() is not None:
                f.close()
                print(f"[{time.time()-t0:7.0f}s] gpu{g} done  {name} "
                      f"rc={p.returncode} in {time.time()-ts:.0f}s", flush=True)
                del running[g]
        time.sleep(5)
    print(f"sweep complete in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
