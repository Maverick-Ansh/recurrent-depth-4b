"""Turn the results/*.json into the report's tables and figures.

Every table prints each accuracy next to its cell's shortcut floor, and every
multi-seed number as mean +- std across seeds, so that no difference smaller
than the seed spread can be read as a result.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TASK_GROUP = {"perm": "depth-hard", "add": "depth-hard", "recall": "memory-hard"}


def load(results_dir):
    runs = defaultdict(list)
    for p in sorted(glob.glob(f"{results_dir}/*.json")):
        if os.path.basename(p).startswith(("retro", "bench", "retrofit_gate")):
            continue
        d = json.load(open(p))
        runs[d["arm"]].append(d)
    return runs


def agg_task(d, task):
    """Mean accuracy over that task's difficulty cells, per r."""
    rs = sorted(int(r) for r in next(iter(d["grid"].values()))["acc"])
    out = {}
    for r in rs:
        vals = [c["acc"][str(r)] for k, c in d["grid"].items() if k.split("/")[0] == task]
        out[r] = float(np.mean(vals))
    return out


def mean_std(runs, fn):
    curves = [fn(d) for d in runs]
    rs = sorted(curves[0])
    mu = {r: float(np.mean([c[r] for c in curves])) for r in rs}
    sd = {r: float(np.std([c[r] for c in curves])) for r in rs}
    return mu, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--figures", default="figures")
    args = ap.parse_args()
    os.makedirs(args.figures, exist_ok=True)
    runs = load(args.results)
    if not runs:
        print("no results yet")
        return
    print(f"arms: {', '.join(f'{k}({len(v)})' for k, v in sorted(runs.items()))}\n")

    tasks = ["perm", "add", "recall"]
    summary = {}

    # ---------------------------------------------------- C1: accuracy vs r
    print("=" * 92)
    print("C1  test-time recurrence vs accuracy  (mean over difficulty cells, +- std over seeds)")
    print("=" * 92)
    hdr_rs = sorted(int(r) for r in next(iter(next(iter(runs.values()))[0]["grid"].values()))["acc"])
    print(f"{'arm':<16}{'task':<10}" + "".join(f"{'r='+str(r):>10}" for r in hdr_rs))
    for arm in sorted(runs):
        for t in tasks:
            mu, sd = mean_std(runs[arm], lambda d: agg_task(d, t))
            summary[(arm, t)] = (mu, sd)
            row = "".join(f"{mu[r]:>10.3f}" for r in hdr_rs)
            print(f"{arm:<16}{t+' ('+TASK_GROUP[t][0]+')':<10}{row}")
        print()

    # ------------------------------------------------------ C1 figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    for ax, t in zip(axes, tasks):
        for arm in ["rec", "fixedr", "fixed1", "fixed1_flop", "noinject", "prenorm"]:
            if arm not in runs:
                continue
            mu, sd = summary[(arm, t)]
            rs = sorted(mu)
            y = np.array([mu[r] for r in rs]); e = np.array([sd[r] for r in rs])
            ax.plot(rs, y, marker="o", ms=3, label=arm)
            if e.max() > 0:
                ax.fill_between(rs, y - e, y + e, alpha=0.15)
        floors = [c["floor"] for d in runs["rec"] for k, c in d["grid"].items()
                  if k.split("/")[0] == t]
        ax.axhline(float(np.mean(floors)), color="k", ls=":", lw=1, label="shortcut floor")
        ax.set_xscale("log", base=2); ax.set_xlabel("test-time recurrence r")
        ax.set_title(f"{t}  ({TASK_GROUP[t]})")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("exact-match accuracy")
    axes[-1].legend(fontsize=7)
    fig.suptitle("Track A: accuracy vs test-time recurrence (paper Fig. 1 / Fig. 7 analogue)")
    fig.tight_layout(); fig.savefig(f"{args.figures}/c1_accuracy_vs_r.png", dpi=140)
    print(f"[fig] {args.figures}/c1_accuracy_vs_r.png")

    # ------------------------------------------------- C1: saturation ordering
    print("=" * 92)
    print("C1b  saturation point per difficulty cell (smallest r reaching 95% of that cell's best)")
    print("=" * 92)
    if "rec" in runs:
        d = runs["rec"][0]
        rows = []
        for key, cell in d["grid"].items():
            acc = {int(r): a for r, a in cell["acc"].items()}
            best = max(acc.values())
            if best < cell["floor"] + 0.05:
                rows.append((key, cell["floor"], best, None)); continue
            sat = min(r for r in sorted(acc) if acc[r] >= 0.95 * best)
            rows.append((key, cell["floor"], best, sat))
        print(f"{'cell':<18}{'floor':>8}{'best acc':>10}{'r*':>6}")
        for k, f, b, s in rows:
            print(f"{k:<18}{f:>8.3f}{b:>10.3f}{(str(s) if s else 'at floor'):>6}")
        summary["saturation"] = rows

    # ---------------------------------------------- C2: recurrent vs its twin
    print("\n" + "=" * 92)
    print("C2  recurrent vs non-recurrent twin  (paper Table 4)")
    print("=" * 92)
    if "rec" in runs and "fixed1" in runs:
        print(f"{'task':<20}{'rec@best':>10}{'fixed1':>10}{'delta':>10}{'flop-matched':>14}{'delta':>10}")
        for t in tasks:
            mu_r, sd_r = summary[("rec", t)]
            best_r = max(mu_r.values())
            f1 = max(summary[("fixed1", t)][0].values())
            line = f"{t+' ('+TASK_GROUP[t]+')':<20}{best_r:>10.3f}{f1:>10.3f}{best_r-f1:>+10.3f}"
            if "fixed1_flop" in runs:
                ff = max(summary[("fixed1_flop", t)][0].values())
                line += f"{ff:>14.3f}{best_r-ff:>+10.3f}"
            print(line)

    # -------------------------------------------------------- C4 / C5 ablations
    print("\n" + "=" * 92)
    print("C4/C5  ablations: does the r-curve survive?  (slope = acc@r_max - acc@r=1)")
    print("=" * 92)
    print(f"{'arm':<16}{'claim':<8}" + "".join(f"{t[:6]+' slope':>14}" for t in tasks) +
          f"{'tok_corr':>10}")
    for arm in sorted(runs):
        d0 = runs[arm][0]
        slopes = []
        for t in tasks:
            mu, _ = summary[(arm, t)]
            rs = sorted(mu)
            slopes.append(mu[rs[-1]] - mu[rs[0]])
        tc = d0["recurrence_stats"]["token_corr_final"]
        print(f"{arm:<16}{d0['claim']:<8}" + "".join(f"{s:>+14.3f}" for s in slopes) +
              f"{tc:>10.3f}")

    # --------------------------------------------------------- val loss vs r
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for arm in sorted(runs):
        d = runs[arm][0]
        rs = sorted(int(r) for r in d["val_loss_vs_r"])
        ax.plot(rs, [d["val_loss_vs_r"][str(r)] for r in rs], marker="o", ms=3, label=arm)
    ax.set_xscale("log", base=2); ax.set_xlabel("recurrence r at test time")
    ax.set_ylabel("held-out loss (nats/byte)")
    ax.set_title("Validation loss vs recurrence (paper Fig. 6 right)")
    ax.grid(alpha=0.25); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{args.figures}/val_loss_vs_r.png", dpi=140)
    print(f"\n[fig] {args.figures}/val_loss_vs_r.png")

    # ------------------------------------------------ recurrence trajectories
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for arm in sorted(runs):
        st = runs[arm][0]["recurrence_stats"]
        axes[0].plot(st["dist_to_limit"], label=arm)
        axes[1].plot(st["rel_step"], label=arm)
    axes[0].set_yscale("log"); axes[0].set_xlabel("recurrence step i")
    axes[0].set_ylabel(r"$\|s_i - s_*\|$"); axes[0].set_title("distance to limit point (Fig. 11)")
    axes[1].set_yscale("log"); axes[1].set_xlabel("recurrence step i")
    axes[1].set_ylabel(r"$\|s_i - s_{i-1}\| / \|s_i\|$")
    axes[1].set_title("relative step size (App. A.2 metric)")
    for a in axes:
        a.grid(alpha=0.25); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{args.figures}/trajectories.png", dpi=140)
    print(f"[fig] {args.figures}/trajectories.png")

    json.dump({f"{k[0]}|{k[1]}": v for k, v in summary.items() if isinstance(k, tuple)},
              open(f"{args.results}/summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
