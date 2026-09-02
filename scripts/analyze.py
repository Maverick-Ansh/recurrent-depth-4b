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
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TASK_GROUP = {"perm": "depth-hard", "add": "depth-hard", "recall": "memory-hard"}


def load(results_dir):
    runs = defaultdict(list)
    for p in sorted(glob.glob(f"{results_dir}/*.json")):
        if os.path.basename(p).startswith(("retro", "bench", "retrofit_gate", "summary",
                                           "mechanisms")):
            continue
        d = json.load(open(p))
        if "arm" not in d:
            continue
        runs[d["arm"]].append(d)
    return runs


def agg_task(d, task, only_claim_cells=False):
    """Mean accuracy over that task's difficulty cells, per r.

    `only_claim_cells` restricts to cells whose prompt space is too large to
    tabulate, i.e. the ones where a score has to be computed rather than recalled.
    Both views are reported: mixing them would let memorisation on the easy end
    masquerade as reasoning.
    """
    from data import tasks as T
    keep = set(T.claim_cells())
    rs = sorted(int(r) for r in next(iter(d["grid"].values()))["acc"])
    out = {}
    for r in rs:
        vals = [c["acc"][str(r)] for k, c in d["grid"].items()
                if k.split("/")[0] == task and (not only_claim_cells or k in keep)]
        out[r] = float(np.mean(vals)) if vals else float("nan")
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
    print("(claim-carrying cells only -- prompt space too large to tabulate)")
    print(f"{'arm':<16}{'task':<22}" + "".join(f"{'r='+str(r):>9}" for r in hdr_rs))
    for arm in sorted(runs):
        for t in tasks:
            mu, sd = mean_std(runs[arm], lambda d: agg_task(d, t, only_claim_cells=True))
            summary[(arm, t)] = (mu, sd)
            mu_all, _ = mean_std(runs[arm], lambda d: agg_task(d, t))
            summary[(arm, t, "all")] = (mu_all, _)
            row = "".join(f"{mu[r]:>9.3f}" for r in hdr_rs)
            print(f"{arm:<16}{t+' ('+TASK_GROUP[t]+')':<22}{row}")
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
        from data import tasks as T
        keep = set(T.claim_cells())
        print(f"{'cell':<18}{'floor':>8}{'best acc':>10}{'r*':>6}  class")
        for k, f, b, sat in rows:
            cls = "must compute" if k in keep else "tabulable"
            print(f"{k:<18}{f:>8.3f}{b:>10.3f}{(str(sat) if sat else '--'):>6}  {cls}")
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

    # ------------------------------------- C3: where did the capability go?
    print("\n" + "=" * 92)
    print("C3  gains live in the recurrence, not the prelude/coda  (paper Table 4)")
    print("=" * 92)
    print("Paper Table 4 reports its recurrent model evaluated at r=1 scoring BELOW the")
    print("fixed-depth twin (ARC-E 34.89 vs 46.42) even though they share an architecture:")
    print("the recurrent model never learned to solve anything in a single pass. If our")
    print("rec@r=1 also sits below fixed1@best, the capability is in the loop, not the ends.")
    if "rec" in runs and "fixed1" in runs:
        print(f"\n{'task':<22}{'rec@r=1':>10}{'fixed1@best':>13}{'delta':>10}{'rec@best':>10}")
        for t in tasks:
            mu_r, _ = summary[("rec", t)]
            r1 = mu_r[min(mu_r)]
            f1 = max(summary[("fixed1", t)][0].values())
            print(f"{t+' ('+TASK_GROUP[t]+')':<22}{r1:>10.3f}{f1:>13.3f}"
                  f"{r1-f1:>+10.3f}{max(mu_r.values()):>10.3f}")

    # -------------------------------------------------------- C4 / C5 ablations
    print("\n" + "=" * 92)
    print("C4/C5  ablations: does the r-curve survive?  (slope = acc@r_max - acc@r=1)")
    print("=" * 92)
    print(f"{'arm':<16}{'claim':<12}" + "".join(f"{t[:6]+' slope':>14}" for t in tasks) +
          f"{'tok_corr':>10}{'ans loss d':>12}")
    for arm in sorted(runs):
        d0 = runs[arm][0]
        slopes = []
        for t in tasks:
            mu, _ = summary[(arm, t)]
            rs = sorted(mu)
            slopes.append(mu[rs[-1]] - mu[rs[0]])
        tc = d0["recurrence_stats"]["token_corr_final"]
        al = d0.get("answer_loss_vs_r", {})
        if al:
            rs_ = sorted(int(r) for r in next(iter(al.values())))
            dl = float(np.mean([al[t][str(rs_[-1])] - al[t][str(rs_[0])] for t in al]))
        else:
            dl = float("nan")
        print(f"{arm:<16}{d0['claim']:<12}" + "".join(f"{s:>+14.3f}" for s in slopes) +
              f"{tc:>10.3f}{dl:>+12.3f}")

    # ------------------------------------------- answer-only loss vs r
    print("\n" + "=" * 92)
    print("Held-out loss on ANSWER TOKENS ONLY vs r  (nats/byte; the full-stream")
    print("loss is dominated by irreducibly random prompt bytes -- see REPORT Sec. 4.1)")
    print("=" * 92)
    any_al = [d for v in runs.values() for d in v if d.get("answer_loss_vs_r")]
    if any_al:
        rs_ = sorted(int(r) for r in next(iter(any_al[0]["answer_loss_vs_r"].values())))
        print(f"{'arm':<16}{'task':<10}" + "".join(f"{'r='+str(r):>9}" for r in rs_))
        for arm in sorted(runs):
            al = runs[arm][0].get("answer_loss_vs_r")
            if not al:
                continue
            for t in tasks:
                print(f"{arm:<16}{t:<10}" + "".join(f"{al[t][str(r)]:>9.3f}" for r in rs_))
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, t in zip(axes, tasks):
            for arm in sorted(runs):
                al = runs[arm][0].get("answer_loss_vs_r")
                if al:
                    ax.plot(rs_, [al[t][str(r)] for r in rs_], marker="o", ms=3, label=arm)
            ax.set_xscale("log", base=2); ax.set_xlabel("recurrence r")
            ax.set_title(f"{t} ({TASK_GROUP[t]})"); ax.grid(alpha=0.25)
        axes[0].set_ylabel("answer-token loss (nats)")
        axes[-1].legend(fontsize=7)
        fig.suptitle("Held-out loss on answer tokens vs test-time recurrence")
        fig.tight_layout(); fig.savefig(f"{args.figures}/answer_loss_vs_r.png", dpi=140)
        print(f"[fig] {args.figures}/answer_loss_vs_r.png")

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
