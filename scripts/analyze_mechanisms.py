"""Sections 6 and 7 of the paper, measured on the trained Track-A model.

  C6  path independence (Sec. 7) and extrapolation past the training depth
  C7a zero-shot KL adaptive exit (Sec. 6.1, the Fig. 10 histogram)
  C7b zero-shot KV-cache sharing (Sec. 6.2)
  Sec. 7 latent trajectories: distance-to-limit heat map (Fig. 11) and PCA
        projections in which the paper reports orbits, fixed points and sliders
        (Fig. 12).

Everything here is measured against a stated baseline: adaptive exit against the
fixed-r accuracy at the same mean depth, KV sharing against the unshared cache,
path independence against the agreement two *independent* models would show.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import tasks
from recurrent_depth.config import RecurrentDepthConfig
from recurrent_depth.model import RecurrentDepthLM
from recurrent_depth.evaluate import exact_match, _pack
from recurrent_depth.diagnostics import path_independence, recurrence_stats
from recurrent_depth.inference import adaptive_exit_forward, generate


def load_model(ckpt, device="cuda"):
    d = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = RecurrentDepthConfig(**d["cfg"])
    m = RecurrentDepthLM(cfg, init_weights=False).to(device)
    m.load_state_dict(d["state_dict"])
    m.eval()
    return m, cfg


@torch.no_grad()
def adaptive_exit_accuracy(m, items, threshold, r_max, device):
    """Accuracy when every position exits at its own KL-converged step."""
    correct, steps = 0, []
    for b in range(0, len(items), 64):
        ids, mask = _pack(items[b:b + 64])
        ids, mask = ids.to(device), mask.to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            out = adaptive_exit_forward(m, ids[:, :-1], r_max=r_max, threshold=threshold)
        pred = out["logits"].float().argmax(-1)
        tgt, mm = ids[:, 1:], mask[:, 1:]
        correct += int(((pred == tgt) | ~mm).all(dim=1).sum())
        steps.append(out["exit_step"][mm].float().cpu().numpy())
    return correct / len(items), np.concatenate(steps)


@torch.no_grad()
def kv_sharing_accuracy(m, items, r, budget, device, n=64):
    """Strict exact match under real cached decoding with a recurrence KV budget."""
    ok = 0
    for prompt, answer in items[:n]:
        p = torch.tensor([prompt], device=device)
        with torch.autocast("cuda", dtype=torch.float16):
            g = generate(m, p, max_new_tokens=len(answer), r=r, kv_budget=budget,
                         generator=torch.Generator(device="cpu").manual_seed(0))
        ok += int(g["generated"][0].tolist() == answer)
    return ok / min(n, len(items))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/rec_s0.pt")
    ap.add_argument("--out", default="results/mechanisms.json")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.figures, exist_ok=True)
    dev = args.device
    m, cfg = load_model(args.ckpt, dev)
    res = {"ckpt": args.ckpt, "rbar": cfg.mean_recurrence, "k": cfg.backprop_depth}
    print(f"loaded {args.ckpt}  rbar={cfg.mean_recurrence} k={cfg.backprop_depth}", flush=True)

    # --------------------------------------------------- C6 path independence
    print("\n== C6 path independence (Sec. 7) ==")
    pi = {}
    probe = tasks.build_eval_set("perm", 16, n=32, seed=7)
    ids, _ = _pack(probe)
    ids = ids.to(dev)
    for r in [1, 2, 4, 8, 16, 32, 48]:
        with torch.autocast("cuda", dtype=torch.float16):
            pi[r] = path_independence(m, ids[:, :-1], r=r, n_seeds=4, device=dev)
        print(f"  r={r:<3} argmax agreement {pi[r]['argmax_agree']:.4f}   "
              f"final-state cos {pi[r]['state_cos']:+.4f}", flush=True)
    res["path_independence"] = pi

    # ------------------------------------------------- C7a adaptive exit
    print("\n== C7a zero-shot KL adaptive exit (Sec. 6.1) ==")
    exits = {}
    for task, level in [("perm", 16), ("add", (3, 3)), ("recall", 16)]:
        items = tasks.build_eval_set(task, level, n=128, seed=11)
        base_r = 16
        fixed_acc = exact_match(m, items, r=base_r, device=dev, amp_dtype=torch.float16)
        row = {"fixed_r": base_r, "fixed_acc": fixed_acc, "by_threshold": {}}
        for th in [5e-4, 1e-3, 1e-2]:
            acc, steps = adaptive_exit_accuracy(m, items, th, base_r, dev)
            row["by_threshold"][th] = {"acc": acc, "mean_steps": float(steps.mean()),
                                       "p10": float(np.percentile(steps, 10)),
                                       "p90": float(np.percentile(steps, 90))}
            print(f"  {task}/{level}  thr={th:g}: acc {acc:.3f} "
                  f"(fixed r={base_r}: {fixed_acc:.3f})  mean steps {steps.mean():.2f}", flush=True)
            if th == 5e-4:
                row["hist"] = np.bincount(steps.astype(int), minlength=base_r + 1).tolist()
        exits[f"{task}/{level}"] = row
    res["adaptive_exit"] = exits

    # ------------------------------------------------ C7b KV-cache sharing
    print("\n== C7b zero-shot KV-cache sharing (Sec. 6.2) ==")
    share = {}
    items = tasks.build_eval_set("add", (3, 3), n=64, seed=13)
    for budget in [None, 1, 2, 4, 8]:
        a = kv_sharing_accuracy(m, items, r=8, budget=budget, device=dev, n=64)
        share[str(budget)] = a
        print(f"  budget {str(budget):<5} r=8: exact match {a:.3f}", flush=True)
    res["kv_sharing"] = share

    # -------------------------------------------- Sec. 7 latent trajectories
    print("\n== Sec. 7 latent trajectories ==")
    ids2, _ = _pack(tasks.build_eval_set("add", (3, 3), n=4, seed=3))
    ids2 = ids2.to(dev)
    with torch.autocast("cuda", dtype=torch.float16):
        traj = m.trajectory(ids2[:, :-1], r=64).float()          # (65, B, S, h)
    st = recurrence_stats(traj)
    res["trajectory_stats"] = st

    # Fig. 11 analogue: per-token distance to the r=64 limit point
    d = (traj - traj[-1:]).norm(dim=-1)[:, 0].cpu().numpy().T      # (S, 65)
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(np.log10(d + 1e-6), aspect="auto", cmap="magma")
    ax.set_xlabel("recurrence step i"); ax.set_ylabel("token position")
    ax.set_title(r"$\log_{10}\|s_i - s_*\|$ per token (paper Fig. 11 analogue)")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(f"{args.figures}/fig11_distance_to_limit.png", dpi=140)
    print(f"  [fig] {args.figures}/fig11_distance_to_limit.png")

    # Fig. 12 analogue: PCA of all token trajectories, six leading directions
    X = traj[:, 0].reshape(traj.shape[0], -1, traj.shape[-1])       # (65, S, h)
    flat = X.reshape(-1, X.shape[-1]).cpu().numpy()
    flat = flat - flat.mean(0, keepdims=True)
    U, S_, Vt = np.linalg.svd(flat, full_matrices=False)
    proj = (X.cpu().numpy() - X.cpu().numpy().mean((0, 1))) @ Vt[:6].T   # (65, S, 6)
    picks = list(range(0, min(X.shape[1], 6)))
    fig, axes = plt.subplots(len(picks), 3, figsize=(9, 2.6 * len(picks)))
    axes = np.atleast_2d(axes)
    for row, tpos in enumerate(picks):
        for col, (a, b) in enumerate([(0, 1), (2, 3), (4, 5)]):
            ax = axes[row, col]
            xy = proj[:, tpos]
            ax.plot(xy[:, a], xy[:, b], lw=0.8, color="0.7")
            ax.scatter(xy[:, a], xy[:, b], c=np.arange(len(xy)), cmap="viridis", s=8)
            ax.scatter([xy[:, a].mean()], [xy[:, b].mean()], color="red", marker="x")
            ax.set_title(f"tok {tpos}  PC{a+1}-PC{b+1}", fontsize=8)
            ax.tick_params(labelsize=6)
    fig.suptitle("Latent trajectories in the leading PCA directions (paper Fig. 12 analogue)")
    fig.tight_layout(); fig.savefig(f"{args.figures}/fig12_pca_trajectories.png", dpi=140)
    print(f"  [fig] {args.figures}/fig12_pca_trajectories.png")
    res["pca_explained"] = (S_[:6] ** 2 / (S_ ** 2).sum()).tolist()

    json.dump(res, open(args.out, "w"), indent=1)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
