# Resuming this work

Short answer to "do I need to save the model weights?" — **no.** Nothing in this
project is irreplaceable. Every seed is fixed, the Track-A corpus is generated
deterministically from a seed, and the Track-B base model re-downloads in about
35 seconds. What follows is the exact state and the exact commands.

---

## What is already permanent (in this repo, on `main`)

- The full implementation, comment-railed and quoting the paper section by section.
- `scripts/smoke.py` — 21 paper-rule assertions, all passing.
- `scripts/check_eval.py` — the Phase-4 gate, passing.
- `REPORT.md` — **every measurement taken so far is transcribed into it**, including
  the complete 17-cell Track-A accuracy grid, the contraction statistics, the
  Track-B surgery numbers and the cost model. No result is lost.
- `README.md`, the generated notebook, and this file.

## What was on the Colab box and is gone when that session ends

| artifact | size | how to get it back | cost |
|---|---|---|---|
| `results/rec_s0.json`, `rec_s1.json` | ~14 KB each | rerun the arm | ~31 min each |
| `results/rec_s0.pt`, `rec_s1.pt` | 78.6 MB each | rerun the arm | ~31 min each |
| `results/retrofit_gate.json` | ~7 KB | `scripts/verify_retrofit.py` | ~4 min |
| `data_cache/*.npy` (Track A) | small | regenerated automatically, seeded | seconds |
| `data_cache/retrofit_*.npy` (Track B) | ~20 MB | `scripts/prep_retrofit_data.py` | ~6 min |
| Qwen3-4B-Base weights | 8 GB | re-downloads from the Hub | ~35 s |
| `retro_identity_trainable.pt` | ~118 MB | **never written — the run had not finished** | ~65 min |

Nothing above needs to be carried between sessions. The only reason to keep any
of it is to skip recomputation.

## Where the runs got to

- **Track A**: `rec` finished at both seeds (3000 steps, 24.6M tokens). Results are
  in `REPORT.md` §5.4. The ablation queue (`fixed1`, `fixed1_flop`, `fixedr`,
  `noinject`, `prenorm`, `dets0`, `hi_lr`, `fullbp`) was running and did **not**
  finish.
- **Track B**: the surgery gate passed and its numbers are in `REPORT.md` §5.1.
  The `retro_identity` run reached step 200 of 1000 before the session ended and
  produced one val-loss-vs-r curve, recorded in `REPORT.md` §5.5. It is a
  negative interim result -- loss *rises* with r (2.624 at r=1 to 2.763 at r=16)
  and flattens by r=4 -- but at 200k of 1.0M intended tokens it is not yet a fair
  test. No checkpoint was written. `retro_paper` never started.

`train_scratch.py` and `train_retrofit.py` both skip a run whose result json
already exists, so a partially-completed sweep resumes rather than restarting.

## Picking up, in order

```bash
# on the GPU box
git clone https://github.com/Maverick-Ansh/recurrent-depth-4b && cd recurrent-depth-4b
python scripts/smoke.py && python scripts/check_eval.py     # ~1 min, both must pass

# 1. Track B first -- the headline, and only 20% trained.
python scripts/prep_retrofit_data.py                        # ~6 min
python scripts/verify_retrofit.py                           # ~4 min, must say "surgery is exact"
CUDA_VISIBLE_DEVICES=0 python scripts/train_retrofit.py \
    --adapter-init identity --tag retro_identity \
    --steps 1000 --batch 2 --accum 2 --seq 256 --rbar 4 --k 2 --lr 1e-4 --eval-every 200
#   ~65 min. The number to look at is the val-loss-vs-r curve it prints. At step
#   200 it read 2.624 at r=1 rising to 2.763 at r=16 -- the wrong direction. If
#   loss at r=4 falls below loss at r=1 by step 1000, recurrence was installed;
#   if it still rises, run --adapter-init identity_eps0.05 next to test whether
#   the zero-initialised state coupling is what is holding it in that basin.

# 2. Track A ablations, on the other GPU, in parallel with the above.
CUDA_VISIBLE_DEVICES=1 python scripts/run_sweep.py --steps 3000 --lr 3e-4 --gpus 0
#   ~2.5 h for the remaining arms. rec_s0/rec_s1 will re-run unless their jsons
#   are restored; skip them with --only if you only want the ablations.

# 3. Then the measurements that need a finished model.
python scripts/analyze.py                                   # tables + figures
python scripts/analyze_mechanisms.py --ckpt results/rec_s0.pt   # C6, C7, Sec. 7 plots
python scripts/eval_benchmarks.py --base-only --n 300 --out results/bench_base.json
python scripts/eval_benchmarks.py --ckpt results/retro_identity_trainable.pt \
    --r 1,2,4,8,16 --n 300 --out results/bench_retro_identity.json
```

## If you want the artifacts to survive next time

Results jsons are small and belong in git — they are **not** gitignored, only
`*.pt` and `*.npy` are. Commit them from the box:

```bash
git config user.email "anshvivek2003@gmail.com" && git config user.name "Maverick-Ansh"
git add results/*.json figures/*.png && git commit -m "results" && git push
```

That needs a token on the box. Add one as a Colab secret named `GITHUB_TOKEN`
(Colab secrets require a human click, so it cannot be set from a script), then:

```python
from google.colab import userdata
tok = userdata.get('GITHUB_TOKEN')
!git remote set-url origin https://$tok@github.com/Maverick-Ansh/recurrent-depth-4b.git
```

Do not print the token. For the 118 MB retrofit checkpoint, push it to a Hugging
Face repo rather than git — it is too large for comfortable version control and
is regenerable in an hour anyway.

## The two open questions, stated so they can be picked up cold

1. **Does the identity-initialised retrofit escape the "ignore s" basin?**
   At 200 of 1000 steps the answer is no, and the slope is slightly the wrong way
   (`REPORT.md` §5.5). Four explanations are still open and the report lists what
   separates them: too little training, the zero-initialised basin, LoRA being too
   weak a lever at 0.73% of parameters, or recurrence genuinely not being
   retrofittable this cheaply. Finishing the 1000-step run costs ~50 more minutes
   and settles the first. Running `--adapter-init identity_eps0.05` settles the
   second. The faithful contrast `--adapter-init paper` starts at loss 11.32
   against the base model's 0.71 and never ran.

2. **Would a task the model cannot solve at low depth stop it collapsing to a
   contraction?** Track A's model reaches its fixed point in about five turns
   because everything it can solve is solvable in four. The prediction is that a
   depth-forcing task within its reach would push the fixed point later. The
   cheapest test is to retrain `rec` on `perm` alone at levels {4, 8, 12, 16},
   which concentrates 2.5× more data on the one task family with a genuine depth
   axis. This needs a `--tasks` flag on `train_scratch.py` that does not exist yet.
