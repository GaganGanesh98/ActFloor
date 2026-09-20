# Experiment 1 at scale — Qwen2.5-1.5B-Instruct

Run 20 September 2026. Notebook: `exp1_gpu.ipynb` (Run 1 cell, `CFG_1_5B`). Free Colab T4,
~80 minutes. Raw driver output: `results/exp1_gpu_results.json` (25 rows), committed
unmodified in `a59182d` before any analysis.

Calibration `wikitext-2-raw-v1` train slice A (`ncalib = 128`, `seqlen = 512`); evaluation
`wikitext-2-raw-v1` test (`neval = 64`); distillation train slice B (`ndistil = 512`,
`distil_seqlen = 256`). Calibration, distillation and evaluation text are disjoint.
Baseline perplexity **12.3471**. 196 factorable matrices, 1.31 B dense parameters.
Floors swept: {0.25, 0.4}. `B_calib` ∈ {0, 100, 300, 500}.

---

## Result 1 — H5 survives, and the margin has collapsed

Both halves of this need saying, because either alone misleads.

The best FLOP-saving operating point on the whole sweep is greedy allocation at floor 0.4,
`keep = 0.9`:

| method | floor | keep | rho | ppl | Δ ppl | factored / dense |
|---|---|---|---|---|---|---|
| act_aware_greedy | 0.4 | 0.90 | **0.8949** | 15.888 | **+28.68%** | 64 / 132 |

`rho = 0.8949` is a 10.5% FLOP reduction for a model 28.68% worse in perplexity. Against
the pre-registered viability criterion — `rho <= 0.7` at `Δ ppl < 5%` — this point does not
even enter the region of interest, because its `rho` is nowhere near 0.7.

Inside the region that the criterion actually names, `rho <= 0.7`, the best result on the
sweep is:

| method | floor | keep | B_calib | rho | ppl | Δ ppl |
|---|---|---|---|---|---|---|
| act_aware_greedy | 0.25 | 0.50 | 500 | 0.6153 | 44.219 | **+258.13%** |

**+258% against a criterion of <5%: missed by a factor of ~50.** H5 holds at 1.5B.

And yet the margin has moved a great deal in the adversary's favour. At 0.5B the best
thing the adversary could show was roughly +105% perplexity for a 20% FLOP saving
(Exp 1a′, `findings-exp1a-prime-nonuniform.md`). At 1.5B the adversary buys a 10.5% saving
for +28.68%. **Compressibility clearly rose with scale.** The direction of that movement is
the single most important number in this run, and §Limitations explains why it cannot yet
be quantified.

---

## Result 2 — non-uniform allocation earns its keep, and this corrects the 0.5B conclusion

At `keep = 0.9` the uniform arm returns `rho = 1.0000`:

| method | floor | keep | rho | ppl | Δ ppl | factored / dense |
|---|---|---|---|---|---|---|
| act_aware_uniform | — | 0.90 | **1.0000** | 12.347 | +0.00% | 0 / 196 |
| act_aware_greedy | 0.25 | 0.90 | 0.8896 | 41.940 | +239.67% | 59 / 137 |
| act_aware_greedy | 0.4 | 0.90 | **0.8949** | 15.888 | **+28.68%** | 64 / 132 |

Not one of the 196 matrices clears the `r < mn/(m+n)` break-even at a uniform 0.9 keep
fraction, so every matrix falls back to its original dense weight. The uniform arm at this
budget is not a weak compression — it is **no compression at all**, at exactly baseline
perplexity, saving exactly nothing. Greedy at the same nominal budget reaches `rho = 0.8949`
for +28.68%.

The mechanism is that a uniform rule must apply one keep fraction everywhere, and at 0.9
that fraction sits above break-even for every matrix in the model. Greedy is free to push
a minority of matrices far below break-even and leave the rest fully dense — 64 factored,
132 dense at floor 0.4 — which is a shape the uniform arm cannot express at any single
ratio. Larger matrices give the allocator more room to do this: the aspect ratios at 1.5B
(1536 × 8960 in the MLP) put break-even at a rank the allocator can actually exploit,
whereas 0.5B's smaller matrices (896 × 4864) leave less headroom between the floor and
break-even.

**`findings-exp1a-prime-nonuniform.md` drew its framing from a model too small to show
this.** That document concluded non-uniform allocation "genuinely helps, but not enough",
on the evidence of a single floor (0.25) cutting the penalty by about a third at matched
`rho`. At 1.5B the floor is not a detail but the dominant free parameter: at `keep = 0.9`,
floor 0.4 beats floor 0.25 by **8.4×** in Δ ppl (+28.68% vs +239.67%) at essentially the
same `rho` (0.8949 vs 0.8896). The earlier document's own closing list anticipated exactly
this — item 4, "only one floor (0.25) was tested, sweep {0.1, 0.25, 0.4}" — and the sweep
now says the concern was correct. The 1a′ verdict should be read as a statement about
0.5B with one floor, not as a general claim about non-uniform allocation.

---

## Result 3 — distillation is the dominant lever, and it has not converged

Exp 1b, LoRA distillation against the full model, at `keep = 0.75`, floor 0.25,
`rho = 0.8270`:

| B_calib | ppl | Δ ppl |
|---|---|---|
| 0 | 83.082 | +572.89% |
| 100 | 20.692 | +67.59% |
| 300 | 19.225 | +55.71% |
| 500 | 18.494 | **+49.78%** |

The same shape holds at every ratio tested:

| keep | rho | B=0 | B=100 | B=300 | B=500 |
|---|---|---|---|---|---|
| 0.75 | 0.8270 | +572.89% | +67.59% | +55.71% | +49.78% |
| 0.50 | 0.6153 | +7316.19% | +580.47% | +317.27% | +258.13% |
| 0.30 | 0.3766 | +12346.12% | +1074.05% | +585.82% | +445.90% |

Two things follow. First, distillation is worth more than any other lever measured here:
500 steps at `keep = 0.75` removes an order of magnitude of damage, and the entire H5
margin at `rho <= 0.7` rests on a number (+258%) that is itself a post-distillation
figure. Second — and this is a limitation on the result, not a flourish — **the curve is
still descending at the last checkpoint in every row.** From 300 to 500 steps the penalty
falls 55.71 → 49.78 at `keep = 0.75`, 317.27 → 258.13 at `keep = 0.5`, 585.82 → 445.90 at
`keep = 0.3`. Nothing here has flattened.

**These are truncated curves, not converged measurements.** No asymptote can be read off
them, and the reported Δ ppl at `B_calib = 500` is an upper bound on what this adversary
achieves with this lever, not an estimate of it. A run to 2 000 or 5 000 steps is required
before any of these figures can be presented as the adversary's best.

---

## Result 4 — the infeasibility flag fired correctly on live data

One allocation in the sweep is not an operating point:

| method | floor | keep | target rho | achieved rho | % of budget | `budget_feasible` |
|---|---|---|---|---|---|---|
| act_aware_greedy | 0.4 | 0.30 | 0.3766 | 0.5016 | **133.2%** | **False** |

A 0.4 rank floor across all 196 matrices costs more than the `keep = 0.3` budget allows,
so the floor itself is infeasible: every matrix sits at the floor
(`keep_frac` min 0.3984, max 0.3997, mean 0.3994) and the allocator has no freedom left.
The driver reported the overrun rather than silently returning an allocation that does not
meet its budget — the behaviour added in `66cc0d8` — and this is its first firing on real
data rather than in a test.

**This row must be excluded from any `Phi(rho)` plot.** Its `rho = 0.5016` is not a point
on the frontier; it is the cost of an allocation that was asked for and could not be
supplied. Plotting its perplexity (+3491.37%) at `rho = 0.5016` would place a spurious
point in the middle of the region the frontier is meant to describe.

---

## Protocol checks

All four end-to-end checks pass — activation-aware dominates plain SVD at equal rank:

| keep | act-aware ppl | plain SVD ppl |
|---|---|---|
| 0.90 | 12.347 | 12.347 |
| 0.75 | 19.917 | 23 103.470 |
| 0.50 | 141.163 | 157 810.517 |
| 0.30 | 802.482 | 3 254 316.125 |

The `keep = 0.90` row passes vacuously and should not be cited as evidence: at `rho = 1.0`
both arms fall back to the original dense weights, so both are the baseline and the
comparison has no content. The three lower rows are the real check, and the separation is
three to four orders of magnitude — the plain-SVD adversary is not merely weaker, it is
destroyed at every budget that saves any FLOPs at all.

The matrix-level `L` vs `Lᵀ` whitening check also passes at layer 0
(`q_proj`: err 3.16e5 with `L` against 2.27e6 with `Lᵀ` and 2.00e6 for plain SVD;
`o_proj`: 8.93e4 / 1.62e6 / 1.35e6). Whitening on the correct side is worth roughly an
order of magnitude, and whitening on the wrong side is *worse than not whitening at all*.

---

## Limitations

### The scale comparison is confounded, and more badly than first assumed

The 0.5B comparator (+105.1% at `rho = 0.8046`) comes from `exp1a_prime.py`, not from this
notebook. That run used **tinyshakespeare**, `seqlen = 256`, `neval = 12`, and has a
baseline perplexity of **42.87**. This run uses **wikitext-2-raw-v1**, `seqlen = 512`,
`ncalib = 128`, `neval = 64`, baseline **12.35**.

So the corpus, the sequence length, the calibration size, the evaluation size, the script
and the baseline all differ between the two numbers. A Δ ppl percentage computed against a
baseline of 42.87 on Shakespeare and one computed against 12.35 on wikitext are not
commensurable quantities, and the difference between them cannot be attributed to model
scale. **No claim of the form "compressibility rises with scale" is defensible from this
pair of runs.** The statement in Result 1 that compressibility rose is a description of
two incomparable measurements pointing the same way; it is a reason to run the experiment,
not a result.

What settles it is re-running **0.5B under `CFG_1_5B` exactly** — same corpus, same
`seqlen`, same `ncalib`/`neval`, same floors, same `B_calib` grid — so that model size is
the only thing that differs. That run is cheap: 0.5B is the smoke-test model and the whole
sweep is minutes on the same T4. It should be done before this finding is shown to anyone.

For this reason no table in this document places 0.5B and 1.5B numbers side by side. The
0.5B figures appear only in prose, only with their provenance attached.

### The strongest arm was never distilled

Exp 1b distils `cfg.floor` only, which was 0.25. The floor sweep {0.25, 0.4} ran at
`B_calib = 0` alone. So the two strongest levers in the experiment have never been
combined:

* floor 0.4 is **8.4× better** than floor 0.25 at `keep = 0.9` (+28.68% vs +239.67%),
  undistilled;
* distillation is worth roughly an order of magnitude at `keep = 0.75` on floor 0.25.

Nobody has measured floor 0.4 with 500 distillation steps. **This is the single experiment
most likely to falsify H5, and it is missing.** If the two effects compose even weakly, the
best operating point moves substantially toward the criterion. The run is one cell and one
config change — set `cfg.floor = 0.4` and re-run the Exp 1b arm — and it should be the next
thing executed, ahead of 7B.

### The viability criterion is a proxy, and it is doing more work than it can bear

H5 is a claim about economics, and `Δ ppl < 5%` is not an economic quantity. The best point
found here saves 10.5% of inference FLOPs. On a frontier-scale serving bill that is not
negligible — it is the kind of number that funds a team. H5 does not hold because 10.5% is
economically uninteresting.

H5 holds because **+28.68% perplexity is a behavioural signal that any published auditor
detects in a handful of queries.** That is the real argument, and this document cannot make
it, because the quantity it depends on — auditor power at query budget `N`, proposal §7 —
has not been measured. W2 is not built.

Until it is, every H5 verdict in this repository including this one rests on a perplexity
threshold standing proxy for a detection probability, and the mapping between them is
assumed rather than established. The threshold was chosen before any data was seen, which
protects it from being fitted to results, but it does not make it the right threshold. A
reviewer is entitled to ask what `Δ ppl = 28.68%` corresponds to in auditor power at
`N = 100`, and the honest answer today is that nobody in this project knows.

---

## Status of H5 after this run

**H5 holds at 1.5B**, at `rho <= 0.7`, by a factor of ~50 on the pre-registered criterion,
with 500 distillation steps applied and all protocol checks passing.

It holds less comfortably than at 0.5B, on a criterion that is a proxy for the thing that
matters, with the strongest available arm untested and the distillation curve still
falling. None of those four qualifications is rhetorical — each names a specific run that
has not happened.

## What is next, in order of how much it could change the verdict

1. **Distil floor 0.4** (`cfg.floor = 0.4`, `B_calib` grid unchanged). Combines the two
   strongest levers. Most likely single run to falsify H5. Minutes on a T4.
2. **Extend distillation to 2 000+ steps.** The curve has not converged; the current
   numbers are upper bounds on the adversary's damage.
3. **Re-run 0.5B under `CFG_1_5B`.** Removes the confound and makes the scale claim
   sayable. Cheap.
4. **7B (Run 2).** Needs a high-RAM runtime; ~33 GB of factor cache.
5. **Build W2 (auditor power).** Until this exists, H5 is being judged on a proxy.
