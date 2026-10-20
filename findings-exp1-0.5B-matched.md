# Experiment 1, Run 4 — Qwen2.5-0.5B under `CFG_1_5B` exactly

Run 20 October 2026. Config: `CFG_0_5B_MATCHED` in `exp1_gpu_src.py` (Run 4 cell),
which is `CFG_1_5B` with `model_id` swapped and nothing else that enters a number.
Laptop CPU (Apple M1, 8 GB), float32, ~85 minutes. Raw driver output:
`results/exp1_0.5B_matched_results.json` (25 rows), committed unmodified in `5da080b`
before any analysis.

This is item 3 of "What is next" in `findings-exp1-scale.md`: re-run 0.5B so that model
size is the only thing that differs from Run 1. Same corpus (`wikitext-2-raw-v1`), same
disjoint calibration / distillation / evaluation slices, `seqlen = 512`, `ncalib = 128`,
`neval = 64`, `ndistil = 512`, `distil_seqlen = 256`, ratios {0.9, 0.75, 0.5, 0.3},
floors {0.25, 0.4}, `B_calib` ∈ {0, 100, 300, 500}, distilled ratios {0.75, 0.5, 0.3}
on floor 0.25.

Baseline perplexity on the identical evaluation set: **18.7371** at 0.5B,
**12.3471** at 1.5B. 168 factorable matrices / 0.36 B dense parameters at 0.5B, against
196 / 1.31 B.

---

## Result 1 — the scale claim does not survive in the region H5 is judged on

`findings-exp1-scale.md` Result 1 reported that "compressibility clearly rose with
scale", and then (correctly) withdrew the claim as unsupportable because the 0.5B
comparator was a different experiment. With the confound removed, the answer is that
it depends on where you look, and in the region the pre-registered criterion names it
goes the *other* way.

Distilled arm (greedy, floor 0.25, `B_calib = 500`) — the arm whose numbers carry H5:

| keep | rho 0.5B | Δ ppl 0.5B | rho 1.5B | Δ ppl 1.5B | better |
|---|---|---|---|---|---|
| 0.75 | 0.8023 | +80.94% | 0.8270 | **+49.78%** | 1.5B |
| 0.50 | 0.6258 | **+189.03%** | 0.6153 | +258.13% | 0.5B |
| 0.30 | 0.3816 | **+331.23%** | 0.3766 | +445.90% | 0.5B |

Inside `rho ≤ 0.7` — the only region where H5 can be falsified — the best distilled
point at 0.5B is **+189% at `rho = 0.6258`**, and at 1.5B it is **+258% at
`rho = 0.6153`**. The smaller model is the more compressible one there.

Both still miss the criterion (`rho ≤ 0.7` at `Δ ppl < 5%`) by a wide margin: ~38× at
0.5B, ~52× at 1.5B. **H5 holds at both scales under matched settings.**

## Result 2 — undistilled, the direction depends on the arm

| method | floor | keep | rho 0.5B | Δ ppl 0.5B | rho 1.5B | Δ ppl 1.5B |
|---|---|---|---|---|---|---|
| act_aware_uniform | — | 0.75 | 0.8997 | +101.63% | 0.8908 | +61.31% |
| act_aware_uniform | — | 0.50 | 0.6357 | +1240.52% | 0.6275 | +1043.29% |
| act_aware_uniform | — | 0.30 | 0.3817 | +10175.29% | 0.3766 | +6399.36% |
| act_aware_greedy | 0.25 | 0.90 | 0.8650 | +124.51% | 0.8896 | +239.67% |
| act_aware_greedy | 0.25 | 0.75 | 0.8023 | +262.72% | 0.8270 | +572.89% |
| act_aware_greedy | 0.25 | 0.50 | 0.6258 | +7616.61% | 0.6153 | +7316.19% |
| act_aware_greedy | 0.4 | 0.90 | 0.8735 | +61.10% | 0.8949 | +28.68% |
| act_aware_greedy | 0.4 | 0.75 | 0.8384 | +105.90% | 0.8410 | +125.63% |
| act_aware_greedy | 0.4 | 0.50 | 0.6347 | +1130.20% | 0.6245 | +2308.16% |

* **Uniform**: 1.5B is better at every ratio. This is the cleanest "compressibility rises
  with scale" signal in the data, and it is on the weakest activation-aware arm.
* **Greedy, floor 0.4**: 1.5B is better only at `keep = 0.9` (+28.68% vs +61.10%) — the
  headline point of `findings-exp1-scale.md` — and worse at 0.75 and 0.5.
* **Greedy, floor 0.25**: 0.5B is better at 0.9 and 0.75, level at 0.5.

So the single most-quoted comparison from Run 1 (+28.68% at `rho ≈ 0.89`) is real, but
it is the one cell of the greedy grid where 1.5B wins, and it is outside `rho ≤ 0.7`.

## Result 3 — distillation is worth more at 1.5B

From `B_calib = 0` to 500 at `keep = 0.75`, floor 0.25: 0.5B goes +262.72% → +80.94%
(3.2× less damage), 1.5B goes +572.89% → +49.78% (11.5×). At `keep = 0.5`: 0.5B 40×,
1.5B 28×. The bigger model starts further behind after truncation at `keep = 0.75` and
recovers more of it. Whether this is a scale effect or a dtype effect (below) cannot be
separated with two runs.

As at 1.5B, **no distillation curve has flattened at 500 steps** (`keep = 0.5`:
+329.65 → +221.99 → +189.03 over 100/300/500). These remain upper bounds on the
adversary's damage, not estimates of it. `CFG_CONVERGE` (Run 3, Job B) is still the run
that answers this.

## Result 4 — the infeasibility flag fired again, at the same cell

Greedy floor 0.4 at `keep = 0.3`: target `rho = 0.3817`, achieved `0.5080`, floor alone
costs 133.1% of the budget, `budget_feasible = False` — the same cell that fired at
1.5B (`0.5016`, 133.2%). Excluded from any `Phi(rho)` plot, as there.

## Protocol checks

All pass. End to end, act-aware beats plain SVD by 3–4 orders of magnitude at every
FLOP-saving budget (`keep = 0.75`: 37.78 vs 82 022.78; 0.5: 251.17 vs 483 222.58; 0.3:
1 925.29 vs 146 992.53); `keep = 0.9` passes vacuously at `rho = 1.0` and is not cited.
Matrix-level, layer 0, rank 448: `q_proj` err 383.5 (`L`) vs 3 045.4 (`Lᵀ`) vs 1 575.1
(plain SVD); `o_proj` 1.51 / 24.50 / 17.38. Wrong-side whitening is again worse than
none.

The run also exercised `save_adapters` end to end: twelve adapters with manifests were
written, and `auditor.py`'s `discover_adapters` rebuilt the rank assignments and
accepted them (the refuse-on-mismatch check passed). That is the code path Run 3
depends on, now tested before spending GPU time on it.

---

## Limitations

* **`rho` is matched by construction, not exactly.** The same keep ratio gives slightly
  different `rho` at the two sizes, because the break-even rank `mn/(m+n)` depends on
  matrix shapes. At `keep = 0.5` the 0.5B point saves slightly *less* (0.6258 vs 0.6153),
  which flatters it a little. The gap in Δ ppl (+189% vs +258%) is much larger than any
  plausible slope of `Phi` over 0.01 of `rho`, but the comparison is not at identical
  `rho` and is not presented as if it were.
* **Weight dtype differs.** Run 1's JSON records `cuda`, `torch.bfloat16`; this run is
  CPU float32 (the notebook has no MPS path, by design). Hessians, Cholesky and SVD are
  fp32/fp64 in both, so the truncation itself is comparable, but perplexity and
  distillation were computed in different weight precisions. This is the residual
  confound. Removing it means re-running one of the two on the other's hardware: 0.5B on
  the same Colab runtime is minutes.
* **One run per cell.** LoRA dropout is unseeded, so `B_calib > 0` rows move by about a
  percent between runs (README). Every difference called out above is far larger than
  that, but none has an error bar.
* **Two points are not a trend.** Even with the confound gone, 0.5B vs 1.5B says nothing
  about 7B. The proposal's expectation that compressibility rises with scale is neither
  confirmed nor refuted by this pair; what is refuted is reading it off Run 1.

## Consequences for the plan

1. `findings-exp1-scale.md` Result 1's "the margin has moved a great deal in the
   adversary's favour" should not be cited. Under matched settings the margin in
   `rho ≤ 0.7` is *wider* at 1.5B (~52×) than at 0.5B (~38×).
2. Run 3 (floor 0.4 distilled; convergence to 3 000 steps) is unchanged in priority: it
   is still the most likely way to falsify H5 at 1.5B. `RUNBOOK_COLAB.md` has the steps.
3. 7B becomes more important, not less: with two sizes pointing in different directions
   depending on the arm, the third point is what decides whether "scale" is a variable
   here at all.
