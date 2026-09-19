# Experiment 1a′ — non-uniform rank allocation

Run 19 September 2026. Script: `exp1a_prime.py`. Qwen2.5-0.5B-Instruct, `B_calib = 0`,
same calibration/eval protocol as Exp 1a (tinyshakespeare, 256-token sequences).

## Method

Exp 1a truncated every matrix to the same rank fraction, which is a strawman: a real
adversary spends rank where it buys the most fidelity. Allocation is posed as a budgeted
resource problem. For matrix `i`, with `sigma_ij` the singular values of the **whitened**
matrix `W_i L_i` (so `sigma^2` is output error in the activation metric):

```
error_i(r_i) = sum_{j>r_i} sigma_ij^2        flops_i(r_i) = r_i (m_i + n_i)
minimise  sum_i error_i   s.t.   sum_i flops_i <= B
```

The objective is separable and each rank increment at matrix `i` costs exactly
`(m_i + n_i)` and removes `sigma_ij^2`, so greedy by marginal gain per FLOP
(`sigma_ij^2 / (m_i+n_i)`) is **exact** for this surrogate. No heuristic, no tuning.

## Result 1 — unconstrained greedy is *worse* than uniform, and the reason matters

| target rho | actual rho | ppl | Δ ppl | min keep fraction |
|---|---|---|---|---|
| 0.90 | 0.7910 | 2 995 | +6 887% | **0.00** |
| 0.75 | 0.7046 | 5 249 | +12 144% | 0.00 |
| 0.6357 | 0.6227 | 2 081 | +4 754% | 0.00 |
| 0.50 | 0.4994 | 3 091 | +7 111% | 0.00 |
| 0.3817 | 0.3814 | 3 388 | +7 802% | 0.00 |

Uniformly catastrophic, and non-monotone in `rho` — a signature of a broken objective
rather than a hard frontier. `min keep = 0.00` is the cause: greedy starves whole matrices
to rank zero whenever their layer-local error contribution is small.

**The surrogate is misspecified.** Summed layer-local whitened error is not a proxy for
end-to-end loss, because error compounds multiplicatively through depth: zeroing one
projection destroys the model regardless of how little local error it accounted for. This
is worth a paragraph in the thesis — it is the reason per-layer sensitivity allocation in
the compression literature always carries a floor, and it is a concrete instance of the
general hazard in `Phi(rho)` estimation: an allocation optimal for a tractable surrogate
can be far from optimal for the real objective.

## Result 2 — with a rank floor, non-uniform allocation genuinely helps, but not enough

Floor = 0.25 of `min(m, n)` per matrix, greedy allocating the remaining budget.

| actual rho | ppl | Δ ppl | mean keep fraction |
|---|---|---|---|
| 0.8046 | 87.92 | +105.1% | 0.71 |
| 0.7264 | 116.73 | +172.3% | 0.61 |
| 0.6341 | 145.72 | +239.9% | 0.54 |
| 0.4997 | 272.85 | +536.4% | 0.46 |

Comparison against Exp 1a's uniform allocation at matched `rho`:

| rho | uniform Δ ppl | non-uniform + floor Δ ppl |
|---|---|---|
| ~0.90 | +51.7% | — |
| ~0.80 | — | +105.1% |
| 0.6357 / 0.6341 | +374.0% | **+239.9%** |
| ~0.50 | — | +536.4% |
| 0.3817 | +1123.4% | — |

At the one directly matched point (`rho ≈ 0.635`) non-uniform allocation cuts the
perplexity penalty by about a third, +374% → +240%. The improvement is real and in the
expected direction. It does not come close to changing the verdict.

## Verdict on H5 at 0.5B

The criterion set in advance was a viable frontier at `rho <= 0.7` with `Δ ppl < 5%`. The
best the adversary achieves at `rho = 0.726` is **+172%** — off by a factor of roughly 35.
The nearest thing to a usable operating point, `rho = 0.80` for +105%, buys a 20% compute
reduction for a model twice as bad.

Across three constructions now — plain SVD (Exp 0), activation-aware uniform (Exp 1a), and
activation-aware non-uniform with a floor (Exp 1a′) — no FLOP-saving operating point
survives at tolerable quality on this model. **H5 holds at 0.5B with `B_calib = 0`.**

The binding constraint remains geometric rather than spectral: a factorisation saves
nothing until `r < mn/(m+n)`, and these models cannot be pushed that far intact by any
allocation of rank.

## What is left before H5 can be called

1. **LoRA calibration (Exp 1b).** The only remaining lever of comparable size. Needs a GPU.
2. **Scale (Exp 2).** 1.5B and 7B. Compressibility tends to rise with scale; a negative at
   0.5B does not settle a 7B.
3. **Calibration corpus.** tinyshakespeare throughout, because `wikitext` was unreachable.
   Same-domain calibration and evaluation flatters the adversary, so this does not rescue
   the negative — but re-run on wikitext-2 / C4 before publication.
4. **Floor sensitivity.** Only one floor (0.25) was tested. Sweep {0.1, 0.25, 0.4} so the
   reported `Phi(rho)` is a minimum over floors rather than a single arbitrary choice.
