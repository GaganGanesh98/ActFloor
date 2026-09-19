# Experiment 1a — activation-aware truncation, B_calib = 0

Run 18 September 2026. Script: `exp1a_actaware.py`. Model: Qwen2.5-0.5B-Instruct (fp32).
Calibration 24 x 256 tokens, evaluation 12 x 256 tokens held out, damping 1e-2.

## Method

Exp 0 established that plain SVD is the wrong adversary: it minimises `||W - W_r||_F`,
weighting every input direction equally. The real adversary minimises output error over
the activation distribution:

```
minimise || (W - W_r) X ||_F     with   H = X X^T = L L^T   (Cholesky)
  =>   W_r = truncate_r( W L ) L^-1
```

Both arms run at identical rank budgets so the comparison isolates the method.
Effort ratio `rho = sum_i min(r_i(m_i+n_i), m_i n_i) / sum_i m_i n_i`; a factorisation
saves FLOPs only when `r < mn/(m+n)`.

## Results

Baseline perplexity 42.87.

| method | rank kept | rho | perplexity | Δ perplexity |
|---|---|---|---|---|
| activation-aware | 0.90 | **1.0000** | 48.53 | +13.2% |
| activation-aware | 0.75 | 0.8997 | 65.04 | +51.7% |
| activation-aware | 0.50 | 0.6357 | 203.21 | +374.0% |
| activation-aware | 0.30 | 0.3817 | 524.49 | +1123.4% |
| plain SVD | 0.90 | 1.0000 | 62 142 | +144 850% |
| plain SVD | 0.75 | 0.8997 | 74 676 | +174 086% |
| plain SVD | 0.50 | 0.6357 | 1 579 299 | +3 683 701% |
| plain SVD | 0.30 | 0.3817 | 1 158 903 | +2 703 104% |

## Two findings

**1. The method matters enormously — three orders of magnitude.** At every rank budget,
activation-aware truncation beats plain SVD by 10^3 or more in perplexity. Exp 0's
conclusion drawn from plain SVD alone would have been an artefact of a weak adversary.
Any future claim about compressibility in this thesis must use the activation-aware arm.

**2. It still does not open the effort gap.** The binding constraint is geometric, not
spectral. At `keep = 0.9` the model is nearly intact (+13% ppl) but `rho = 1.0000` — *zero*
FLOPs saved, because `r = 0.9d` sits above the `mn/(m+n)` break-even everywhere. The first
budget that saves anything at all (`keep = 0.75`) buys a 10% compute reduction for a 1.5x
worse model. By `rho = 0.64` the model is 4.7x worse; by `rho = 0.38`, 12x worse.

Phi(rho) at `B_calib = 0` is therefore already high at the smallest FLOP-saving `rho`.
This is evidence for **H5** (ghost weights exist algebraically, not economically) obtained
against the strong compression method rather than the weak one.

## Corrections made during the run

- v1 OOM'd: activation Hessians for all 168 linear layers held simultaneously
  (`down_proj` alone is 4864^2 fp32 x 24). v2 chunks the decoder and spills to disk.
- **A whitening bug was caught by the result being backwards.** `torch.linalg.cholesky`
  returns lower `L` with `H = L L^T`; the objective is `||(W-W_r) L||_F`, so whitening must
  use `L`, not `L^T`. v1/v2 used `L^T`, which made the activation-aware arm *worse* than
  plain SVD — 14.8M perplexity at `keep=0.75`. Verified the corrected identity on a
  synthetic anisotropic case (output error: act-aware-with-L 475 < act-aware-with-L^T 532 <
  plain SVD 646) before re-running. Worth keeping in the thesis as a sanity-check protocol:
  activation-aware must dominate plain SVD at equal rank, and if it does not, the whitening
  is wrong.

## Caveats — all of which flatter the adversary or need follow-up

1. **Calibration corpus.** HF `wikitext` was unreachable from the container; both
   calibration and evaluation fall back to tinyshakespeare. Calibrating and evaluating on
   the same narrow domain *helps* the adversary, so the failure above is not an artefact of
   a bad calibration set. Re-run on wikitext/C4 before publication.
2. **Uniform rank allocation.** Every matrix is truncated to the same fraction. A real
   adversary allocates rank non-uniformly — more to sensitive layers, less to redundant
   ones — which is standard practice and would improve `Phi(rho)` materially. **This is the
   single biggest remaining understatement of the adversary and should be Exp 1a'.**
3. **`B_calib = 0`.** No LoRA distillation. Exp 1b.
4. **Scale.** 0.5B only; compressibility tends to rise with scale. 1.5B and 7B outstanding.
5. **Perplexity is a screen, not the metric.** Per proposal §7, the reported quantity must
   be auditor power at budget `N`. Perplexity here only decides whether a candidate inner
   model is worth auditing at all.
