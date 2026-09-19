# Experiment 0 — How much free hollowing do real LLM weights allow?

Run 18 September 2026. Scripts: `spectral_scan.py`, `structural_scan.py`.
Models: Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct (bf16 checkpoints, cast to fp32).

## Question

Before building any auditor, bound the adversary. A hollow provider must commit to
weights `W` that satisfy the declared architecture `A_out` while computing inference
more cheaply. In weight space there are exactly two construction families that
actually reduce dense-matmul FLOPs:

1. **Low-rank factorisation** — replace `W (m x n)` by `U_r V_r`. Cost drops from
   `m*n` to `r*(m+n)`, so this saves anything only when `r < m*n/(m+n)`.
   **For a square `d x d` matrix that means discarding more than half the rank.**
2. **Structured removal** — drop whole MLP neurons or whole attention heads.

(Unstructured sparsity is excluded: it does not reduce dense-matmul cost without
hardware support, so it is not a hollowing construction.)

If real trained weights do not admit either at tolerable error, the blind region of
the effort gap is narrow and Hollow-LLM's framing is wider than practice.

## Result 1 — low-rank: no free hollowing at all

`rho` = total factored cost / total dense cost, per the model. Rank chosen as the
smallest `r` meeting the stated relative Frobenius reconstruction error
(Eckart-Young).

**Qwen2.5-0.5B-Instruct** (168 weight matrices)

| max rel. Frobenius error | model-wide rho | matrices where truncation saves any FLOPs |
|---|---|---|
| 0.001 | 1.0000 | 0 / 168 |
| 0.010 | 1.0000 | 0 / 168 |
| 0.050 | 0.9996 | 2 / 168 |
| 0.100 | 0.9985 | 21 / 168 |

**Qwen2.5-1.5B-Instruct** (196 weight matrices)

| max rel. Frobenius error | model-wide rho | matrices where truncation saves any FLOPs |
|---|---|---|
| 0.001 | 1.0000 | 0 / 196 |
| 0.010 | 1.0000 | 0 / 196 |
| 0.050 | 1.0000 | 0 / 196 |
| 0.100 | 0.9997 | 14 / 196 |

Square matrices in 1.5B (q_proj/o_proj), n=56: mean retained rank at 1% error = 0.915 of full rank; break-even needs < 0.500.

**Reading.** At 1% reconstruction error — already far past what leaves output
behaviour intact — *not one matrix in either model* can be factored at a profit.
The spectra are close to flat: square projections still need 91.5% of their rank at
1% error, where break-even demands under 50%. Even at 10% error, which would visibly
degrade the model, the model-wide saving is 0.15% and 0.03%.

## Result 2 — structured removal: ~2%, and not free

Neuron importance for MLP unit `j` taken as `||gate_j|| * ||up_j|| * ||down_:,j||`;
head importance as the Frobenius norm of that head's `o_proj` block.

| model | MLP neurons needed for 99% of importance mass | attention heads needed |
|---|---|---|
| Qwen2.5-0.5B | 0.978 (min 0.971, max 0.984) | 1.000 |
| Qwen2.5-1.5B | 0.976 (min 0.965, max 0.985) | 1.000 |

Best-case structured `rho` ~ 0.98 on the MLP and exactly 1.00 on attention — no head
in either model is droppable at this threshold. And 2% is the *optimistic* figure:
it counts importance mass, not preserved behaviour.

## What this means for the thesis

Naive weight-space hollowing does not work on these models. That is a real finding
and it is the first evidence bearing on **H5** (exact hollowing is largely
unachievable in practice). If it holds up, the thesis shifts from "can auditing see
the hollow provider" to "the cryptographic gap is real but the economically useful
part of it is much narrower than claimed" — a correction to Hollow-LLM rather than
an extension of it, and still publishable.

## What this does NOT show — the decisive follow-up

This is a **lower bound on the adversary**, tested in weight space with no
calibration data. Three things could still open the gap, in descending order of
threat:

1. **Activation-aware truncation.** Plain SVD minimises `||W - W_r||_F`. The
   adversary does not care about the weights; it cares about the outputs, i.e.
   `||(W - W_r)X||` over the real activation distribution. Weighting the truncation
   by activation second moments is strictly stronger and routinely achieves far
   better compression at equal quality. **Until this is run, the result above
   understates the attack.** This is the next experiment and it decides H5.
2. **Hollow-LLM's own constructions.** Their §III characterises weight structures
   compatible with standard transformers. Only two families were tested here; theirs
   may not be either. Read §III in full before treating this as general.
3. **Post-hollowing recovery.** A real adversary would distil or fine-tune after
   truncating, recovering much of the lost quality. Out of scope under the no-training
   rule, so the reported blind region is a floor, not the true one — state this
   explicitly in the write-up rather than letting a reviewer find it.

Scale caveat: both models are small, and compressibility tends to rise with scale.
A 7-8B confirmation run is needed before any general claim.
