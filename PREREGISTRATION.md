# Pre-registration record

**Date:** 19 September 2026

This file freezes the pre-registered hypotheses H1–H5 for the MSc thesis in this
repository, and binds them to a specific commit and a specific content hash so
that the hypotheses can be shown to a reviewer to have been fixed *before* the
1.5B and 7B scale runs were executed.

## Bindings

| Item | Value |
|---|---|
| Commit on `main` at pre-registration | `66cc0d818c3e801d1c2f2b1ce396ceaaf4973b82` |
| Short hash | `66cc0d8` |
| SHA-256 of `proposal.md` at that commit | `aee813703cf67c64a9a701d76d9745a4df0535b72a31a614f01dc5a3f9be07df` |
| Remote | `https://github.com/GaganGanesh98/ActFloor.git` |

The hypotheses below are reproduced **verbatim** from `proposal.md`, section 5
("Pre-registered hypotheses"), as that file stands at commit `66cc0d8`. The
SHA-256 above is the hash of the whole file; it can be reproduced with
`git show 66cc0d8:proposal.md | shasum -a 256`.

## Amendment chain for proposal.md

The hypotheses above are frozen. `proposal.md` as a whole is not: it may be corrected where
it states something factually wrong about work by other people. Each such change is logged
in `proposal.md` Appendix B and its effect on the file hash is recorded here, so that a
reader can verify any version and see exactly what moved between them.

| Date | `proposal.md` SHA-256 | Amendments applied |
|---|---|---|
| 2026-09-19 (commit `66cc0d8`) | `aee813703cf67c64a9a701d76d9745a4df0535b72a31a614f01dc5a3f9be07df` | — (pre-registration baseline) |
| 2026-09-20 | `a74137b57e848c9f8af08560969d7ab41fe8366a3a5904443c4ef8a81761bb6c` | A1, A2 |

* **A1** — §6.2: the access assumption for arXiv:2506.06975 corrected from "top-`k`
  logprobs" to "sampled text only (reference logprobs computed locally)". The paper's
  rank-based uniformity test needs only completions from the target. A factual correction
  about someone else's method, not a change to any claim of mine.
* **A2** — W2 scope: a logprob **oracle** (`auditor.py`) added ahead of the four
  pre-registered auditors, as an explicit bound rather than an auditor. Logged at the point
  it was scoped, before any of its results existed.

**Neither amendment touches H1–H5, the viability criterion, or §7's definition of auditor
power at budget `N` (`α = 0.01`, budgets fixed in advance).** Those remain exactly as
verified against `66cc0d8`, and the verbatim text below is unchanged. To check the baseline:

    git show 66cc0d8:proposal.md | shasum -a 256

## Experimental status at commit 66cc0d8

Exp 0 (spectral/structural scan), Exp 1a (uniform activation-aware truncation)
and Exp 1a′ (non-uniform greedy allocation under a global FLOP budget) were
complete at this commit, all at Qwen2.5-0.5B with `B_calib = 0`; the 1.5B and 7B
runs, and Exp 1b (LoRA calibration), had **not** been run.

## Disclosure: window of public visibility

The GitHub repository `GaganGanesh98/ActFloor` was created as **public** and was
publicly readable for approximately six minutes on 19 September 2026 before being
set to private. Precise bounds, and where each timestamp comes from:

| Event | Time (UTC) | Source |
|---|---|---|
| Repository created (public) | 2026-09-19T19:02:41Z | GitHub API `createdAt` |
| Commits pushed — contents first readable | 2026-09-19T19:04:17Z | GitHub API `pushedAt` |
| Visibility set to private | 2026-09-19T19:10:07Z | local clock at the successful `gh repo edit`, verified against GitHub server time to within 1 s |
| Private status confirmed by API read | 2026-09-19T19:10:17Z | GitHub API `visibility: PRIVATE` |

Window of public readability of repository contents: **19:04:17Z – 19:10:07Z, a
little under six minutes.** At the moment of the visibility change GitHub
reported 0 forks, 0 stars, 0 watchers, and its traffic API reported 0 clones and
0 views (total and unique). No exposure was recorded, though GitHub's traffic
data is sampled and lagged and cannot be treated as proof that no third party
fetched the repository during that window.

This disclosure is recorded deliberately: the repository's public history is
independently visible in GitHub's public events feed, and it is better that this
document state the window than that a reader discover it unmentioned.

## Pre-registered hypotheses H1–H5 (verbatim from proposal.md §5)

- **H1 (testbed correctness).** The embedded ghost and the inner model agree to float
  tolerance on every input, and all four auditors score the ghost endpoint identically to
  the bare inner-model endpoint. *(Expected true by construction; a correctness check on
  the embedding, not a finding. If it fails, the §IV construction is mis-implemented.)*
- **H2 (the frontier).** `Φ(ρ)` — minimum achievable behavioural divergence from `M_ref` at
  effort ratio `ρ` — is measurable, monotone decreasing in `ρ`, and bounded away from zero
  for all `ρ` that save meaningful FLOPs. Equivalently: there is a boundary `ρ*(N)` below
  which each auditor's power at `α = 0.01` exceeds 0.5, and it differs materially between
  auditors.
- **H3 (probes).** Capability-cliff probes achieve strictly higher power than
  distribution-matching auditors at equal `N`, because they place probability mass on
  inputs where capacity is load-bearing rather than on typical traffic.
- **H4 (side channels).** Latency side channels detect hollowing at effort ratios where
  output-based methods fail, *and* are defeated by a provider padding to a sampled honest
  profile at a cost that is latency-only, not compute. If both halves hold, the negative
  result extends to side channels. Inter-token variance under padding (`Var(Δt_i)`) is the
  specific statistic to test — padding the mean is cheap, padding the conditional variance
  is not.
- **H5 (the economic-viability claim).** No inner model derived from `M_ref` achieves both
  a FLOP-saving `ρ` and a divergence small enough to evade published auditors within
  realistic budgets — i.e. ghost weights exist algebraically but not economically.
  **Status (18 Sep 2026): two experiments in, both favour H5.**
  *Exp 0* (`findings-exp0-spectral.md`): plain low-rank truncation yields model-wide
  `ρ = 1.000` at 1% reconstruction error on Qwen2.5-0.5B and 1.5B — not one matrix of
  168/196 factors at a profit — and structured neuron/head removal gives at best `ρ ≈ 0.98`.
  *Exp 1a* (`findings-exp1a-actaware.md`, `B_calib = 0`, Qwen2.5-0.5B): activation-aware
  truncation beats plain SVD by three orders of magnitude in perplexity at every rank
  budget, confirming that Exp 0 measured a weak adversary — **but it still does not open
  the gap.** At `keep = 0.9` the model is near-intact (+13% ppl) with `ρ = 1.0000`, zero
  FLOPs saved, because `r = 0.9d` is above the `mn/(m+n)` break-even everywhere. The first
  FLOP-saving budget (`keep = 0.75`, `ρ = 0.90`) costs +52% perplexity; `ρ = 0.64` costs
  +374%; `ρ = 0.38` costs +1123%. The binding constraint is geometric, not spectral.
  *Exp 1a′* (`findings-exp1a-prime-nonuniform.md`, 19 Sep): non-uniform rank allocation
  under a global FLOP budget, solved exactly by greedy on the separable whitened-error
  surrogate. Unconstrained greedy is *worse* than uniform (ppl 2 000–5 000, non-monotone)
  because it starves matrices to rank 0 — the surrogate ignores that error compounds
  through depth. With a 0.25 rank floor it genuinely helps: at matched `ρ ≈ 0.635` the
  penalty falls from +374% to +240%. **It does not change the verdict.** Best operating
  point is `ρ = 0.80` for +105% perplexity — a 20% compute saving for a model twice as
  bad; the pre-set viability criterion (`ρ ≤ 0.7` at `Δppl < 5%`) is missed by ~35×.
  **H5 holds at 0.5B with `B_calib = 0` across three constructions.** Remaining levers,
  both needing a GPU: LoRA calibration (Exp 1b), then scale (Exp 2).

