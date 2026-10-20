# W2 at 0.5B — the oracle bound, MET and RUT

Run 20 October 2026, laptop CPU (Apple M1, float32). First measurements of the primary
metric of proposal §7: auditor power at `α = 0.01` against `ρ` and query budget `N`.
Until now every H5 verdict in this repository rested on a perplexity threshold standing
in for detectability (`findings-exp1-scale.md`, §"The viability criterion is a proxy").

Raw outputs, committed unmodified before analysis:

* `results/auditor_oracle_0.5B.json` (`40324ee`) — `auditor.py`, the logprob oracle.
* `results/auditor_met_0.5B.json` (`4c12f6f`) — `auditor_met.py`, MET (arXiv:2410.20247).

Both use the hollow models of Run 4 (`findings-exp1-0.5B-matched.md`): greedy, floor
0.25, `keep` ∈ {0.75, 0.5} (`ρ` = 0.8023, 0.6258), `B_calib` ∈ {0, 100, 300, 500},
rebuilt from the factor cache with the saved adapters (rank assignments checked against
each manifest). Fractional dilution `ε = 1` only. Both see the same 25 pinned wikitext
test prompts (32 tokens) and 64-token responses sampled exactly at temperature 1, so the
oracle is a valid bound on MET here.

---

## Result 1 — the oracle detects every cell from one response

| keep | ρ | B_calib | Δ ppl | KL / token (exact) | KL / response | oracle power at N = 1 |
|---|---|---|---|---|---|---|
| 0.75 | 0.8023 | 0 | +262.72% | 1.2230 | 78.3 nats | 1.000 |
| 0.75 | 0.8023 | 100 | +119.29% | 0.9922 | 63.5 | 1.000 |
| 0.75 | 0.8023 | 300 | +92.54% | 0.8724 | 55.8 | 1.000 |
| 0.75 | 0.8023 | 500 | +80.94% | 0.8636 | 55.3 | 1.000 |
| 0.50 | 0.6258 | 0 | +7616.61% | 4.3308 | 277.2 | 1.000 |
| 0.50 | 0.6258 | 100 | +329.65% | 1.6151 | 103.4 | 1.000 |
| 0.50 | 0.6258 | 300 | +221.99% | 1.4135 | 90.5 | 1.000 |
| 0.50 | 0.6258 | 500 | +189.03% | 1.3840 | 88.6 | 1.000 |

`N* = 1` in every cell: a single 64-token response separates hollow from honest at
`α = 0.01` with power 1. With 55–277 nats of KL per response there is nothing for a
likelihood-ratio test to struggle with. Chernoff–Stein predicts `N* ≈ 0.006–0.03`, i.e.
below one query, which is consistent.

Distillation does lower the KL (78 → 55 nats/response at `keep = 0.75`), but by far
less than it lowers perplexity, and nowhere near the point where it would matter to an
auditor that knows both distributions.

The top-K KL that the distillation loss uses understates the exact KL by 9–12% in every
cell (`kl_coarsening_gap`), in the direction `auditor.py` warned about: a lower bound on
KL is an optimistic bound on `N*`.

## Result 2 — MET needs ~300 queries, and distillation pushes it past that

MET power at fixed budgets (1 000 simulated null / alternative trials each; `N = 1` has no
within-sample pairs and `N = 1000` exceeds the pools, so neither is reported):

| keep | ρ | B_calib | N = 10 | N = 30 | N = 100 | N = 300 | N* (power 0.8) |
|---|---|---|---|---|---|---|---|
| 0.75 | 0.8023 | 0 | 0.008 | 0.008 | 0.034 | 0.746 | > 300 |
| 0.75 | 0.8023 | 100 | 0.008 | 0.008 | 0.031 | 0.635 | > 300 |
| 0.75 | 0.8023 | 300 | 0.001 | 0.012 | 0.037 | 0.549 | > 300 |
| 0.75 | 0.8023 | 500 | 0.007 | 0.005 | 0.026 | **0.321** | > 300 |
| 0.50 | 0.6258 | 0 | 0.255 | 0.842 | 1.000 | 1.000 | 30 |
| 0.50 | 0.6258 | 100 | 0.006 | 0.013 | 0.058 | 0.957 | 300 |
| 0.50 | 0.6258 | 300 | 0.009 | 0.014 | 0.078 | 0.890 | 300 |
| 0.50 | 0.6258 | 500 | 0.017 | 0.009 | 0.092 | 0.963 | 300 |

Three things in this table.

1. **The gap between the bound and a published auditor is 30× to more than 300×.** Every
   cell the oracle catches with one response, MET needs 30 (the undistilled `keep = 0.5`
   model, +7 617% ppl) to more than 300 responses for; every distilled cell needs ≥ 300. The oracle shows the
   information is there; MET's prompt-conditional Hamming kernel does not extract it.
   Completions sampled at temperature 1 diverge within a few tokens, so most of the
   kernel's mass is in the first positions and in shared padding after EOS.
2. **Distillation evades MET, monotonically, at `keep = 0.75`.** Power at `N = 300` falls
   0.746 → 0.635 → 0.549 → 0.321 as `B_calib` goes 0 → 500. The oracle's KL barely moves
   over the same range. Distillation is reshaping exactly what MET measures.
3. **In `ρ ≤ 0.7` distillation stops at the first step.** At `keep = 0.5`, 100 steps
   take MET's `N*` from 30 to 300, and 300–500 more steps do not push it further (0.957 /
   0.890 / 0.963 — differences of the size the pool subsampling produces, not a trend).

## Result 3 — what this does to the H5 argument

`findings-exp1-scale.md` argued that H5 holds not because a 10% FLOP saving is
uninteresting but because "+28.68% perplexity is a behavioural signal that any published
auditor detects in a handful of queries". **For MET at 0.5B that sentence is false.**
Here a model with +81% perplexity and a 20% FLOP saving (`ρ = 0.80`) holds MET to
power 0.32 at 300 queries, and every distilled model, including those with `ρ ≤ 0.7`,
holds it below 0.1 at 100 queries.

What that does and does not mean:

* **It does not falsify H5 as pre-registered.** The `keep = 0.75` arm is at `ρ = 0.80`,
  outside `ρ ≤ 0.7`. Inside it, MET reaches power ≥ 0.89 at `N = 300`. And H5 is about
  *published auditors*, plural; three of the four have not been run.
* **It moves the weight of H5 onto the remaining auditors.** The oracle proves the
  hollow model is trivially distinguishable *in principle*. Whether H5 survives on the
  primary metric now depends on whether a text-only auditor closes most of that 300×
  gap. Rank-based uniformity (arXiv:2506.06975) scores sampled tokens under locally
  computed reference log-probabilities, which is structurally much closer to the oracle
  than an MMD on token identities. **Result 4: at `keep = 0.75` it does** — `N* ≤ 5`
  on every distilled cell.
* **The perplexity proxy is badly calibrated against MET.** +81% Δ ppl corresponds to
  MET power 0.32 at N = 300; +189% to 0.96. The pre-registered `Δ ppl < 5%` threshold
  assumed small perplexity damage is the hard case for auditors. Against MET, much
  larger damage is already hard.

## Result 4 — RUT closes most of the gap (keep 0.75; keep 0.5 pending)

`results/auditor_rut_0.5B.json`, `auditor_rut.py` (arXiv:2506.06975): mean log-rank of the
response under the reference, randomised rank against 100 reference responses per prompt,
Cramér–von Mises test. Same prompts, responses, `α` and budgets as above. The laptop run
was stopped after its four `keep = 0.75` cells; the `keep = 0.5` cells go to Colab
(`RUNBOOK_COLAB.md`), on the same uploaded factors and adapters.

Realised level, held-out reference responses through the identical test: 0.005–0.016 at
every `N` (0.009 at `N = 300`). The test holds its size.

| keep | ρ | B_calib | Δ ppl | N = 1 | N = 2 | N = 3 | N = 5 | N = 10 | N* (power 0.8) |
|---|---|---|---|---|---|---|---|---|---|
| 0.75 | 0.8023 | 0 | +262.72% | 0.879 | 0.943 | 0.959 | 0.998 | 1.000 | **1** |
| 0.75 | 0.8023 | 100 | +119.29% | 0.491 | 0.820 | 0.930 | 0.990 | 1.000 | **2** |
| 0.75 | 0.8023 | 300 | +92.54% | 0.354 | 0.625 | 0.773 | 0.968 | 1.000 | **5** |
| 0.75 | 0.8023 | 500 | +80.94% | 0.278 | 0.602 | 0.739 | 0.934 | 1.000 | **5** |

On the cell where MET reaches only 0.32 at 300 queries (`B_calib = 500`), **RUT reaches
0.93 at 5 queries and 1.0 at 10.** The oracle-to-auditor gap is 1–5×, not ≥ 300×. So for
these models the gap in Result 2 was MET's weakness, not a limit on text-only auditing.
For RUT, "detected in a handful of queries" holds at `ρ = 0.80`.

Distillation still costs the auditor something: RUT's `N*` goes 1 → 2 → 5 → 5 over
`B_calib` 0 → 500. The mean hollow log-rank falls 4.69 → 2.83 against 1.66 for the
reference. That curve is moving toward the reference and has not flattened, which makes
Run 3's 3 000-step convergence arm the experiment that tests whether RUT's margin
survives a stronger adversary.

---

## Bugs found and fixed before these numbers existed

Both were in `auditor.py` as committed in `2663367`, before any oracle result had been
produced, and both would have made the oracle numbers wrong:

* **Wrong rejection tail.** The statistic was `log p_ref − log p_hollow`, which is
  *small* under the alternative, while `oracle_power` rejected on *large* sums. The first
  live cell (KL 1.22 nats/token) reported power 0.0 at every `N`. Fixed by orienting the
  statistic toward the alternative; `_check_orientation` now refuses to report power
  unless the null mean is below zero and the alternative mean above it.
* **Responses not drawn from the scored distribution.** `model.generate` merged
  Qwen2.5-Instruct's `generation_config` (`top_k = 20`, `repetition_penalty = 1.1`),
  which survived the explicit `top_p = 1, temperature = 1`. Responses came from a
  truncated, history-dependent distribution and were scored under the plain softmax, so
  the test was not the likelihood-ratio test it claimed to be. Replaced by an exact,
  seeded ancestral sampler (batched; verified token-identical to the single-draw version
  at `B = 1` and score-identical to the old scoring).

## Limitations

* **One model size, one floor, `ε = 1`.** No dilution adversary yet; IRIS-style routing
  dilution would push MET further out.
* **Pool-subsampled power.** MET draws each trial's samples without replacement from
  1 000 reference / 600 hollow completions, and the oracle bootstraps from 200 + 200
  draws. Both understate trial-to-trial variance. For the oracle this is irrelevant here
  (power saturates at `N = 1`), and `pool_false_positive_at_N` records where its
  large-`N` figures would be pool noise. For MET, budgets where trials would share most
  of the pool are reported as `None` (caps measured on synthetic data, see
  `auditor_met.py`). No error bars are given; differences under ~0.1 at `N = 300` should
  not be read as real.
* **MET deviations from the paper.** Token space instead of character space (the
  tokenizer is shared, which is the paper's own simulation setting), `L = 64` instead of
  50 (so MET and the oracle see the same responses), `α = 0.01` per the
  pre-registration. All listed in the module docstring.
* **The `N` grid stops at 300 for MET.** The pre-registered grid goes to 1 000. Reaching
  it needs a reference pool of ≥ 3 000 completions, ~35 more minutes per arm on this
  machine.

## Next

1. **Finish RUT at `keep = 0.5`** on Colab (`RUNBOOK_COLAB.md`, second runbook): the
   `ρ ≤ 0.7` cells, the ones that bear on H5 as pre-registered.
2. **IRIS and KBF** (arXiv:2607.20860, 2605.29524): the last two pre-registered auditors.
   Neither is implemented; both need their papers read first.
3. **Run 3 on Colab**, then the oracle and MET at 1.5B on its saved adapters, so W2 has
   a second scale. `RUNBOOK_COLAB.md`.
4. `ε < 1` (fractional dilution) for both auditors at 0.5B; it is already supported in
   the code and only costs CPU time.
