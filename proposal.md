# The Effort Gap in Verifiable LLM Inference

**Can black-box behavioural auditing detect a provider that passes cryptographic verification but did not do the work?**

MSc thesis proposal — Gagan Ganesh, SRH Hochschule Berlin (AI/ML, Big Data)
Draft v0.5 — 19 September 2026 — solo project, ~6 months
(v0.2: reframed around the composed adversary after reading Hollow-LLM §IV; see §4.1)
(v0.3: added §2.1 answering the "just substitution" objection; §8 and §10 brought in line)
(v0.4: Exp 1a results folded into H5 and §6.1; Exp 1a′ non-uniform rank allocation added)
(v0.5: Exp 1a′ run and folded in; H5 holds at 0.5B across three constructions)
Target venue class: IEEE S&P / USENIX Security / CCS / NDSS

---

## 1. The problem in one paragraph

Zero-knowledge verification of LLM inference proves that an output is arithmetically
consistent with a committed set of weights under a declared architecture. It does not
prove that the provider spent the compute a model of that size requires. The Hollow-LLM
attack (arXiv:2607.28884, July 2026) demonstrates this concretely. Its §IV gives two
constructions — *depth hollowing* (layers that pass their input through unchanged) and
*width inflation* (a replication matrix `R = 1_m ⊗ I_d` and `blkdiag(M,…,M)` weights that
confine the signal to a small subspace) — which embed a small **inner model** inside a
large **declared architecture** such that
`f_A_out(W_ghost, x) ≡ f_A_in(W_in, x)` exactly. The proof verifies; the provider serves
the inner model.

That exactness is the pivot of this proposal. Because the served function *is* the inner
model, the auditor's task is not to detect an exotic new object: it is to detect that the
endpoint is serving a different, smaller model — which the black-box substitution
literature already does well. **The effort gap therefore does not pose a new detection
problem. It poses a new design problem for the adversary**: verification pins the declared
architecture but leaves the choice of inner model entirely free. This thesis asks how good
that choice can be made, and whether auditing survives the best one.

---

## 2. Why the two literatures do not already cover this

**What a ZK inference proof binds.** The prover publishes a commitment `C = Com(W)` to
weights and a declared architecture `A` (layers `L`, hidden dim `d`, heads `h`, vocab `V`).
For query `x` and output `y` it produces a proof `π` of the statement

```
∃W : Com(W) = C  ∧  y = f_A(W, x)
```

The verifier checks `π` against `C` and `A`. What is bound is an *arithmetic relation*.
What is not bound is that evaluating `f_A(W, ·)` on the provider's hardware cost anything
close to the FLOPs that `params(A)` implies.

**The objection a reviewer will raise first, and the answer.** In most proof systems the
prover's cost scales with circuit size, so a hollow prover still pays full proving cost and
the attack appears to buy nothing. This is true *only when every request is proved*. The
attack has economic force precisely in the deployment model the practical systems use:
selective or sampled verification, where a small fraction of requests carry proofs and the
rest are served bare under the same commitment. IMMACULATE (arXiv:2602.22700) is exactly
this design — selective verification of a subset of requests, under 1% throughput overhead
— and it explicitly audits for substitution, quantisation abuse and token overbilling
against a *committed* model, not for whether the committed model was of the claimed size.
So the hollow provider pays proving cost on the sampled requests, saves serving cost on all
of them, and the sampled proofs verify. **Locating the gap in the sampled-verification
regime is the first framing contribution of this thesis** and it is what joins the two
papers into one problem rather than two.

### 2.1 Why this is not "substitution with a better substitute"

The sharpest objection a reviewer can raise: the composed adversary serves a distilled
model and lies about it, so this is model substitution with a well-chosen substitute, and
substitution auditing is a solved problem — making the contribution an evaluation paper,
not a security paper.

The answer is that the proof changes the client's behaviour, and that is a security
property, not an evaluation detail. Under plain substitution the client has no assurance
at all and knows it; auditing is their only recourse and they are motivated to spend the
query budget. Under verified serving the client holds a proof that verifies, and the
rational response to a verifying proof is to audit *less*. **Verification therefore
suppresses precisely the defence that would have caught the attack.** The attack surface
is created by the assurance mechanism, which is the classical shape of a security result
rather than a benchmarking one.

Two consequences follow, and both belong in the paper. First, the right baseline is not
"can an auditor detect substitution" but "would a client holding a verifying proof have
audited at all, and at what budget" — which makes the query-budget axis load-bearing
rather than incidental. Second, any fix has to come from the verification side (certifying
realised work) or from mandating an audit floor as a condition of relying on a proof.
A better auditor alone does not close it, because the proof is what stopped the auditing.

---

## 3. Threat model

**Parties.** A provider `P` serving an API; a client or regulator `V` with black-box query
access and the ability to verify proofs; a reference model `M_ref` of the declared size,
available to `V` (open-weights setting, or a trusted replica).

**Provider capabilities — the composed adversary.** `P` proceeds in two stages, and the
composition is what makes it hard to catch:

1. **Compression (quality stage).** Starting from a genuine model of the declared size,
   `P` derives the best inner model `M_in = (A_in, W_in)` it can at a target effort ratio
   `ρ`, using activation-aware low-rank truncation, structured pruning, and a bounded
   calibration budget.
2. **Embedding (commitment stage).** `P` embeds `M_in` into the declared architecture by
   Hollow-LLM §IV — depth hollowing, width inflation, or both — producing `W_ghost` with
   `Com(W_ghost) = C` and `f_A_out(W_ghost, ·) ≡ f_A_in(W_in, ·)`.

Neither stage alone is a threat. Compression alone does not satisfy the circuit.
Embedding alone (an off-the-shelf small model as `M_in`) is behaviourally loud and is
caught by published auditors. The composition is the adversary this thesis studies.

**Adversary calibration budget.** `P` is permitted `B_calib ≤ 500` steps of LoRA
distillation on a single GPU after truncation. Pre-training and full-parameter fine-tuning
remain out of scope. The budget exists because a commercial provider would obviously spend
it, and an uncalibrated truncated model is a strawman a reviewer would reject; it is capped
so the thesis stays inside its compute constraint. Results are reported at both
`B_calib = 0` and `B_calib = 300`.

`P` may additionally pad latency, randomise which requests are served hollow, and knows the
audit method (adaptive adversary). `P` may not forge proofs or break the commitment.

**Provider objective.** Minimise serving cost subject to (a) all sampled proofs verifying
and (b) audit detection probability staying below some tolerance.

**Auditor capabilities.** Black-box queries within budget `q`; output text always, top-`k`
log-probabilities where the API exposes them (stated as an assumption per auditor, because
it is the weakest link in several published methods); wall-clock timing; no white-box
access, no telemetry attestation, no trusted hardware.

**Explicitly out of scope.** Training provenance, key management, TEE-based attestation,
network-layer timing attacks, and building a production ZK proving system. Commitments are
simulated (see §6.1).

---

## 4. Formalising "effort"

Let `N_A = params(A)` be the declared parameter count. Define the **effective capacity** of
a deployment as

```
N_eff = min { params(A') : ∃ A', W'  s.t.  f_A'(W', x) = f_A(W, x) for all x ∈ D }
```

over the input domain `D` of interest, and the **effort ratio**

```
ρ = N_eff / N_A ∈ (0, 1]
```

An honest provider has `ρ ≈ 1`. A hollow provider has `ρ ≪ 1`. The effort gap is the
statement that `(C, A, π)` constrains `W` but places no lower bound on `ρ`.

**The detection problem.** Given black-box access to an endpoint claiming `A`, a verifying
proof, and a query budget `q`, decide whether `ρ < τ`.

### 4.1 The adversary's frontier — the central claim of the thesis

An earlier draft framed this as a dichotomy between *exact* and *lossy* hollowing, with a
region where behavioural auditing is information-theoretically blind. Reading Hollow-LLM
§IV shows that framing is wrong in an instructive way, and the correction is the
contribution.

**Why the blind region is empty.** Under the §IV constructions the ghost is exactly
equivalent *to the inner model*, not to an honest model of the declared size. So the
auditor is never asked to distinguish two identical functions. It is asked to distinguish
`M_in` from an honest `M_ref` of the declared size — two genuinely different maps. There is
always signal. The information-theoretic impossibility argument applies only to the
degenerate case `ρ = 1`, where the adversary gains nothing.

**Where the difficulty actually lives.** Verification constrains `A_out`, not `M_in`. The
adversary's entire freedom is the choice of inner model, and its optimal play is to choose
the `M_in` that minimises behavioural divergence from `M_ref` at a given `ρ`. The detection
question is therefore **not** a property of the auditor alone; it is a property of the
achievable frontier

```
    Φ(ρ) = min over M_in with effort ratio ρ  of  D(M_in ‖ M_ref)
```

where `D` is whatever divergence the auditor can estimate from `q` queries. Everything
interesting about the effort gap is the shape of `Φ`.

Three regimes, only one of which is open:

| inner model | `ρ` | status |
|---|---|---|
| honest model of declared size | 1.0 | adversary gains nothing |
| off-the-shelf smaller checkpoint | ≪1 | loud; IRIS reports 0.99 AUROC on family verification. **Closed.** |
| compressed + distilled from the honest model | tunable | `Φ(ρ)` unmeasured. **This is the thesis.** |

**Why this is the contribution.** Two literatures have each assumed the other's problem
away. The cryptographic side shows ghost weights exist but treats the inner model as given.
The auditing side detects substitution but assumes the substitute is an unrelated
checkpoint. Neither has asked what happens when the substitute is *derived from the model
it is impersonating*. Characterising `Φ(ρ)` and measuring auditor power along it answers,
for the first time, whether verifiable serving delivers a compute guarantee or only an
architecture guarantee.

**Both outcomes are publishable, and the negative one is stronger.**
If `Φ(ρ)` stays high — no economically useful inner model is behaviourally close — then
ghost weights exist algebraically but not economically, and the thesis is a correction to
Hollow-LLM: *the attack does not scale to a viable business*. If `Φ(ρ)` collapses — a
distilled inner model at `ρ ≈ 0.4` is indistinguishable within realistic budgets — then
verification plus behavioural auditing jointly fail to certify effort, which has direct
consequences for any compute- or capability-tied regulatory obligation.

**First evidence is already in and it favours the first branch.** Experiment 0
(`findings-exp0-spectral.md`) shows plain low-rank truncation yields model-wide `ρ = 1.000`
on Qwen2.5-0.5B and 1.5B — the weight spectra are too flat for factorisation to save a
single FLOP at 1% reconstruction error. That bounds the *weakest* compression stage only.
Whether activation-aware truncation plus a calibration budget moves `Φ` is the experiment
that decides the thesis.

**A structural note for the defence side.** Width inflation uses `blkdiag(M,…,M)`, which is
*full rank*. The degeneracy lives in the activation geometry, not the weight spectrum, so a
zero-knowledge proof of a rank lower bound on `W` would not bind the attack. Any
circuit-level countermeasure has to certify realised work, not weight structure. Worth one
paragraph in the paper; out of scope to build.

---

## 5. Pre-registered hypotheses

Recorded before any experiment is run. Outcomes are reported against these regardless of
which way they fall.

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

---

## 6. Workstreams

### 6.1 W1 — Hollow-server testbed and commitment simulation

Base models: Qwen2.5-0.5B / 1.5B for the sweep, one 7B confirmation run. Final choice
gated on availability of top-`k` logprobs through the serving path, which the
rank-uniformity auditor needs.

**Two-stage construction, matching the composed adversary in §3.**

*Stage 1 — compression (produces `M_in`).* Activation-aware low-rank truncation, **not
plain SVD**: Experiment 0 ruled Frobenius-optimal truncation out on these models
(`ρ = 1.000` at 1% error) and Experiment 1a showed it understates the adversary by three
orders of magnitude, so plain SVD is a baseline only. Rank must additionally be allocated
**non-uniformly** across matrices — the uniform sweep in Exp 1a is itself a weak-adversary
artefact; per-layer sensitivity allocation is standard in the compression literature and
is the next refinement (Exp 1a′). A protocol check for this stage: activation-aware
truncation must dominate plain SVD at equal rank, and if it does not, the whitening factor
is wrong (`H = L Lᵀ`, whiten with `L`, not `Lᵀ`). The adversary does not care about `‖W − W_r‖_F`; it cares
about `‖(W − W_r)X‖` over the real activation distribution. Collect activation second
moments on a calibration set, whiten, truncate in the whitened basis, un-whiten. Rank `r`
is the continuous knob generating `ρ`. Optionally followed by `B_calib ≤ 500` steps of LoRA
distillation against `M_ref`. Plain SVD is retained as the weak-adversary baseline; the gap
between the two is itself a reportable result.

*Stage 2 — embedding (produces `W_ghost`).* Embed `M_in` into `A_out` by Hollow-LLM §IV:
depth hollowing (identity pass-through residual blocks) and width inflation
(`R = 1_m ⊗ I_d` replication with `blkdiag(M,…,M)` weights). This is what makes the
deployment commitment-consistent. Note that `blkdiag(M,…,M)` is full rank — the degeneracy
is in the activation geometry, not the weight spectrum — so the ghost passes a spectral
inspection of `W`.

Three executors:

- `dense_executor(W_ghost, x)` — runs `A_out` literally. What the ZK circuit would prove.
- `hollow_executor(W_in, x)` — runs the inner model directly. What the provider actually
  serves. Cost ratio `ρ`.
- `honest_executor(W_ref, x)` — the genuine model of the declared size. The auditor's
  reference.

Correctness harness: assert `dense_executor(W_ghost, x) == hollow_executor(W_in, x)` to
float tolerance on every test input. **That assertion is the simulated proof** — it stands
in for the ZK statement without pretending to be one. The auditor's job is then to
distinguish `hollow_executor` from `honest_executor`, which is a real difference, not an
identity.

Commitment simulation: Merkle tree over quantised weight tensors, SHA-256, plus a signed
manifest of `A`. Reuses commitment machinery already built in prior work. No SNARK. This
boundary is written into the proposal deliberately: any drift toward implementing a real
prover is scope failure, not progress.

Instrumentation: FLOP counter, wall-clock TTFT and inter-token latency, peak memory, all
logged per request alongside `ρ`.

### 6.2 W2 — Existing auditors against the hollow server

Implement and run, all black-box, all with reference access to the honest model:

| Auditor | Source | Signal | Assumption |
|---|---|---|---|
| Model Equality Testing | arXiv:2410.20247 (ICLR'25) | MMD two-sample on output distributions | sampled text only |
| Rank-based uniformity | arXiv:2506.06975 (ICLR'26) | rank statistics of reference tokens | top-`k` logprobs |
| IRIS | arXiv:2607.20860 | RNG-response fingerprint, self-calibrating budget | text only |
| KBF | arXiv:2605.29524 | recall pattern near knowledge boundary | text only |

Sweep `ρ` × `q`. Report power at FPR 0.01. Include the fractional-dilution adversary (hollow
on a random `ε` fraction of requests) since IRIS already models routing dilution and the
hollow provider can trivially adopt it.

### 6.3 W3 — Effort-targeted probes

Design principle: probe where capacity is *structurally* required, not where outputs merely
happen to differ.

- **Rank-bound routing probes.** A rank-`r` attention head can maintain at most `r`
  independent routing channels. Construct tasks requiring `k` simultaneous bindings and
  sweep `k` across `r`. Predicted sharp failure at `k > r` in the hollow deployment and not
  in the honest one. This is the most principled probe family and the one with a stated
  mechanism rather than an empirical correlation.
- **Depth-bound composition probes.** `n`-hop compositional recall and multi-digit
  arithmetic carry depth, where serial depth in the contracted network is reduced.
- **Long-range copy / induction probes.** Retrieval at context depth beyond contracted
  routing capacity.
- **Side channels.** TTFT and inter-token latency distributions under controlled load;
  energy and memory measured locally on the testbed only, with external validity stated as
  a limitation rather than claimed.
- **The padding adversary.** Every side-channel result is re-run against a provider that
  pads timing to the honest profile, and the padding cost is reported in latency and in
  retained compute saving. This is the experiment that decides the second half of H4.

Probes are calibrated so the *honest small model passes them*, otherwise the probe measures
model scale in general rather than the hollowing.

### 6.4 W4 — Characterising the detection boundary

Deliverable: a detection-boundary surface over `(ρ, q)` per auditor, and the **invisible
over-claiming region** — the set of `(ρ, q)` where no method exceeds power 0.5 at FPR 0.01.
Stated in economic terms alongside the statistical ones: what serving-cost reduction is
available inside the invisible region.

---

## 7. Evaluation

- **Primary metric: auditor power `1 − β` at fixed FPR `α = 0.01`, as a function of `ρ` and
  query budget `N`.** Perplexity and reconstruction error are *screening* metrics used to
  build the inner model; they are never reported as detectability. No published auditor
  tests perplexity, so reporting `Δ perplexity < 1%` as a success criterion would measure
  compression quality and mislabel it as invisibility.
- Query budgets are fixed in advance and power estimated over repeated draws. Walking `N`
  upward and reporting the first `p < α` is sequential testing without alpha spending; it
  biases queries-to-detection downward and invalidates the p-value. Where an adaptive
  budget is wanted, use an anytime-valid procedure (e-values / test martingales), which is
  the correct tool for budgeted auditing and a small methodological contribution in itself.
- Secondary: queries-to-detection at power 0.8; probe efficiency relative to MET baseline.
- Adversary adaptivity: each auditor re-run against a provider that knows it.
- Reproducibility: fixed seeds, logged configs, released code and weight-construction
  scripts.

## 8. Threats to validity — stated up front

1. **Scale.** Capability cliffs demonstrated on a 1B model may not transfer to the scale at
   which hollow serving is economically attractive. Mitigated by probe calibration, not
   eliminated. This is the limitation a reviewer will press hardest.
2. **Construction breadth.** The embedding stage implements Hollow-LLM §IV (depth
   hollowing, width inflation). The compression stage covers activation-aware low-rank
   truncation and structured pruning with a bounded calibration budget. Other routes to a
   good inner model — quantisation, MoE-style conditional routing, speculative cascades —
   are untested, so `Φ(ρ)` as reported is an upper bound on achievable divergence: the real
   adversary may do better.
5. **"Isn't this just substitution with a better substitute?"** The strongest reviewer
   objection, answered in §2.1. It must be answered in the paper, not left implicit.
6. **Calibration budget is a judgement call.** `B_calib ≤ 500` is chosen for tractability,
   not derived. A provider with real money would spend more. Report `Φ(ρ)` as a function of
   `B_calib` over {0, 100, 300, 500} so the trend is visible and the reader can extrapolate
   rather than take 500 as a claim about the limit.
3. **Simulated commitments.** No claim is made about real proving costs or about any
   specific deployed ZK-LLM system.
4. **Side-channel external validity.** Latency measured against a local testbed is not
   latency measured against a commercial endpoint behind a load balancer.

## 9. Six-month plan

| Weeks | Work | Gate |
|---|---|---|
| 1 | Read Hollow-LLM §IV and 2608.27954 in full; confirm the gap is open; email the Hollow-LLM authors; finalise this proposal and pre-register H1–H5 | Proposal signed off; hypotheses frozen |
| 2 | **Exp 1a ✅ / Exp 1a′ ✅** done 18–19 Sep (0.5B, uniform + non-uniform rank); activation-aware truncation (activation second moments, whiten → truncate → un-whiten), `B_calib = 0`, Qwen2.5-0.5B then 1.5B. Runs on CPU. | `Φ(ρ)` screening curve; does any FLOP-saving `ρ` survive? |
| 3 | **Exp 1b** same pipeline + 300-step LoRA distillation. **Needs a GPU — Colab/Kaggle free tier**, not the local box. | `ρ` at which `Δ` perplexity is negligible, with calibration |
| 4 | **Exp 2** best pipeline from 1b at 7B scale. Runs regardless of 1b's outcome — compressibility rising with scale is the expected result, so "fails at 1.5B, works at 7B" is the interesting finding, not a stopping condition. | Scale dependence of `Φ` |
| 5–11 | W1: testbed, embedding construction (§IV depth hollowing + width inflation), commitment simulation, instrumentation | Ghost/inner equality assertion passes; `ρ` sweep produces a clean quality curve |
| 12–16 | W2: four auditors implemented and swept | Power surfaces for all four; H1 and H2 answered |
| 17–19 | W3: probe families and side channels, including the padding adversary | H3 and H4 answered |
| 20–21 | W4: boundary characterisation; invisible-region map | Headline figure exists |
| 22–24 | Thesis write-up and paper draft | Submittable draft |

**Phase discipline.** Any week spent on proving-system implementation is a scope breach.
Any expansion into general agent security, prompt injection or agent evaluation is a scope
breach. Both are written here so the supervisor can enforce them.

## 10. Risks

- **The Hollow-LLM group publishes the defence first.** Their §VI-B already sketches
  circuit-level proof-of-effort and tighter commitment binding, which suggests a
  cryptographic follow-up in flight. Mitigation: this thesis is deliberately on the
  black-box side they declared out of scope, and early contact may convert the risk into a
  collaboration.
- **H5 confirms early** — no FLOP-saving inner model survives calibration, so `Φ(ρ)` never
  collapses. The thesis becomes a correction to Hollow-LLM (algebraically real, economically
  void) plus the negative-result methodology. Smaller, still publishable. **Decision point
  is week 3–4, at the end of Exp 1b/2**, not week 10 — the point of front-loading the
  compression experiments is to learn this before the testbed is built.
- **Top-`k` logprob assumption fails** for the chosen serving path, disabling the
  rank-uniformity auditor. Mitigation: three of the four auditors need text only.

---

## Appendix A — Citation verification log (18 September 2026)

Every reference below was fetched and read before being cited. Status is explicit because
the proposal's framing rests on two specific claims in two specific papers.

**Construction verified 18 Sep 2026 (second pass).** The §IV constructions were fetched
and read directly, after an earlier pass had read only the abstract and threat model. Both
are confirmed as stated in §1: *Attack A, depth hollowing* — "some layers effectively pass
their inputs through unchanged" while the architecture declares full depth; *Attack B,
width inflation* — "wide layers are arranged so that only a small subspace carries the
actual signal and the remainder remains inactive", built from a replication matrix
`R = 1_m ⊗ I_d_model` and `blkdiag(M,…,M)` weights with selector matrices routing through
the inner model. The exactness constraint `f_A_out(W_ghost, x; s) = y_in` is quoted
verbatim in the paper. §VI-B could not be retrieved in full; the countermeasure summary in
§10 comes from a first-pass summary and is marked **assumed, not verified** until §VI-B is
read directly. Do not cite it as read.

**Confirmed, load-bearing:**

- **arXiv:2607.28884** — *Hollow-LLM Attack: Computationally Trivial Weights in
  Zero-Knowledge Verification of LLM Inference*. Chen Gong, Beijie Liu, Mengyuan Li;
  30 July 2026. The out-of-scope sentence is verbatim, §III-A: *"Deployment-level audits
  such as behavioral checks, reference-based comparisons, or telemetry may still detect
  aggressive over-claiming, but evaluating such detection signals is beyond the scope of
  this paper."* Note also §VI-B proposes circuit-level countermeasures — see §10 Risks.
  **§IV read and verified 18 Sep 2026** (the construction is in §IV, not §III): Attack A
  *depth hollowing* — *"some layers effectively pass their inputs through unchanged"*;
  Attack B *width inflation* — *"wide layers are arranged so that only a small subspace
  carries the actual signal and the remainder remains inactive"*, via a replication matrix
  *"R = 1_m ⊗ I_d_model"* and *"blkdiag(M,...,M)"*. The binding requirement is
  *"f_{A_out}(W_out^ghost, x; s) = y_in"* — exact output equality with the inner model.
  This is inflation of a small model into a large declared architecture, **not** compression
  of a large one, and correcting that misreading is what produced the v0.2 reframing in §4.1.
- **arXiv:2602.22700** — *IMMACULATE: A Practical LLM Auditing Framework via Verifiable
  Computation*. Guo, Qu, Wu, Zhai, Wang, Xu, Liu, Yuan, Song, Zhang; 26 February 2026.
  Selective verification, <1% throughput overhead, audits substitution / quantisation abuse
  / token overbilling. Confirms the sampled-verification regime that §2 builds on.
- **arXiv:2608.27954** — *Not to Break, but to Attest: Adversarial Probes for
  Privacy-Preserving LLM Verification*. Cameron Wilding, Mina Shaker, Fatemeh Ganji;
  28–31 August 2026. **Checked specifically for contribution overlap: it does not eat this
  thesis.** It detects behavioural *drift from an approved baseline* — model identity, not
  computational capacity. It does not cite Hollow-LLM, ghost weights, or compute
  verification. However, its "stress probes" (MLP-targeted and attention-targeted, designed
  to overstimulate specific pathways) are mechanically close to the rank-bound routing
  probes in W3. **A reviewer will ask how W3 differs. The answer must be in the paper:
  their probes need gray-box intermediate access and target drift; W3's are black-box and
  target capacity. Write that distinction explicitly.**

**Confirmed, supporting:**

- arXiv:2410.20247 — Model Equality Testing (ICLR 2025).
- arXiv:2506.06975 — Auditing Black-Box LLM APIs with a Rank-Based Uniformity Test
  (ICLR 2026). Zhu, Ye, Qiu, Zhu, Tan, Mannan, Michala, Popa, Neiswanger.
- arXiv:2607.20860 — IRIS. Yuewei Zhang, Zhi-Hai Zhang, Hanzhang Qin; 23 July 2026.
  0.99 AUROC on family verification; detects 30% routing dilution at 0.85 mean power.
- arXiv:2605.29524 — KBF: Knowledge Boundary as Fingerprint. Read and confirmed to be
  *endpoint identity fingerprinting*, explicitly not capacity or compute provisioning —
  so it belongs in W2 as a baseline, not as prior art on the effort question.

**Exists but not yet read — check in week 1:**

- arXiv:2608.29930 — *Token Counts Are Not Model Lineage: A Frozen-Threshold Holdout Study
  of Black-Box LLM API Fingerprinting*. A negative result in the adjacent space; matters
  for how W2's baselines are framed.
- eprint.iacr.org/2026/1849 — *Private and Verifiable Outsourcing of Open-Weight LLM
  Inference*. Crypto-side; check whether it already binds effort.

**Note on sourcing.** Several search results surfaced via pith.science, a paper-mirror
site. All citations above were verified against arxiv.org directly. Cite arXiv, never the
mirror.
