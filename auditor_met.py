"""W2 phase 2, auditor 1 of 4 — Model Equality Testing (arXiv:2410.20247).

The first pre-registered text-only auditor (proposal.md section 6.2). It sees
sampled completions only, so it sits at or below the oracle curve in
`auditor.py`; the gap between the two is the quantity of interest.

The method, as the paper states it
-----------------------------------
* Prompt-conditional kernel  k(z, z') = 1{x = x'} * k~(y, y'), with the
  Hamming completion kernel  k~(y, y') = sum_{i<=L} 1{y_i = y'_i}.  Shorter
  completions are right-padded with a special token (pad matches pad).
* Statistic: the MMD estimator with within-sample terms averaged over
  N(N-1) pairs and the cross term over N^2 pairs.
* Null by simulation from the reference: draw both samples from the reference,
  compute the statistic, take the (1 - alpha) quantile.
* Prompts: uniform over m prompts (m = 25 on the paper's Wikipedia task),
  sampled at temperature 1.

Deviations, all deliberate
--------------------------
* Token space, not character space. The paper decodes to characters only
  because the endpoints it audits use different tokenizers; reference and
  hollow here share Qwen's tokenizer, which is the paper's own simulation
  setting (section 4.1).
* L = auditor.RESP_TOKENS (64), not 50, so MET and the oracle see responses of
  the same length. Otherwise the oracle would not be a bound on this auditor.
* alpha = 0.01 and the fixed budget grid of auditor.py (pre-registration,
  section 7), not the paper's simulation alpha of 0.05.
* Finite pools. Fresh generations per trial are unaffordable, so each trial
  draws its samples WITHOUT replacement from a pool of exact samples
  (replacement would put duplicate completions into the within-sample term
  and inflate it by L per duplicate). Power is reported only for budgets the
  pools can supply with room to spare (3N <= reference pool, 2N <= target
  pool); larger budgets are None, not extrapolated.
  Subsampling a finite pool understates trial-to-trial variance, so the
  power figures are approximations of the infinite-pool quantity, and the
  pool sizes are recorded in every row.

Same pinned prompts and seed discipline as the oracle: the first `M_PROMPTS`
prompts of auditor.build_prompts(PromptSpec()).
"""

import json
import os
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import torch

import auditor
from auditor import (ALPHA, N_GRID, POWER_TARGET, RESP_TOKENS, SEED,
                     PromptSpec, build_prompts, rebuild_hollow, _ranks_for,
                     discover_adapters, queries_to_detection,
                     _sample_exact_batch, DRAW_BATCH)
from exp1_gpu_src import (Cfg, Cache, DEV, log, sweep, make_splits,
                          run_calibration, adapter_root,
                          AutoModelForCausalLM, AutoTokenizer)

M_PROMPTS = 25          # paper, Wikipedia task
N_REF_POOL = 1000       # reference completions, shared by every arm
N_TGT_POOL = 600        # hollow completions per arm (caps put N=300 in reach)
N_SIM = 1000            # null simulations and alternative trials (paper: 1000)
PAD = -1                # right-padding token; never a real token id

ACCESS_NOTE = (
    "MODEL EQUALITY TESTING (arXiv:2410.20247): text-only auditor. Sees sampled "
    "completions (token ids, shared tokenizer) and nothing else. Prompt-"
    "conditional Hamming-kernel MMD, null simulated from the reference, "
    "alpha=0.01, budgets fixed in advance. Pool-subsampled power; see "
    "module docstring for deviations from the paper."
)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_pool(model, prompts: List[torch.Tensor], n: int, gen,
                tag: str = "") -> Tuple[torch.Tensor, torch.Tensor]:
    """n exact temperature-1 completions, prompts drawn uniformly.
    Returns (prompt_ids [n], completions [n, L] right-padded with PAD)."""
    pid = torch.randint(len(prompts), (n,), generator=gen)
    P = torch.stack(prompts)
    Y = torch.full((n, RESP_TOKENS), PAD, dtype=torch.long)
    for a in range(0, n, DRAW_BATCH):
        b = min(DRAW_BATCH, n - a)
        Y[a:a + b] = _sample_exact_batch(model, P[pid[a:a + b]], RESP_TOKENS, DEV, gen)
        log(f"  {tag} sample {a + b}/{n}")
        sweep()
    return pid, Y


# ---------------------------------------------------------------------------
# Kernel and statistic
# ---------------------------------------------------------------------------

def gram(pid: torch.Tensor, Y: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    """K[i, j] = 1{x_i = x_j} * sum_l 1{y_il = y_jl}, pad matching pad."""
    n = Y.shape[0]
    K = torch.empty(n, n, dtype=torch.float32)
    for a in range(0, n, chunk):
        ham = (Y[a:a + chunk, None, :] == Y[None, :, :]).sum(-1).float()
        same = (pid[a:a + chunk, None] == pid[None, :]).float()
        K[a:a + chunk] = ham * same
    return K


def mmd(K: torch.Tensor, P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """Batched MMD estimate. P, Q: [T, N] index tensors into K, rows disjoint
    within each trial. Within terms over N(N-1) pairs, cross over N^2."""
    N = P.shape[1]
    Kpp = K[P[:, :, None], P[:, None, :]]
    Kqq = K[Q[:, :, None], Q[:, None, :]]
    Kpq = K[P[:, :, None], Q[:, None, :]]
    diag = lambda M: M.diagonal(dim1=1, dim2=2).sum(-1)
    within = (Kpp.sum((1, 2)) - diag(Kpp) + Kqq.sum((1, 2)) - diag(Kqq))
    return within / (N * (N - 1)) - 2 * Kpq.sum((1, 2)) / (N * N)


def _disjoint(pool: torch.Tensor, T: int, sizes: Tuple[int, ...], gen):
    """T trials, each a set of disjoint index draws of the given sizes."""
    outs = [[] for _ in sizes]
    for _ in range(T):
        perm = pool[torch.randperm(len(pool), generator=gen)]
        o = 0
        for k, s in enumerate(sizes):
            outs[k].append(perm[o:o + s])
            o += s
    return [torch.stack(x) for x in outs]


def met_power(K: torch.Tensor, ref_idx: torch.Tensor, tgt_idx: torch.Tensor,
              n_grid=N_GRID, alpha: float = ALPHA, T: int = N_SIM,
              gen=None, batch: int = 50) -> Dict[int, Optional[float]]:
    """Power at each fixed budget N. Null: two disjoint reference samples.
    Alternative: one reference sample vs one target sample. Every N is an
    independent experiment with its own threshold (no sequential peeking)."""
    power: Dict[int, Optional[float]] = {}
    for n in n_grid:
        # N=1 has no within-sample pairs. Beyond the caps, trials share most of
        # their completions, the null collapses toward one fixed split, and
        # the test turns conservative (measured on synthetic data: level 0.000
        # at 2N = ref pool, 0.003 at 3N = ref pool, ~0.01 well inside).
        if n < 2 or 3 * n > len(ref_idx) or 2 * n > len(tgt_idx):
            power[n] = None
            continue
        null, alt = [], []
        for b in range(0, T, batch):
            t = min(batch, T - b)
            P, Q = _disjoint(ref_idx, t, (n, n), gen)
            null.append(mmd(K, P, Q))
            (P,) = _disjoint(ref_idx, t, (n,), gen)
            (Q,) = _disjoint(tgt_idx, t, (n,), gen)
            alt.append(mmd(K, P, Q))
        null, alt = torch.cat(null), torch.cat(alt)
        tau = torch.quantile(null, 1 - alpha)
        power[n] = float((alt > tau).float().mean())
    return power


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(cfg: Cfg, out: str = "results/auditor_met.json",
        epsilons: Tuple[float, ...] = (1.0,),
        methods: Tuple[str, ...] = ("act_aware_greedy",),
        floors: Optional[Tuple[float, ...]] = None) -> Dict:
    """MET power for every (arm, B_calib) of `cfg` that the oracle also covers.
    `epsilon` < 1 is the fractional-dilution adversary: each target response is
    hollow with probability epsilon, honest otherwise."""
    gen = torch.Generator().manual_seed(SEED + 1)
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    cache = Cache(cfg)
    calib_ids, _e, _d = make_splits(tok, cfg)
    man = run_calibration(cfg, cache, calib_ids, DEV)
    shapes, svals = man["shapes"], cache.get("svals")

    prompts, spec = build_prompts(cfg, PromptSpec(), tok)
    prompts = prompts[:M_PROMPTS]

    ref = AutoModelForCausalLM.from_pretrained(cfg.model_id, dtype=torch.float32)
    ref.eval().to(DEV)
    log(f"MET: reference pool of {N_REF_POOL}")
    ref_pid, ref_Y = sample_pool(ref, prompts, N_REF_POOL, gen, "ref")

    root = adapter_root(cfg)
    rows: List[Dict] = []
    floors = floors if floors is not None else (cfg.floors or (cfg.floor,))
    for method in methods:
        for floor in (floors if method == "act_aware_greedy" else (None,)):
            for ratio in cfg.ratios:
                ranks, _f = _ranks_for(shapes, svals, method, ratio, floor)
                adapters = (discover_adapters(root, cache.key, method, ratio,
                                              floor, ranks) if root else {})
                for b in [0] + sorted(adapters):
                    hollow, rho, _s, feasible = rebuild_hollow(
                        cfg, cache, shapes, svals, method, ratio, floor,
                        adapters=adapters, b_calib=b)
                    hollow.eval().to(DEV)
                    log(f"MET: {method} floor={floor} keep={ratio} B={b} rho={rho:.4f}")
                    h_pid, h_Y = sample_pool(hollow, prompts, N_TGT_POOL, gen, "hollow")
                    del hollow
                    sweep()
                    for eps in epsilons:
                        # Dilution (eps < 1): each target completion is hollow
                        # with prob eps, else honest. The honest ones are taken
                        # from the last N_TGT_POOL reference completions, which
                        # are then withheld from the reference side so no
                        # completion appears on both sides of a trial.
                        if eps < 1.0:
                            mix = torch.rand(N_TGT_POOL, generator=gen) < eps
                            r_take = torch.arange(N_REF_POOL - N_TGT_POOL, N_REF_POOL)
                            t_pid = torch.where(mix, h_pid, ref_pid[r_take])
                            t_Y = torch.where(mix[:, None], h_Y, ref_Y[r_take])
                            ref_use = torch.arange(0, N_REF_POOL - N_TGT_POOL)
                        else:
                            t_pid, t_Y = h_pid, h_Y
                            ref_use = torch.arange(N_REF_POOL)
                        K = gram(torch.cat([ref_pid, t_pid]), torch.cat([ref_Y, t_Y]))
                        tgt_idx = torch.arange(N_REF_POOL, N_REF_POOL + N_TGT_POOL)
                        power = met_power(K, ref_use, tgt_idx, gen=gen)
                        rows.append({
                            "auditor": "MET", "model": cfg.model_id,
                            "method": method, "floor": floor,
                            "keep_ratio": ratio, "rho": rho, "B_calib": b,
                            "adapter": adapters.get(b), "budget_feasible": feasible,
                            "epsilon": eps, "power_at_N": power,
                            "n_star_empirical": queries_to_detection(
                                {k: v for k, v in power.items() if v is not None}),
                            "alpha": ALPHA, "power_target": POWER_TARGET,
                            "resp_tokens": RESP_TOKENS, "m_prompts": M_PROMPTS,
                            "n_ref_pool": len(ref_use), "n_tgt_pool": N_TGT_POOL,
                            "n_sim": N_SIM, "access_model": ACCESS_NOTE,
                            "is_bound": False, "prompts": asdict(spec),
                        })
                        log(f"  eps={eps} power={power}")
                    _flush(out, rows)
    return _flush(out, rows)


def _flush(out: str, rows: List[Dict]) -> Dict:
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    payload = {"_meta": {"auditor": "MET", "paper": "arXiv:2410.20247",
                         "alpha": ALPHA, "n_grid": list(N_GRID), "seed": SEED + 1,
                         "access_model": ACCESS_NOTE},
               "rows": rows}
    with open(out, "w") as f:
        json.dump(payload, f, indent=1)
    return payload
