"""W2 phase 2, auditor 2 of 4 — Rank-based Uniformity Test (arXiv:2506.06975).

Text-only access to the target, as pre-registered (proposal.md section 6.2, as
amended in A1): the auditor sees sampled responses from the target and computes
reference log-probabilities locally on its own copy of the declared model.

The method, as the paper states it
-----------------------------------
* One target response y_i per query, scored by s_i = f(y_i, x_i), with f the
  **log-rank**: mean over response tokens of log(rank of the token in the
  reference model's next-token distribution).
* m reference responses per prompt, scored the same way, give a randomised
  rank (the randomised quantile residual):
      r_i = (1/m) [ sum_j 1{s_i > s_ij} + U_i * sum_j 1{s_i = s_ij} ],
  U_i ~ Uniform[0, 1]. Under H0 (target = reference) r_i ~ Uniform[0, 1].
* Cramér-von Mises test of uniformity on r_1..r_n:
      w2 = 1/(12n) + sum_i ((2i - 1)/(2n) - r_(i))^2.
* m = 100 reference responses per prompt in the paper.

Deviations, all deliberate
--------------------------
* Temperature 1 and 64-token responses (paper: 0.5 and 30), so RUT sees exactly
  the responses the oracle bounds and MET sees.
* Prompts: the same 25 pinned wikitext prompts as the oracle and MET, drawn
  uniformly with replacement (paper: one query per distinct WildChat prompt).
  A bound only holds on a shared query distribution.
* alpha = 0.01 and the fixed budget grid of auditor.py (pre-registration
  section 7), not the paper's 0.05.
* The CvM null quantile is simulated from exact Uniform(0,1) draws (the paper
  does not say how it computes the null). Because r_i is built from a finite m
  and the target pool is finite, the realised level is ALSO measured, by
  running held-out reference responses through the identical test, and is
  recorded per row as `null_level_at_N`.
* Finite target pool, subsampled without replacement per trial; budgets the
  pool cannot supply with room to spare (3N > pool) are None.
"""

import json
import os
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import torch

from auditor import (ALPHA, N_GRID, POWER_TARGET, RESP_TOKENS, SEED,
                     PromptSpec, build_prompts, rebuild_hollow, _ranks_for,
                     discover_adapters, queries_to_detection,
                     _sample_exact_batch, DRAW_BATCH)
from exp1_gpu_src import (Cfg, Cache, DEV, log, sweep, make_splits,
                          run_calibration, adapter_root,
                          AutoModelForCausalLM, AutoTokenizer)

M_PROMPTS = 25          # same query distribution as auditor_met / oracle
M_REF = 100             # reference responses per prompt (paper)
N_NULL_POOL = 900       # held-out reference responses: realised-level check
N_TGT_POOL = 900        # hollow responses per arm (3N <= pool reaches N=300)
N_SIM = 1000            # trials per budget (power and level)
SCORE_BATCH = 4         # [B, T, V] logits at V = 152k: keep B small on CPU

ACCESS_NOTE = (
    "RANK-BASED UNIFORMITY TEST (arXiv:2506.06975): text-only access to the "
    "target (sampled responses); reference log-ranks computed locally on the "
    "declared model, as the paper and amendment A1 specify. Log-rank score, "
    "randomised rank against m=100 reference responses per prompt, "
    "Cramér-von Mises uniformity test, alpha=0.01, budgets fixed in advance. "
    "See module docstring for deviations from the paper."
)


# ---------------------------------------------------------------------------
# Sampling and scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def log_rank_scores(ref, P: torch.Tensor, toks: torch.Tensor) -> torch.Tensor:
    """Mean log-rank of each response's tokens under `ref`.
    P [B, T] prompts, toks [B, L] responses with -1 after the end."""
    out = torch.empty(P.shape[0])
    T = P.shape[1]
    for a in range(0, P.shape[0], SCORE_BATCH):
        p, t = P[a:a + SCORE_BATCH], toks[a:a + SCORE_BATCH]
        y = torch.cat([p, t.clamp_min(0)], 1).to(DEV)
        logits = ref(y).logits[:, T - 1:-1].float()            # [b, L, V]
        tgt = logits.gather(-1, t.clamp_min(0).to(DEV).unsqueeze(-1))
        rank = (logits > tgt).sum(-1).float() + 1.0             # 1 = argmax
        valid = (t >= 0).to(DEV).float()
        out[a:a + p.shape[0]] = ((rank.log() * valid).sum(1)
                                 / valid.sum(1).clamp_min(1)).cpu()
        del logits
    return out


@torch.no_grad()
def scored_pool(model, ref, prompts: List[torch.Tensor], pid: torch.Tensor,
                gen, tag: str) -> torch.Tensor:
    """Sample one exact response from `model` per entry of `pid` and score it
    by log-rank under `ref`. Returns scores [len(pid)]."""
    P = torch.stack(prompts)
    s = torch.empty(len(pid))
    for a in range(0, len(pid), DRAW_BATCH):
        idx = pid[a:a + DRAW_BATCH]
        toks = _sample_exact_batch(model, P[idx], RESP_TOKENS, DEV, gen)
        s[a:a + len(idx)] = log_rank_scores(ref, P[idx], toks)
        if (a // DRAW_BATCH) % 5 == 0:
            log(f"  {tag} {a + len(idx)}/{len(pid)}")
        sweep()
    return s


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------

def randomised_rank(s: torch.Tensor, pid: torch.Tensor,
                    ref_scores: torch.Tensor, gen) -> torch.Tensor:
    """r_i against the m reference scores of its own prompt.
    s [n], pid [n], ref_scores [M_PROMPTS, m]."""
    R = ref_scores[pid]                                  # [n, m]
    gt = (s[:, None] > R).float().sum(1)
    eq = (s[:, None] == R).float().sum(1)
    U = torch.rand(len(s), generator=gen)
    return (gt + U * eq) / R.shape[1]


def cvm(r: torch.Tensor) -> torch.Tensor:
    """Cramér-von Mises w2 against Uniform(0,1); batched over the first dim."""
    n = r.shape[-1]
    rs = r.sort(-1).values
    i = torch.arange(1, n + 1, dtype=rs.dtype)
    return 1.0 / (12 * n) + (((2 * i - 1) / (2 * n) - rs) ** 2).sum(-1)


def cvm_threshold(n: int, alpha: float, gen, T: int = 20000) -> float:
    """(1 - alpha) quantile of w2 under exact Uniform(0,1) ranks."""
    return float(torch.quantile(cvm(torch.rand(T, n, generator=gen)), 1 - alpha))


def rut_power(s: torch.Tensor, pid: torch.Tensor, ref_scores: torch.Tensor,
              gen, n_grid=N_GRID, alpha: float = ALPHA,
              T: int = N_SIM) -> Dict[int, Optional[float]]:
    """Rejection rate at each fixed budget N, drawing N (response, prompt)
    pairs from the pool without replacement per trial. Each N is its own
    experiment with its own threshold."""
    out: Dict[int, Optional[float]] = {}
    for n in n_grid:
        # Synthetic check: level ~0.01 well inside the pool, but 0.000 at
        # 2N = pool, where every trial is nearly the same subset.
        if 3 * n > len(s):
            out[n] = None
            continue
        tau = cvm_threshold(n, alpha, gen)
        rej = 0
        for _ in range(T):
            k = torch.randperm(len(s), generator=gen)[:n]
            rej += int(cvm(randomised_rank(s[k], pid[k], ref_scores, gen)) > tau)
        out[n] = rej / T
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(cfg: Cfg, out: str = "results/auditor_rut.json",
        methods: Tuple[str, ...] = ("act_aware_greedy",),
        floors: Optional[Tuple[float, ...]] = None) -> Dict:
    gen = torch.Generator().manual_seed(SEED + 2)
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    cache = Cache(cfg)
    calib_ids, _e, _d = make_splits(tok, cfg)
    man = run_calibration(cfg, cache, calib_ids, DEV)
    shapes, svals = man["shapes"], cache.get("svals")
    prompts, spec = build_prompts(cfg, PromptSpec(), tok)
    prompts = prompts[:M_PROMPTS]

    ref = AutoModelForCausalLM.from_pretrained(cfg.model_id, dtype=torch.float32)
    ref.eval().to(DEV)

    log(f"RUT: {M_REF} reference responses x {M_PROMPTS} prompts")
    ref_pid = torch.arange(M_PROMPTS).repeat_interleave(M_REF)
    ref_scores = scored_pool(ref, ref, prompts, ref_pid, gen, "ref").view(M_PROMPTS, M_REF)

    log(f"RUT: {N_NULL_POOL} held-out reference responses (realised level)")
    null_pid = torch.randint(M_PROMPTS, (N_NULL_POOL,), generator=gen)
    null_s = scored_pool(ref, ref, prompts, null_pid, gen, "null")
    level = rut_power(null_s, null_pid, ref_scores, gen)
    log(f"  realised level: {level}")

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
                    log(f"RUT: {method} floor={floor} keep={ratio} B={b} rho={rho:.4f}")
                    t_pid = torch.randint(M_PROMPTS, (N_TGT_POOL,), generator=gen)
                    t_s = scored_pool(hollow, ref, prompts, t_pid, gen, "hollow")
                    del hollow
                    sweep()
                    power = rut_power(t_s, t_pid, ref_scores, gen)
                    log(f"  power={power}")
                    rows.append({
                        "auditor": "RUT", "model": cfg.model_id,
                        "method": method, "floor": floor, "keep_ratio": ratio,
                        "rho": rho, "B_calib": b, "adapter": adapters.get(b),
                        "budget_feasible": feasible, "epsilon": 1.0,
                        "power_at_N": power, "null_level_at_N": level,
                        "n_star_empirical": queries_to_detection(
                            {k: v for k, v in power.items() if v is not None}),
                        "score_mean_hollow": float(t_s.mean()),
                        "score_mean_ref": float(ref_scores.mean()),
                        "alpha": ALPHA, "power_target": POWER_TARGET,
                        "resp_tokens": RESP_TOKENS, "m_prompts": M_PROMPTS,
                        "m_ref": M_REF, "n_tgt_pool": N_TGT_POOL, "n_sim": N_SIM,
                        "access_model": ACCESS_NOTE, "is_bound": False,
                        "prompts": asdict(spec),
                    })
                    _flush(out, rows, level)
    return _flush(out, rows, level)


def _flush(out: str, rows: List[Dict], level) -> Dict:
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    payload = {"_meta": {"auditor": "RUT", "paper": "arXiv:2506.06975",
                         "alpha": ALPHA, "n_grid": list(N_GRID), "seed": SEED + 2,
                         "null_level_at_N": level, "access_model": ACCESS_NOTE},
               "rows": rows}
    with open(out, "w") as f:
        json.dump(payload, f, indent=1)
    return payload
