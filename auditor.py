"""W2 phase 1 — the logprob oracle auditor.

What this measures
------------------
The *oracle* is the best test that any auditor could run given sampled responses
from the target: a Neyman-Pearson likelihood-ratio test between the reference
model and the hollow model, with exact knowledge of both distributions.

No real auditor has that knowledge. It is therefore a **bound**, not an auditor:

    * lower bound on N*  (queries needed to detect)
    * upper bound on detectability at any fixed N

Every row written by this module carries `access_model` and `is_bound` fields
saying so, because someone will read the JSON without the prose.

Why it is built before the pre-registered W2 auditors
-----------------------------------------------------
It calibrates the axis. If the oracle needs thousands of queries then no
text-only auditor (MET, RUT, IRIS, KBF) can do better and W2's answer is already
visible. If it needs ten, the interesting quantity becomes how much each
text-only auditor gives away relative to the bound. See the amendment log in
proposal.md -- this is an addition to the pre-registered plan, scoped before its
results existed.

Protocol constraints inherited from the pre-registration (proposal.md, 66cc0d8)
-------------------------------------------------------------------------------
* alpha = 0.01 (section 7), NOT 0.05.
* Power is estimated over repeated draws at budgets **fixed in advance**.
  Section 7 explicitly forbids walking N upward and reporting the first
  p < alpha: that is sequential testing without alpha spending, it biases
  queries-to-detection downward, and it invalidates the p-value. `N_GRID` below
  is fixed and every budget is evaluated independently.
* Secondary metric: queries-to-detection at power 0.8.
* Perplexity is a screening metric and is never reported here as detectability.

Scope
-----
Measurement only, no training. Runs on Colab beside the experiment notebook;
it will not run on a laptop (needs the calibration pass and two resident
models).

Usage
-----
    import auditor
    auditor.run(auditor.CFG_1_5B, out="results/auditor_oracle.json")
"""

import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# `exp1_gpu_src` is importable as of be43385: its executing cells sit behind a
# __main__ guard. Reuse, never reimplement -- the truncation machinery under
# test must be the same code that produced results/exp1_gpu_results.json.
from exp1_gpu_src import (  # noqa: E402
    Cfg, Cache, CFG_1_5B, DEV, log, sweep,
    make_splits, run_calibration, build_truncated,
    uniform_ranks, greedy_ranks, rho_of, dense_params, adapter_root,
    AutoModelForCausalLM, AutoTokenizer,
)

# ---------------------------------------------------------------------------
# Protocol constants. Fixed in advance; see the module docstring.
# ---------------------------------------------------------------------------

ALPHA = 0.01                      # proposal.md section 7
POWER_TARGET = 0.8                # secondary metric: queries-to-detection
N_GRID = (1, 2, 3, 5, 10, 30, 100, 300, 1000)
N_TRIALS = 2000                   # draws per budget for the power estimate
N_NULL_CALIB = 20000              # null draws used to set the threshold
RESP_TOKENS = 64                  # tokens per simulated API response
SEED = 20260920


@dataclass
class PromptSpec:
    """Pinned evaluation prompts. The oracle and the phase-2 text-only auditors
    MUST run on the same prompts or the comparison is meaningless, so the split,
    the offsets and the seed are recorded in every output row."""
    split: str = "test"
    dataset: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    n_prompts: int = 256
    prompt_tokens: int = 32
    offsets: Tuple[int, ...] = ()     # filled in at build time, then recorded
    seed: int = SEED


# ---------------------------------------------------------------------------
# Model reconstruction -- reuses the experiment's own machinery
# ---------------------------------------------------------------------------

def discover_adapters(root: str, cache_key: str, method: str, ratio: float,
                      floor: Optional[float],
                      ranks: Dict[str, int]) -> Dict[int, str]:
    """Find every saved checkpoint for one arm: {B_calib: path}.

    Each manifest records the rank assignment its adapter was trained on. That
    is checked against the ranks recomputed here and a mismatch is refused, not
    warned about: an adapter applied to a differently-truncated base is silently
    wrong, and silently wrong is the failure mode this whole module exists to
    rule out.
    """
    fl = "none" if floor is None else f"{floor}"
    arm = os.path.join(root, cache_key, f"{method}_keep{ratio}_floor{fl}")
    if not os.path.isdir(arm):
        return {}
    found: Dict[int, str] = {}
    for name in sorted(os.listdir(arm)):
        if not name.startswith("B"):
            continue
        path = os.path.join(arm, name)
        mpath = os.path.join(path, "manifest.json")
        if not os.path.isfile(mpath):
            log(f"  adapter at {path} has no manifest -- skipped")
            continue
        man = json.load(open(mpath))
        if {k: int(v) for k, v in man["ranks"].items()} != {k: int(v) for k, v in ranks.items()}:
            raise RuntimeError(
                f"adapter {path} was trained on a different rank assignment "
                f"than the one rebuilt here -- refusing to use it")
        found[int(man["B_calib"])] = path
    return found


def rebuild_hollow(cfg: Cfg, cache: Cache, shapes, svals,
                   method: str, ratio: float, floor: Optional[float],
                   adapters: Optional[Dict[int, str]] = None,
                   b_calib: int = 0):
    """Rebuild one hollow model at one calibration budget.

    `adapters` maps B_calib -> checkpoint directory, as written by
    save_adapter_checkpoint and located by discover_adapters; each arm now has
    several checkpoints, so the mapping is the natural shape. `b_calib=0` is the
    undistilled base and needs no adapter.

    Before the improvement run saves any, `adapters` is empty and only
    `b_calib=0` is reachable: Run 1's adapters died with the Colab runtime, and
    re-deriving them is training, which this module does not do.
    """
    ranks, alloc_feasible = _ranks_for(shapes, svals, method, ratio, floor)

    model, rho, stats = build_truncated(cfg, cache, shapes, ranks, method, DEV)

    if b_calib:
        if not adapters or b_calib not in adapters:
            raise RuntimeError(
                f"no saved adapter for B_calib={b_calib} on this arm; "
                f"available: {sorted(adapters or {})}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapters[b_calib])
        model = model.merge_and_unload()

    return model, rho, stats, alloc_feasible


# ---------------------------------------------------------------------------
# KL profile on held-out text
# ---------------------------------------------------------------------------

@torch.no_grad()
def kl_profile(ref, hollow, prompts: List[torch.Tensor], device,
               topk: int = 256) -> Dict[str, float]:
    """Per-token KL(ref || hollow) on held-out text, teacher-forced.

    Returns the exact full-vocabulary KL and the top-K coarsened KL, so the
    coarsening gap is measured rather than assumed. The training loop's
    top-K KL is a *lower* bound on the full-vocab KL (log-sum inequality), and
    a lower bound on KL is an optimistic bound on N* -- which is exactly the
    wrong direction to be wrong in. Hence both.
    """
    tot_exact = tot_topk = 0.0
    n_tok = 0
    for ids in prompts:
        x = ids.unsqueeze(0).to(device)
        lp_ref = F.log_softmax(ref(x).logits.float(), dim=-1)
        lp_hol = F.log_softmax(hollow(x).logits.float(), dim=-1)
        p_ref = lp_ref.exp()

        kl = (p_ref * (lp_ref - lp_hol)).sum(-1)          # [1, T]
        tot_exact += float(kl.sum())

        idx = lp_ref.topk(topk, dim=-1).indices
        p_k = p_ref.gather(-1, idx)
        q_k = lp_hol.gather(-1, idx).exp()
        p_tail = (1 - p_k.sum(-1)).clamp_min(1e-12)
        q_tail = (1 - q_k.sum(-1)).clamp_min(1e-12)
        kl_k = (p_k * (p_k.clamp_min(1e-12).log() - q_k.clamp_min(1e-12).log())).sum(-1)
        kl_k = kl_k + p_tail * (p_tail.log() - q_tail.log())
        tot_topk += float(kl_k.sum())

        n_tok += x.shape[1]
        sweep()

    return {"kl_exact_per_token": tot_exact / n_tok,
            "kl_topk_per_token": tot_topk / n_tok,
            "kl_coarsening_gap": (tot_exact - tot_topk) / max(tot_exact, 1e-12),
            "kl_tokens": n_tok}


# ---------------------------------------------------------------------------
# The oracle test
# ---------------------------------------------------------------------------

@torch.no_grad()
def _llr_samples(ref, hollow, prompts, device, n_draws: int,
                 from_hollow: bool, epsilon: float, gen) -> torch.Tensor:
    """Per-response log-likelihood ratios log p_ref(y) - log p_hollow(y).

    `from_hollow=False` draws the null (target is honest); True draws the
    alternative. `epsilon` is the fractional-dilution adversary of section 3:
    the provider serves hollow on an epsilon fraction of requests. Note this is
    the plug-in LLR against a mixture, not the optimal mixture test, so power
    under epsilon < 1 is itself a lower bound.
    """
    out = torch.empty(n_draws)
    for i in range(n_draws):
        prompt = prompts[int(torch.randint(len(prompts), (1,), generator=gen))]
        serve_hollow = from_hollow and (epsilon >= 1.0 or
                                        float(torch.rand(1, generator=gen)) < epsilon)
        src = hollow if serve_hollow else ref
        y = src.generate(prompt.unsqueeze(0).to(device),
                         max_new_tokens=RESP_TOKENS, do_sample=True,
                         top_p=1.0, temperature=1.0, pad_token_id=0)
        resp = y[:, prompt.shape[0]:]
        s = 0.0
        for m, sign in ((ref, 1.0), (hollow, -1.0)):
            lp = F.log_softmax(m(y).logits.float()[:, :-1], dim=-1)
            tgt = y[:, 1:]
            tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            s += sign * float(tok_lp[:, -resp.shape[1]:].sum())
        out[i] = s
        if i % 50 == 0:
            sweep()
    return out


def oracle_power(null_llr: torch.Tensor, alt_llr: torch.Tensor,
                 n_grid=N_GRID, alpha: float = ALPHA,
                 n_trials: int = N_TRIALS, gen=None) -> Dict[int, float]:
    """Power at each FIXED budget N. No sequential peeking: every N is an
    independent experiment with its own threshold, as section 7 requires.

    The threshold at budget N is the (1-alpha) quantile of the null distribution
    of the *summed* LLR over N responses, estimated by resampling the null draws.
    """
    power = {}
    for n in n_grid:
        idx = torch.randint(len(null_llr), (n_trials, n), generator=gen)
        null_sum = null_llr[idx].sum(1)
        tau = torch.quantile(null_sum, 1 - alpha)

        idx = torch.randint(len(alt_llr), (n_trials, n), generator=gen)
        alt_sum = alt_llr[idx].sum(1)
        power[n] = float((alt_sum > tau).float().mean())
    return power


def queries_to_detection(power: Dict[int, float],
                         target: float = POWER_TARGET) -> Optional[int]:
    """Smallest budget on the fixed grid reaching `target` power. None if the
    grid tops out below it -- reported as None rather than extrapolated."""
    for n in sorted(power):
        if power[n] >= target:
            return n
    return None


def n_star_chernoff_stein(kl_per_token: float, resp_tokens: int = RESP_TOKENS,
                          beta: float = 1 - POWER_TARGET) -> float:
    """Asymptotic prediction N* ~ log(1/beta) / KL_per_response.

    This is a *prediction*, reported beside the empirical curve and never in
    place of it. Chernoff-Stein gives an error exponent for i.i.d. samples at
    fixed type-I error; tokens within a response are not independent and the
    auditor observes whole responses. The empirical/predicted ratio is recorded
    as its own quantity -- if it is off by 10x, that gap is a finding about
    applying i.i.d. sample-complexity bounds to autoregressive auditing, not a
    caveat.
    """
    per_response = kl_per_token * resp_tokens
    if per_response <= 0:
        return float("inf")
    return math.log(1.0 / beta) / per_response


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

ACCESS_NOTE = (
    "ORACLE BOUND, NOT AN AUDITOR. Assumes exact knowledge of the full "
    "next-token distributions of BOTH the reference and the hollow model -- "
    "strictly more than any API exposes. Therefore a LOWER bound on N* and an "
    "UPPER bound on detectability. Every real auditor sits at or below this "
    "curve. alpha=0.01 per proposal.md section 7; power at budgets fixed in "
    "advance, no sequential peeking."
)


def build_prompts(cfg: Cfg, spec: PromptSpec, tok) -> Tuple[List[torch.Tensor], PromptSpec]:
    """Pinned prompts from the held-out test split, disjoint from calibration
    and distillation text. Offsets are recorded so phase 2 reuses them exactly."""
    _calib, eval_ids, _distil = make_splits(tok, cfg)
    gen = torch.Generator().manual_seed(spec.seed)
    offsets, prompts = [], []
    for _ in range(spec.n_prompts):
        j = int(torch.randint(len(eval_ids), (1,), generator=gen))
        seq = eval_ids[j]
        o = int(torch.randint(max(1, len(seq) - spec.prompt_tokens), (1,), generator=gen))
        offsets.append([j, o])
        prompts.append(seq[o:o + spec.prompt_tokens])
    return prompts, PromptSpec(**{**asdict(spec), "offsets": tuple(map(tuple, offsets))})


def _ranks_for(shapes, svals, method: str, ratio: float,
               floor: Optional[float]) -> Tuple[Dict[str, int], bool]:
    """The rank assignment for one arm. Same code path as run_model, so the
    adapter manifests written during distillation compare equal to these."""
    if method in ("plain_svd", "act_aware_uniform"):
        return uniform_ranks(shapes, ratio), True
    if method == "act_aware_greedy":
        target = rho_of(uniform_ranks(shapes, ratio), shapes)
        alloc = greedy_ranks(svals, shapes, target * dense_params(shapes), floor=floor)
        return alloc.ranks, alloc.feasible
    raise ValueError(f"unknown method {method!r}")


def _oracle_one(rows: List[Dict], cfg: Cfg, cache: Cache, shapes, svals,
                method: str, ratio: float, floor: Optional[float], b_calib: int,
                adapters: Dict[int, str], ref, prompts, spec,
                epsilons, gen) -> None:
    """One (arm, B_calib) cell: rebuild, profile KL, measure the oracle curve."""
    hollow, rho, _stats, feasible = rebuild_hollow(
        cfg, cache, shapes, svals, method, ratio, floor,
        adapters=adapters, b_calib=b_calib)
    hollow.eval().to(DEV)

    kl = kl_profile(ref, hollow, prompts, DEV, topk=cfg.topk)

    for eps in epsilons:
        null = _llr_samples(ref, hollow, prompts, DEV,
                            N_NULL_CALIB // 100, False, eps, gen)
        alt = _llr_samples(ref, hollow, prompts, DEV,
                           N_TRIALS // 10, True, eps, gen)
        power = oracle_power(null, alt, gen=gen)
        n_emp = queries_to_detection(power)
        n_pred = n_star_chernoff_stein(kl["kl_exact_per_token"])
        rows.append({
            "model": cfg.model_id, "method": method, "floor": floor,
            "keep_ratio": ratio, "rho": rho, "B_calib": b_calib,
            "adapter": adapters.get(b_calib),
            "budget_feasible": feasible,
            "epsilon": eps,
            **kl,
            "power_at_N": power,
            "n_star_empirical": n_emp,
            "n_star_predicted": n_pred,
            "empirical_over_predicted":
                (n_emp / n_pred) if (n_emp and math.isfinite(n_pred)) else None,
            "alpha": ALPHA, "power_target": POWER_TARGET,
            "resp_tokens": RESP_TOKENS,
            "access_model": ACCESS_NOTE,
            "is_bound": True,
            "prompts": asdict(spec),
        })
    del hollow
    sweep()


def run(cfg: Cfg = CFG_1_5B, out: str = "results/auditor_oracle.json",
        epsilons: Tuple[float, ...] = (1.0, 0.1),
        methods: Tuple[str, ...] = ("act_aware_greedy", "act_aware_uniform"),
        floors: Optional[Tuple[float, ...]] = None) -> Dict:
    """Measure the oracle curve for every arm of `cfg`, at B_calib=0 and at
    every checkpoint the improvement run has saved an adapter for.

    The quantity this is building toward is N*(rho, B_calib): how much
    calibration the adversary needs before the auditor stops seeing it.
    """
    gen = torch.Generator().manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    cache = Cache(cfg)
    calib_ids, _eval, _distil = make_splits(tok, cfg)
    man = run_calibration(cfg, cache, calib_ids, DEV)
    shapes, svals = man["shapes"], cache.get("svals")

    prompts, spec = build_prompts(cfg, PromptSpec(), tok)
    ref = AutoModelForCausalLM.from_pretrained(cfg.model_id, torch_dtype=torch.float32)
    ref.eval().to(DEV)

    root = adapter_root(cfg)
    if root is None:
        log("no Drive mounted: only B_calib=0 is reachable this session")

    rows: List[Dict] = []
    floors = floors if floors is not None else getattr(cfg, "floors", (cfg.floor,))
    for method in methods:
        fl_grid = floors if method == "act_aware_greedy" else (None,)
        for floor in fl_grid:
            for ratio in cfg.ratios:
                ranks, _feasible = _ranks_for(shapes, svals, method, ratio, floor)
                adapters = (discover_adapters(root, cache.key, method, ratio,
                                              floor, ranks) if root else {})
                budgets = [0] + sorted(adapters)
                log(f"oracle: {method} floor={floor} keep={ratio} "
                    f"B_calib={budgets}")
                for b in budgets:
                    _oracle_one(rows, cfg, cache, shapes, svals, method, ratio,
                                floor, b, adapters, ref, prompts, spec,
                                epsilons, gen)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    payload = {"_meta": {"alpha": ALPHA, "n_grid": list(N_GRID), "seed": SEED,
                         "access_model": ACCESS_NOTE, "is_bound": True},
               "rows": rows}
    with open(out, "w") as f:
        json.dump(payload, f, indent=1)
    log(f"{len(rows)} oracle rows -> {out}")
    return payload


if __name__ == "__main__":
    run()
