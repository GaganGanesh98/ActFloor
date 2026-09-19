"""
Exp 1a' — non-uniform rank allocation under a global FLOP budget.

Exp 1a truncated every matrix to the same rank fraction. That is a strawman: a real
adversary spends its rank budget where it buys the most output fidelity.

Formulation. For matrix i, let sigma_{i,1..k} be the singular values of the WHITENED
matrix W_i L_i (so sigma^2 is output error in the activation metric, per Eckart-Young).
Truncating to rank r_i costs

    error_i(r_i)  = sum_{j>r_i} sigma_{ij}^2          (output error)
    flops_i(r_i)  = r_i (m_i + n_i)                    (factored cost)

Minimise sum_i error_i subject to sum_i flops_i <= B. The objective is separable and
each rank increment at matrix i costs exactly (m_i + n_i) and removes sigma_{ij}^2 of
error, so the greedy solution is EXACT: sort all (i,j) increments by

    marginal gain per FLOP  =  sigma_{ij}^2 / (m_i + n_i)

descending, and buy them until the budget runs out. No heuristic, no tuning.

Caveat worth stating in the write-up: this minimises the SUM OF LAYER-LOCAL whitened
errors, not end-to-end perplexity. Errors compound through depth, so the allocation is
optimal for a surrogate, not for the true objective. It is still strictly stronger than
uniform, which is the point of the experiment.
"""
import argparse, gc, json, math, os, time
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)
T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

def get_ids(tok, n, seed):
    import urllib.request
    txt = urllib.request.urlopen(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    ).read().decode()
    ids = tok(txt, return_tensors="pt").input_ids[0]
    g = torch.Generator().manual_seed(seed)
    s = torch.randint(0, max(1, len(ids)-n-1), (1,), generator=g).item()
    return ids[s:s+n]

@torch.no_grad()
def perplexity(model, ids, seqlen, nseq):
    nll = ntok = 0
    for i in range(nseq):
        c = ids[i*seqlen:(i+1)*seqlen].unsqueeze(0)
        if c.shape[1] < seqlen: break
        o = model(c, labels=c)
        nll += o.loss.item()*(c.shape[1]-1); ntok += c.shape[1]-1
    return math.exp(nll/ntok)

def allocate(specs, budget, floor=0.0):
    """Greedy allocation on the separable surrogate, with a per-matrix rank FLOOR.

    Without a floor the greedy starves whole matrices to rank 0. That is catastrophic
    end-to-end even when the layer-local error it removes is small, because errors
    compound multiplicatively through depth. Any real adversary imposes a floor, so
    running without one is a different strawman, not a stronger adversary."""
    rank = {k: max(1, int(round(floor*min(sp["m"], sp["n"])))) for k, sp in specs.items()}
    spent = sum(rank[k]*(specs[k]["m"]+specs[k]["n"]) for k in specs)
    cand = []
    for k, sp in specs.items():
        w = sp["m"] + sp["n"]
        for j in range(rank[k], len(sp["sv2"])):
            cand.append((sp["sv2"][j].item()/w, k, j))
    cand.sort(reverse=True)
    for gain, k, j in cand:
        w = specs[k]["m"] + specs[k]["n"]
        if spent + w > budget: continue
        rank[k] += 1; spent += w
    return rank, spent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--seqlen", type=int, default=256)
    ap.add_argument("--neval", type=int, default=12)
    ap.add_argument("--hdir", default="/home/claude/effortgap/hess")
    ap.add_argument("--svddir", default="/home/claude/effortgap/wsvd")
    ap.add_argument("--rhos", default="0.9,0.75,0.6357,0.5,0.3817")
    ap.add_argument("--floor", type=float, default=0.0)
    ap.add_argument("--out", default="results/exp1a_prime.json")
    a = ap.parse_args()
    os.makedirs(a.svddir, exist_ok=True); os.makedirs("results", exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    ev = get_ids(tok, a.seqlen*(a.neval+2), 1)

    lins = {}
    for i, layer in enumerate(model.model.layers):
        for n, m in layer.named_modules():
            if isinstance(m, nn.Linear): lins[f"{i}.{n}"] = m
    orig = {k: m.weight.data.clone() for k, m in lins.items()}
    dense = sum(w.shape[0]*w.shape[1] for w in orig.values())

    # ---- whitened SVD once per matrix, cached to disk -----------------------
    specs = {}
    for k, m in lins.items():
        f = os.path.join(a.svddir, k.replace(".", "_")+".pt")
        W = orig[k]; o_d, i_d = W.shape
        if not os.path.exists(f):
            Hc = torch.load(os.path.join(a.hdir, k.replace(".", "_")+".pt"))
            U, S, Vh = torch.linalg.svd(W @ Hc, full_matrices=False)
            torch.save({"U": U, "S": S, "Vh": Vh, "Hc": Hc}, f)
            del U, S, Vh, Hc; gc.collect()
        S = torch.load(f)["S"]
        specs[k] = {"sv2": (S.double()**2), "m": o_d, "n": i_d}
    log(f"whitened SVDs ready for {len(specs)} matrices")

    base = perplexity(model, ev, a.seqlen, a.neval)
    log(f"baseline ppl = {base:.4f}")

    rows = []
    for target in [float(x) for x in a.rhos.split(",")]:
        rank, spent = allocate(specs, target*dense, floor=a.floor)
        cost = 0
        for k, m in lins.items():
            d = torch.load(os.path.join(a.svddir, k.replace(".", "_")+".pt"))
            o_d, i_d = orig[k].shape
            r = max(1, rank[k])
            if r*(o_d+i_d) >= o_d*i_d:          # factorisation not worth it: keep dense
                m.weight.data = orig[k].clone(); cost += o_d*i_d
            else:
                U, S, Vh, Hc = d["U"], d["S"], d["Vh"], d["Hc"]
                Yr = (U[:, :r]*S[:r]) @ Vh[:r]
                m.weight.data = torch.linalg.solve_triangular(Hc.T, Yr.T, upper=True).T
                cost += r*(o_d+i_d)
            del d; gc.collect()
        rho = cost/dense
        ppl = perplexity(model, ev, a.seqlen, a.neval)
        rr = [rank[k]/min(*orig[k].shape) for k in lins]
        rows.append({"floor": a.floor, "target_rho": target, "rho": rho, "ppl": ppl,
                     "d_ppl_pct": 100*(ppl-base)/base,
                     "keep_frac_min": min(rr), "keep_frac_max": max(rr),
                     "keep_frac_mean": sum(rr)/len(rr),
                     "ranks": {k: rank[k] for k in lins}})
        log(f"floor={a.floor} target_rho={target:<7} actual_rho={rho:.4f} ppl={ppl:9.3f} "
            f"dppl={rows[-1]['d_ppl_pct']:+9.2f}%  keep_frac "
            f"min={min(rr):.2f} mean={sum(rr)/len(rr):.2f} max={max(rr):.2f}")
        json.dump({"model": a.model, "base_ppl": base, "rows": rows}, open(a.out, "w"), indent=1)
    for k, m in lins.items(): m.weight.data = orig[k]
    log("done -> " + a.out)

main()
