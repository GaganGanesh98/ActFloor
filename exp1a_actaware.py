"""
Exp 1a v2 — activation-aware low-rank truncation, memory-bounded.

v1 OOM'd: it accumulated X^T X for every nn.Linear simultaneously. The MLP
down_proj Hessians alone are 4864^2 fp32 each. v2 walks the decoder in chunks,
hooking only the layers in the current chunk and spilling each Hessian to disk.

Method (unchanged): the adversary minimises output error over the real activation
distribution, not weight error.
    H = X X^T = S^T S  (Cholesky)      W_r = truncate_r(W S) S^{-1}
Plain SVD is run as the weak-adversary baseline.

Effort ratio:  rho = sum_i min(r_i(m_i+n_i), m_i n_i) / sum_i m_i n_i
A rank-r factorisation only saves FLOPs when r < mn/(m+n).
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--seqlen", type=int, default=256)
    ap.add_argument("--ncalib", type=int, default=24)
    ap.add_argument("--neval", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=6)
    ap.add_argument("--ratios", default="0.9,0.75,0.5,0.3")
    ap.add_argument("--hdir", default="/home/claude/effortgap/hess")
    ap.add_argument("--out", default="results/exp1a.json")
    a = ap.parse_args()
    os.makedirs(a.hdir, exist_ok=True); os.makedirs("results", exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    calib = get_ids(tok, a.seqlen*(a.ncalib+2), 0)
    ev    = get_ids(tok, a.seqlen*(a.neval+2), 1)

    decoder = model.model.layers
    nL = len(decoder)
    lin = lambda i: {f"{i}.{n}": m for n, m in decoder[i].named_modules() if isinstance(m, nn.Linear)}

    # ---- pass 1: activation Hessians, chunked over decoder layers -------------
    for lo in range(0, nL, a.chunk):
        hi = min(lo+a.chunk, nL)
        H, hooks = {}, []
        targets = {}
        for i in range(lo, hi): targets.update(lin(i))
        def mk(k):
            def f(mod, inp, out):
                x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
                H[k] = x.T@x if k not in H else H[k] + x.T@x
            return f
        for k, m in targets.items(): hooks.append(m.register_forward_hook(mk(k)))
        for s in range(a.ncalib):
            c = calib[s*a.seqlen:(s+1)*a.seqlen].unsqueeze(0)
            if c.shape[1] < a.seqlen: break
            model(c)
        for h in hooks: h.remove()
        for k, h in H.items():
            d = h.shape[0]
            h = h + 1e-2*torch.diag(h).mean()*torch.eye(d)
            try: c = torch.linalg.cholesky(h.double())
            except Exception:
                h = h.double() + 1e-2*torch.diag(h.double()).mean()*torch.eye(d, dtype=torch.float64)
                c = torch.linalg.cholesky(h)
            torch.save(c.float(), os.path.join(a.hdir, k.replace(".", "_")+".pt"))
        log(f"hessians layers {lo}-{hi-1} done ({len(H)} matrices)")
        del H, targets; gc.collect()

    all_lin = {}
    for i in range(nL): all_lin.update(lin(i))
    orig = {k: m.weight.data.clone() for k, m in all_lin.items()}
    dense = sum(w.shape[0]*w.shape[1] for w in orig.values())
    base = perplexity(model, ev, a.seqlen, a.neval)
    log(f"baseline ppl = {base:.4f}")

    rows = []
    for mode in ("act_aware", "plain_svd"):
        for ratio in [float(x) for x in a.ratios.split(",")]:
            cost = 0
            for k, m in all_lin.items():
                W = orig[k]; o_d, i_d = W.shape
                r = max(1, int(round(ratio*min(o_d, i_d))))
                if mode == "act_aware":
                    Hc = torch.load(os.path.join(a.hdir, k.replace(".", "_")+".pt"))
                    Ws = W @ Hc          # H = L L^T -> whiten with L, NOT L^T
                    U, S, Vh = torch.linalg.svd(Ws, full_matrices=False)
                    Wr = (U[:, :r]*S[:r]) @ Vh[:r]
                    m.weight.data = torch.linalg.solve_triangular(Hc.T, Wr.T, upper=True).T
                    del Hc, Ws, U, S, Vh, Wr
                else:
                    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                    m.weight.data = (U[:, :r]*S[:r]) @ Vh[:r]
                    del U, S, Vh
                cost += min(r*(o_d+i_d), o_d*i_d)
                gc.collect()
            rho = cost/dense
            ppl = perplexity(model, ev, a.seqlen, a.neval)
            rows.append({"mode": mode, "keep": ratio, "rho": rho, "ppl": ppl,
                         "d_ppl_pct": 100*(ppl-base)/base})
            log(f"{mode:10s} keep={ratio:<5} rho={rho:.4f} ppl={ppl:10.3f} "
                f"dppl={rows[-1]['d_ppl_pct']:+9.2f}%")
            json.dump({"model": a.model, "base_ppl": base, "rows": rows}, open(a.out, "w"), indent=1)
    for k, m in all_lin.items(): m.weight.data = orig[k]
    log("done -> " + a.out)

main()
