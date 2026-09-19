"""
Effort-gap thesis, experiment 0: how much free hollowing do real LLM weights allow?

For every 2-D weight matrix W (m x n) in a model, compute the singular value
spectrum and ask: what is the smallest rank r that preserves W to a given
relative Frobenius error, and does a rank-r factorisation actually cost less
than the dense matmul?

Cost model (per token, ignoring constants):
    dense      W @ x        -> m * n
    factored   U_r @ (V_r@x)-> r * (m + n)
    rho_matrix = r * (m + n) / (m * n)

Rank truncation only saves FLOPs when r < m*n/(m+n)  ("r_breakeven").
For a square d x d matrix that means r < d/2.

Streams one shard at a time so peak memory is one matrix, not one model.
"""
import argparse, gc, json, os, sys
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

ERR_TOLS = [0.001, 0.01, 0.05, 0.10]   # relative Frobenius reconstruction error


def rank_for_error(svals: torch.Tensor, tol: float) -> int:
    """Smallest r with ||W - W_r||_F / ||W||_F <= tol.
    Eckart-Young: ||W - W_r||_F^2 = sum_{i>r} s_i^2."""
    sq = svals.double() ** 2
    total = sq.sum()
    tail = torch.flip(torch.cumsum(torch.flip(sq, [0]), 0), [0])  # tail[r] = sum_{i>=r} s_i^2
    ratio = (tail / total).sqrt()
    idx = (ratio <= tol).nonzero()
    return int(idx[0].item()) if len(idx) else len(svals)


def scan(model_id: str, out_path: str, skip_substrings=("embed", "lm_head")):
    path = snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json"])
    files = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
    rows = []
    for fn in files:
        with safe_open(os.path.join(path, fn), framework="pt") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                if t.ndim != 2 or any(s in name for s in skip_substrings):
                    del t; continue
                W = t.float()
                m, n = W.shape
                del t
                s = torch.linalg.svdvals(W)
                r_break = (m * n) / (m + n)
                row = {
                    "model": model_id, "name": name, "m": m, "n": n,
                    "dense_cost": m * n, "r_breakeven": r_break,
                    "s_max": float(s[0]), "s_min": float(s[-1]),
                    "stable_rank": float((s.double() ** 2).sum() / s[0].double() ** 2),
                }
                for tol in ERR_TOLS:
                    r = rank_for_error(s, tol)
                    row[f"r@{tol}"] = r
                    row[f"rho@{tol}"] = r * (m + n) / (m * n)
                rows.append(row)
                print(f"  {name:55s} {m:5d}x{n:<6d} "
                      + " ".join(f"r@{t}={row[f'r@{t}']:5d}(rho={row[f'rho@{t}']:.2f})"
                                 for t in ERR_TOLS), flush=True)
                del W, s; gc.collect()
    with open(out_path, "w") as fh:
        json.dump(rows, fh, indent=1)
    return rows


def summarise(rows):
    """Aggregate to a whole-model effort ratio: total factored cost / total dense cost."""
    out = {}
    dense = sum(r["dense_cost"] for r in rows)
    for tol in ERR_TOLS:
        fact = sum(min(r[f"r@{tol}"] * (r["m"] + r["n"]), r["dense_cost"]) for r in rows)
        helped = sum(1 for r in rows if r[f"r@{tol}"] < r["r_breakeven"])
        out[tol] = {"rho_model": fact / dense, "matrices_helped": helped, "n_matrices": len(rows)}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--outdir", default="/home/claude/effortgap/results")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    for mid in a.models:
        print(f"\n=== {mid} ===", flush=True)
        p = os.path.join(a.outdir, mid.replace("/", "__") + ".json")
        rows = scan(mid, p)
        s = summarise(rows)
        print(f"\n--- SUMMARY {mid} ---")
        for tol, v in s.items():
            print(f"  err<={tol:<6}  rho_model={v['rho_model']:.3f}  "
                  f"matrices where truncation saves FLOPs: {v['matrices_helped']}/{v['n_matrices']}")
        with open(os.path.join(a.outdir, "summary.jsonl"), "a") as fh:
            fh.write(json.dumps({"model": mid, "summary": {str(k): v for k, v in s.items()}}) + "\n")
