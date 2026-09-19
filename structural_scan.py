"""
Effort-gap experiment 0b: second hollowing family — structured pruning.

Low-rank factorisation is one way to make a committed matrix cheap. The other
family that actually saves FLOPs on real hardware is STRUCTURED removal:
dropping whole MLP neurons (a row of gate_proj/up_proj + matching column of
down_proj) or whole attention heads. Unstructured sparsity is excluded: it does
not reduce dense-matmul cost without hardware support.

Question: do real weights contain structurally dead units a hollow provider
could drop for free?
"""
import json, os, sys, gc
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

class LazyTensors:
    """Fetch one tensor at a time; never hold the whole model in RAM."""
    def __init__(self, path):
        self.handles, self.index = [], {}
        for fn in sorted(f for f in os.listdir(path) if f.endswith(".safetensors")):
            h = safe_open(os.path.join(path, fn), framework="pt")
            self.handles.append(h)
            for k in h.keys():
                self.index[k] = h
    def __getitem__(self, k):
        return self.index[k].get_tensor(k).float()

def load_all(model_id):
    path = snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json"])
    cfg = json.load(open(os.path.join(path, "config.json")))
    return LazyTensors(path), cfg

def run(model_id):
    T, cfg = load_all(model_id)
    L, H = cfg["num_hidden_layers"], cfg["num_attention_heads"]
    d = cfg["hidden_size"]; hd = d // H
    print(f"\n=== {model_id} ===  layers={L} heads={H} d={d}")

    neuron_keep, head_keep = [], []
    for i in range(L):
        g = T[f"model.layers.{i}.mlp.gate_proj.weight"]   # [inter, d]
        u = T[f"model.layers.{i}.mlp.up_proj.weight"]     # [inter, d]
        dn = T[f"model.layers.{i}.mlp.down_proj.weight"]  # [d, inter]
        # importance of MLP neuron j ~ ||g_j|| * ||u_j|| * ||dn_:,j||
        imp = g.norm(dim=1) * u.norm(dim=1) * dn.norm(dim=0)
        imp = imp / imp.sum()
        srt = torch.sort(imp, descending=True).values
        c = torch.cumsum(srt, 0)
        # neurons needed to retain 99% of total importance mass
        k99 = int((c < 0.99).sum()) + 1
        neuron_keep.append(k99 / len(imp))

        o = T[f"model.layers.{i}.self_attn.o_proj.weight"]  # [d, H*hd]
        himp = torch.stack([o[:, h*hd:(h+1)*hd].norm() for h in range(H)])
        himp = himp / himp.sum()
        hs = torch.sort(himp, descending=True).values
        hc = torch.cumsum(hs, 0)
        hk99 = int((hc < 0.99).sum()) + 1
        head_keep.append(hk99 / H)
        del g, u, dn, o; gc.collect()

    nk = torch.tensor(neuron_keep); hk = torch.tensor(head_keep)
    print(f"  MLP neurons needed for 99% importance mass: "
          f"mean {nk.mean():.3f}  min {nk.min():.3f}  max {nk.max():.3f}  (1.0 = none droppable)")
    print(f"  Attn heads   needed for 99% importance mass: "
          f"mean {hk.mean():.3f}  min {hk.min():.3f}  max {hk.max():.3f}")
    print(f"  => best-case structured rho (MLP): {nk.mean():.3f}   (attn): {hk.mean():.3f}")

for mid in sys.argv[1:]:
    run(mid)
