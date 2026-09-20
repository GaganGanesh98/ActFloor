# ---
# jupyter:
#   jupytext:
#     formats: ipynb,_src.py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3.12 (effortgap .venv)
#     language: python
#     name: effortgap-venv
# ---

# %% [markdown]
# # Effort-gap Exp 1a / 1a' / 1b — single-GPU notebook
#
# Port of `exp1a_actaware.py` (uniform activation-aware truncation) and
# `exp1a_prime.py` (non-uniform greedy allocation) to one GPU notebook, plus
# **Exp 1b** (LoRA distillation against the full model at `B_calib` steps).
#
# ## What is preserved exactly
#
# * `H = X Xᵀ = L Lᵀ` by Cholesky; whitening uses **`L`, not `Lᵀ`**;
#   `W_r = truncate_r(W L) L⁻¹`.
# * Damping `H += damp · mean(diag H) · I`, `damp = 1e-2`, with retry on failure.
# * Effort ratio `ρ = Σᵢ min(rᵢ(mᵢ+nᵢ), mᵢnᵢ) / Σᵢ mᵢnᵢ`. The break-even check
#   `r < mn/(m+n)` is kept: above it the factorisation is not bought.
# * Greedy allocator: exact for the separable surrogate, sorted by
#   `σ²ᵢⱼ / (mᵢ+nᵢ)`, with a per-matrix rank floor.
# * Plain SVD retained as the weak-adversary baseline, and the protocol check
#   *activation-aware must dominate plain SVD at equal rank* is enforced as an
#   assertion, both synthetically and on real matrices in the activation metric.
#
# ## Deliberate changes from the CPU scripts (all flagged in the output JSON)
#
# 1. **Calibration corpus is `wikitext-2-raw-v1`**, not tinyshakespeare.
#    Calibration and distillation come from disjoint slices of `train`;
#    evaluation is `test`. Nothing is calibrated and evaluated on the same text.
# 2. **Truncation is materialised as a real factorisation.** A truncated matrix
#    becomes two `nn.Linear`s (`n→r`, `r→m`), so `ρ` is physically realised
#    rather than only accounted for. This is also what makes a 7B student fit
#    for Exp 1b.
# 3. **Dense fallback.** When `r(m+n) ≥ mn` the factorisation is not bought, and
#    the matrix keeps its **original** weight (this is `exp1a_prime.py`'s
#    behaviour). `exp1a_actaware.py` instead served a truncated-but-dense matrix
#    at full price, which is strictly dominated — no adversary paying `mn` would
#    do that. Set `cfg.dense_fallback = "truncated"` to reproduce the old arm.
# 4. **Cached factors are `(U, S, Z)` with `Z = Vh L⁻¹`**, not `L` itself.
#    `W_r = U_r S_r Z_r` is algebraically identical (row selection commutes with
#    right-multiplication by `L⁻¹`) and is 4× smaller to store for `down_proj`.
# 5. **Exp 1b KL is computed over a top-K partition of the vocabulary**
#    (`K` classes individually + one lumped tail), so teacher logits can be
#    cached once. This is a proper KL between coarsened distributions and a
#    lower bound on the full-vocab KL. A validation cell measures the gap.
# 6. **LoRA is merged into the factors conceptually**: adapters sit on the two
#    factor matrices, so merging changes no shape and `ρ` is unaffected by
#    distillation. Evaluation runs with adapters active, which is exactly equal
#    to evaluating the merged weights.
#
# ## Hardware reality check — read this before starting a long run
#
# | model | VRAM (streamed pass) | system RAM to load | factor cache on disk |
# |---|---|---|---|
# | Qwen2.5-0.5B | ~3 GB | ~2 GB | ~1.5 GB |
# | Qwen2.5-1.5B | ~8 GB | ~4 GB | ~6.5 GB |
# | Qwen2.5-7B | ~10 GB | **~16 GB** | **~33 GB** |
#
# * The Hessian pass, truncation and perplexity all run **layer-streamed**, so
#   7B fits in 15 GB of VRAM. The binding constraint for 7B is **system RAM**
#   (the model is held on CPU between layers): a free Colab T4 runtime has
#   12.7 GB and **cannot** load 7B. Use a high-RAM runtime.
# * T4 is Turing and has **no bfloat16**. The notebook auto-selects fp16 there
#   and says so. bf16 is used on Ampere and later.
# * 33 GB of 7B factors does not fit a free 15 GB Drive. The cache is tiered:
#   everything goes to local scratch (survives cell reruns), and Drive keeps the
#   small, expensive-to-recompute artefacts plus all results.
#
# Run the **smoke test** at the bottom first. It exercises every path on
# Qwen2.5-0.5B in a few minutes.

# %%
# --- dependencies ------------------------------------------------------------
import subprocess, sys

def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)

if __name__ == "__main__":  # skipped on import (auditor.py); Jupyter/Colab set __name__ == "__main__"
    _pip("transformers>=4.44", "datasets", "accelerate", "peft", "safetensors")
    print("deps ok")

# %%
# --- imports, preflight ------------------------------------------------------
import gc, hashlib, json, math, os, shutil, time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

T0 = time.time()
def log(m):
    print(f"[{time.time() - T0:8.1f}s] {m}", flush=True)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cpu":
    print("WARNING: no CUDA device. This notebook is written for a GPU; it will "
          "run on CPU but slowly.")
    DTYPE = torch.float32
    BF16_OK = False
else:
    BF16_OK = torch.cuda.is_bf16_supported()
    DTYPE = torch.bfloat16 if BF16_OK else torch.float16
    p = torch.cuda.get_device_properties(0)
    log(f"GPU: {p.name}  {p.total_memory/2**30:.1f} GiB  sm_{p.major}{p.minor}")
    if not BF16_OK:
        print("NOTE: this GPU has no bfloat16 (pre-Ampere). Falling back to "
              "float16 for model weights. All accumulators, Hessians, Cholesky "
              "factors and SVDs stay in fp32/fp64 regardless, so the numerics "
              "of the method are unaffected.")
log(f"weight dtype: {DTYPE}")

try:
    import psutil
    log(f"system RAM: {psutil.virtual_memory().total/2**30:.1f} GiB total, "
        f"{psutil.virtual_memory().available/2**30:.1f} GiB available")
except Exception:
    pass


def vram(tag=""):
    if DEV != "cpu":
        log(f"  vram {tag}: {torch.cuda.memory_allocated()/2**30:5.2f} GiB alloc, "
            f"{torch.cuda.max_memory_allocated()/2**30:5.2f} GiB peak")


def sweep():
    gc.collect()
    if DEV != "cpu":
        torch.cuda.empty_cache()


def _cache_base() -> str:
    """Root directory for on-disk caches, resolved once at import time.

    Priority: `EFFORTGAP_SCRATCH` env var > `/content` (Colab; keeps existing
    Colab behaviour unchanged) > a folder next to this file > `~/.cache`.
    `/content` only exists and is writable on Colab, so off-Colab this falls
    through to a repo-local or home-directory cache instead of the
    hardcoded Colab paths the notebook used to have."""
    env = os.environ.get("EFFORTGAP_SCRATCH")
    if env:
        return env
    if os.path.isdir("/content") and os.access("/content", os.W_OK):
        return "/content"
    base = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() \
        else os.getcwd()
    if os.access(base, os.W_OK):
        return base
    return os.path.expanduser("~/.cache")


_CACHE_BASE = _cache_base()
_ON_COLAB = _CACHE_BASE == "/content"
DEFAULT_SCRATCH = os.path.join(_CACHE_BASE, "effortgap")
DEFAULT_SMOKE_SCRATCH = os.path.join(_CACHE_BASE, "effortgap_smoke")
DEFAULT_DRIVE = "/content/drive/MyDrive/effortgap" if _ON_COLAB else None
log(f"cache base: {_CACHE_BASE} (colab={_ON_COLAB})")


# %% [markdown]
# ## Configuration
#
# One `Cfg` per model. `ratios` drives the uniform arms; the `ρ` values they
# produce are then fed back as the greedy arm's targets, so every method is
# compared at matched `ρ` (this is what `exp1a_prime.py` did by hand with
# `--rhos 0.9,0.75,0.6357,0.5,0.3817`).

# %%
@dataclass
class Cfg:
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # --- data -----------------------------------------------------------------
    dataset: str = "Salesforce/wikitext"  # HF deprecated the old loading-script
                                           # "wikitext" repo; same corpus, new host
    dataset_config: str = "wikitext-2-raw-v1"
    seqlen: int = 512
    ncalib: int = 128           # calibration sequences  (train, slice A)
    neval: int = 64             # evaluation sequences   (test)
    ndistil: int = 512          # distillation sequences (train, slice B)
    distil_seqlen: int = 256

    # --- truncation -----------------------------------------------------------
    damp: float = 1e-2
    ratios: Tuple[float, ...] = (0.9, 0.75, 0.5, 0.3)
    floor: float = 0.25         # per-matrix rank floor for the greedy allocator
    floors: Tuple[float, ...] = ()   # sweep the floor; () -> just `floor`.
                                # `floor` is NOT part of calib_key(), so a sweep
                                # reuses the cached factors and costs only
                                # re-truncation + eval. Exp 1b always distils the
                                # `floor` arm, so that value must be in the sweep.
    methods: Tuple[str, ...] = ("plain_svd", "act_aware_uniform", "act_aware_greedy")
    plain_svd_ratios: Optional[Tuple[float, ...]] = None  # None -> all of `ratios`
    dense_fallback: str = "original"   # "original" (prime) | "truncated" (1a)
    chol_fp64: bool = True
    fast_svd: bool = False      # True -> Gram+eigh instead of exact SVD (7B lever)

    # --- Exp 1b ---------------------------------------------------------------
    b_calib: Tuple[int, ...] = (0, 100, 300, 500)
    distil_methods: Tuple[str, ...] = ("act_aware_greedy",)
    distil_ratios: Optional[Tuple[float, ...]] = None  # None -> all of `ratios`
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lr: float = 1e-4
    grad_clip: float = 1.0
    topk: int = 256             # vocab classes kept individually in the KL
    grad_checkpointing: bool = False

    # --- caching / io ---------------------------------------------------------
    scratch: str = DEFAULT_SCRATCH
    drive: Optional[str] = DEFAULT_DRIVE
    drive_factors: bool = True  # mirror the big (U,S,Z) factors to Drive too
    factor_dtype: str = "float32"
    cache_plain: bool = True     # also cache the plain-SVD factors
    cache_cholesky: bool = False # also cache L itself (4x the size of (U,Z)
                                 # for down_proj; nothing downstream needs it,
                                 # since Z = Vh L⁻¹ already encodes it)
    out_name: str = "exp1_gpu_results.json"

    # --- adapter checkpointing ------------------------------------------------
    # Off by default, so CFG_1_5B / CFG_7B / SMOKE are untouched. When on,
    # lora_distil writes the LoRA adapter (NOT merged weights) plus a manifest
    # at every B_calib checkpoint, to Drive rather than scratch -- the point is
    # surviving a runtime death, which is what lost the Run 1 adapters.
    save_adapters: bool = False
    adapter_dir: Optional[str] = None   # None -> <drive>/adapters

    # --- protocol checks ------------------------------------------------------
    sanity_layers: Tuple[int, ...] = (0,)   # layers on which to run the L vs Lᵀ check
    sanity_paths: Tuple[str, ...] = ("self_attn.q_proj", "self_attn.o_proj")
    sanity_ratio: float = 0.5
    enforce_sanity: bool = True

    def calib_key(self) -> str:
        """Hash of everything that changes the Hessians / factors. Prevents a
        stale cache from being silently reused after a config change."""
        h = hashlib.sha1(json.dumps({
            "model": self.model_id, "ds": self.dataset, "cfg": self.dataset_config,
            "seqlen": self.seqlen, "ncalib": self.ncalib, "damp": self.damp,
            "chol_fp64": self.chol_fp64, "fast_svd": self.fast_svd,
            "factor_dtype": self.factor_dtype,
        }, sort_keys=True).encode()).hexdigest()[:12]
        return f"{self.model_id.replace('/', '__')}.{h}"


def mount_drive(cfg: Cfg) -> Optional[str]:
    if cfg.drive is None:
        return None
    try:
        from google.colab import drive as _d
        if not os.path.ismount("/content/drive"):
            _d.mount("/content/drive")
    except Exception as e:
        log(f"Drive unavailable ({e}); scratch-only caching.")
        return None
    os.makedirs(cfg.drive, exist_ok=True)
    return cfg.drive


# %% [markdown]
# ## Two-tier cache
#
# `scratch` is local disk — fast, big, survives cell reruns, dies with the
# runtime. `drive` survives runtime restarts but is small. Reads check scratch
# first, then Drive (copying down). Writes always go to scratch, and to Drive
# when the artefact is small enough or `drive_factors` is on.
#
# **Cost of `drive_factors=True`.** The factor cache is ~1.5 GB at 0.5B, ~6.5 GB
# at 1.5B and ~33 GB at 7B. Drive writes run at a few MB/s, so mirroring 1.5B
# adds roughly 20 minutes to the calibration pass — worth it once, because a
# rerun after a runtime restart then costs nothing. At 7B it is set to `False`:
# 33 GB does not fit a 15 GB Drive, so only the singular values, the manifest
# and the results go there, and a restart re-runs the calibration pass.

# %%
class Cache:
    def __init__(self, cfg: Cfg, key: Optional[str] = None):
        self.key = key or cfg.calib_key()
        self.scratch = os.path.join(cfg.scratch, self.key)
        os.makedirs(self.scratch, exist_ok=True)
        d = mount_drive(cfg)
        self.drive = os.path.join(d, self.key) if d else None
        if self.drive:
            os.makedirs(self.drive, exist_ok=True)
        self.mirror_big = cfg.drive_factors and self.drive is not None
        log(f"cache key {self.key}")
        log(f"  scratch: {self.scratch}")
        log(f"  drive  : {self.drive or '(none)'}")

    @staticmethod
    def _fn(name: str) -> str:
        return name.replace("/", "_").replace(".", "_") + ".pt"

    def has(self, name: str) -> bool:
        return os.path.exists(os.path.join(self.scratch, self._fn(name))) or (
            self.drive is not None
            and os.path.exists(os.path.join(self.drive, self._fn(name))))

    def get(self, name: str, map_location="cpu"):
        f = self._fn(name)
        p = os.path.join(self.scratch, f)
        if not os.path.exists(p):
            if self.drive is None:
                raise FileNotFoundError(name)
            q = os.path.join(self.drive, f)
            if not os.path.exists(q):
                raise FileNotFoundError(name)
            shutil.copyfile(q, p)          # pull down so later reads are fast
        return torch.load(p, map_location=map_location, weights_only=False)

    def put(self, name: str, obj, big: bool = False):
        f = self._fn(name)
        p = os.path.join(self.scratch, f)
        torch.save(obj, p)
        if self.drive is not None and (not big or self.mirror_big):
            try:
                shutil.copyfile(p, os.path.join(self.drive, f))
            except Exception as e:
                log(f"  drive write failed for {name}: {e}")

    def disk_free_gb(self) -> float:
        return shutil.disk_usage(self.scratch).free / 2**30


# %% [markdown]
# ## Data — wikitext-2-raw-v1, disjoint splits
#
# * calibration : `train`, tokens `[0, ncalib·seqlen)`
# * distillation: `train`, tokens `[ncalib·seqlen, …)`  — disjoint from calibration
# * evaluation  : `test`                                 — disjoint from both
#
# The Hessians therefore never see a token that perplexity is measured on.

# %%
_TOK_CACHE: Dict[Tuple[str, str, str], torch.Tensor] = {}


def _tokenised(tok, cfg: Cfg, split: str) -> torch.Tensor:
    k = (cfg.dataset, cfg.dataset_config, split, getattr(tok, "name_or_path", ""))
    if k in _TOK_CACHE:
        return _TOK_CACHE[k]
    from datasets import load_dataset
    ds = load_dataset(cfg.dataset, cfg.dataset_config, split=split)
    txt = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(txt, return_tensors="pt").input_ids[0]
    _TOK_CACHE[k] = ids
    log(f"{cfg.dataset_config}/{split}: {len(ids):,} tokens")
    return ids


def make_splits(tok, cfg: Cfg):
    """Returns (calib, eval, distil) as lists of 1-D LongTensors."""
    train = _tokenised(tok, cfg, "train")
    test = _tokenised(tok, cfg, "test")

    def chunks(ids, seqlen, n, offset=0):
        out = []
        for i in range(n):
            s = offset + i * seqlen
            if s + seqlen > len(ids):
                break
            out.append(ids[s:s + seqlen])
        return out

    calib = chunks(train, cfg.seqlen, cfg.ncalib, 0)
    d_off = cfg.ncalib * cfg.seqlen
    distil = chunks(train, cfg.distil_seqlen, cfg.ndistil, d_off)
    ev = chunks(test, cfg.seqlen, cfg.neval, 0)
    assert len(calib) == cfg.ncalib, f"train split too short for {cfg.ncalib} calib seqs"
    assert len(ev) > 0, "eval split empty"
    if len(distil) < cfg.ndistil:
        log(f"NOTE: only {len(distil)} distillation sequences available "
            f"(asked {cfg.ndistil}); they will be cycled.")
    log(f"splits: calib {len(calib)}x{cfg.seqlen} (train[0:{d_off}]), "
        f"distil {len(distil)}x{cfg.distil_seqlen} (train[{d_off}:]), "
        f"eval {len(ev)}x{cfg.seqlen} (test) — disjoint")
    return calib, ev, distil


# %% [markdown]
# ## Protocol check 0 — synthetic whitening test
#
# From `findings-exp1a-actaware.md`: the whitening bug was caught by the result
# being backwards. On an anisotropic `X` the ordering of output error must be
#
# ```
# act-aware with L   <   act-aware with Lᵀ   <   plain SVD
# ```
#
# If this cell fails, the rest of the notebook is measuring the wrong thing.

# %%
def truncate_to_rank(M: torch.Tensor, r: int) -> torch.Tensor:
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    return (U[:, :r] * S[:r]) @ Vh[:r]


def synthetic_whitening_check(m=96, n=128, N=4096, r=32, seed=0, verbose=True):
    g = torch.Generator().manual_seed(seed)
    # strongly anisotropic activations: a few directions carry all the energy
    scale = torch.logspace(0, -3, n).unsqueeze(1)
    X = scale * torch.randn(n, N, generator=g, dtype=torch.float64)
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
    X = Q @ X
    W = torch.randn(m, n, generator=g, dtype=torch.float64)

    H = X @ X.T
    H = H + 1e-2 * torch.diagonal(H).mean() * torch.eye(n, dtype=torch.float64)
    L = torch.linalg.cholesky(H)                       # H = L Lᵀ, L lower

    def out_err(Wr):
        return float(((W - Wr) @ X).pow(2).sum().sqrt())

    # correct: whiten with L
    Wr_L = torch.linalg.solve_triangular(
        L.T, truncate_to_rank(W @ L, r).T, upper=True).T
    # the bug: whiten with Lᵀ
    Wr_Lt = torch.linalg.solve_triangular(
        L, truncate_to_rank(W @ L.T, r).T, upper=False).T
    # weak adversary
    Wr_p = truncate_to_rank(W, r)

    e_L, e_Lt, e_p = out_err(Wr_L), out_err(Wr_Lt), out_err(Wr_p)
    if verbose:
        log(f"synthetic check r={r}: act-aware(L)={e_L:.4f}  "
            f"act-aware(Lᵀ)={e_Lt:.4f}  plain SVD={e_p:.4f}")
    assert e_L < e_p, (
        f"activation-aware (L) {e_L:.4f} does NOT beat plain SVD {e_p:.4f} — "
        "the whitening factor is wrong")
    assert e_L < e_Lt, (
        f"whitening with L {e_L:.4f} should beat whitening with Lᵀ {e_Lt:.4f}")
    return {"act_aware_L": e_L, "act_aware_LT": e_Lt, "plain_svd": e_p}


# Deliberately NOT under a __main__ guard: this is a cheap, self-contained
# protocol check (L vs L-transpose whitening on synthetic data) and running it
# on every import -- including from auditor.py -- is a feature, not a cost.
_ = synthetic_whitening_check()


# %% [markdown]
# ## Model plumbing and layer streaming
#
# Everything heavy runs **one decoder layer at a time**: the layer is moved to
# the GPU, the whole batch of hidden states is pushed through it, then it goes
# back to CPU. Peak VRAM is therefore one layer plus the hidden-state buffer,
# not the whole model. This is what lets a 7B run in 15 GB.
#
# `Qwen2DecoderLayer` needs `position_embeddings` (the rotary cos/sin) passed
# in by the parent `Qwen2Model`. They depend only on the position ids, which are
# identical for every equal-length sequence, so they are captured once from a
# stub forward and reused.

# %%
def load_model(cfg: Cfg, device="cpu"):
    kw = dict(low_cpu_mem_usage=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(cfg.model_id, dtype=DTYPE, **kw)
    except TypeError:                       # transformers < 4.56 spells it differently
        model = AutoModelForCausalLM.from_pretrained(cfg.model_id, torch_dtype=DTYPE, **kw)
    model.eval()
    model.config.use_cache = False
    if device != "cpu":
        model.to(device)
    return model


def decoder_layers(model):
    return model.model.layers


def set_submodule(root: nn.Module, path: str, value: nn.Module):
    parts = path.split(".")
    obj = root
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], value)


def get_submodule(root: nn.Module, path: str) -> nn.Module:
    obj = root
    for p in path.split("."):
        obj = getattr(obj, p)
    return obj


def target_linears(model) -> List[Tuple[str, int, str, nn.Module]]:
    """(key, layer_index, path_within_layer, module) for every nn.Linear inside
    the decoder stack. Embeddings and lm_head are deliberately excluded — they
    are not what ρ is defined over."""
    out = []
    for i, layer in enumerate(decoder_layers(model)):
        for name, mod in layer.named_modules():
            if isinstance(mod, nn.Linear):
                out.append((f"{i}.{name}", i, name, mod))
    return out


# Qwen2 / Llama share inputs between these projections. One Hessian per group
# instead of one per matrix: 4 per layer rather than 7. Verified at runtime.
SHARED_INPUT_GROUP = {
    "self_attn.q_proj": "attn_in",
    "self_attn.k_proj": "attn_in",
    "self_attn.v_proj": "attn_in",
    "self_attn.o_proj": "attn_out",
    "mlp.gate_proj": "mlp_in",
    "mlp.up_proj": "mlp_in",
    "mlp.down_proj": "mlp_out",
}


def group_name(path: str) -> str:
    return SHARED_INPUT_GROUP.get(path, path)   # unknown architecture -> own group


class _Stop(Exception):
    pass


@torch.no_grad()
def capture_layer_kwargs(model, sample_ids: torch.Tensor, device):
    """Run just far enough to see what Qwen2Model hands the first decoder layer."""
    layers = decoder_layers(model)
    holder = {}

    class Catcher(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, hidden_states, **kw):
            holder["hs"] = hidden_states
            holder["kw"] = kw
            raise _Stop()

    orig0 = layers[0]
    layers[0] = Catcher(orig0)
    model.model.embed_tokens.to(device)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb.to(device)
    try:
        model(sample_ids.unsqueeze(0).to(device))
    except _Stop:
        pass
    finally:
        layers[0] = orig0

    kw = dict(holder["kw"])
    for k in ("past_key_value", "past_key_values"):
        kw.pop(k, None)
    kw["use_cache"] = False
    # sanity: hidden states entering layer 0 are exactly the token embeddings
    emb = model.model.embed_tokens(sample_ids.unsqueeze(0).to(device))
    assert torch.equal(holder["hs"], emb), (
        "hidden states entering layer 0 are not the raw embeddings; this model "
        "scales or otherwise pre-processes them and the streaming path must be "
        "adapted")
    return kw


@torch.no_grad()
def embed_batch(model, ids_list: List[torch.Tensor], device) -> torch.Tensor:
    model.model.embed_tokens.to(device)
    return torch.stack([
        model.model.embed_tokens(ids.unsqueeze(0).to(device))[0] for ids in ids_list
    ])


def _layer_out(out):
    return out[0] if isinstance(out, (tuple, list)) else out


@torch.no_grad()
def stream_layer(layer, hs: torch.Tensor, kw, device, micro=1) -> torch.Tensor:
    """Push (B, L, d) hidden states through one decoder layer, in place so the
    buffer is never duplicated."""
    for s in range(0, hs.shape[0], micro):
        hs[s:s + micro] = _layer_out(layer(hs[s:s + micro], **kw))
    return hs


@torch.no_grad()
def stream_head(model, hs: torch.Tensor, ids_list, device, offload=True) -> float:
    """Final norm + lm_head, returning perplexity over the batch."""
    norm = model.model.norm.to(device)
    head = model.lm_head.to(device)
    nll = 0.0
    ntok = 0
    for b, ids in enumerate(ids_list):
        h = norm(hs[b:b + 1].to(device))
        logits = head(h).float()
        tgt = ids.unsqueeze(0).to(device)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]), tgt[:, 1:].reshape(-1))
        n = tgt.shape[1] - 1
        nll += float(loss) * n
        ntok += n
        del logits, h
    if offload:
        norm.to("cpu")
        head.to("cpu")
    return math.exp(nll / ntok)


def model_is_resident(model, device) -> bool:
    try:
        return next(model.parameters()).device.type == torch.device(device).type
    except StopIteration:
        return False


@torch.no_grad()
def perplexity(model, ids_list, device, kw=None, micro=1, streamed=None) -> float:
    """Perplexity on held-out sequences.

    Accounting is identical to the CPU scripts: mean NLL over the L-1 predicted
    positions of each sequence, weighted by L-1, then exponentiated.

    `streamed=None` auto-selects: a model already resident on the GPU is run
    whole (fast); a model living on CPU is streamed one decoder layer at a time
    so that peak VRAM is one layer plus the hidden-state buffer.
    """
    if streamed is None:
        streamed = not model_is_resident(model, device)

    if not streamed:
        nll = 0.0
        ntok = 0
        for ids in ids_list:
            c = ids.unsqueeze(0).to(device)
            out = model(c, labels=c)
            n = c.shape[1] - 1
            nll += float(out.loss) * n
            ntok += n
        return math.exp(nll / ntok)

    if kw is None:
        kw = capture_layer_kwargs(model, ids_list[0], device)
    hs = embed_batch(model, ids_list, device)
    model.model.embed_tokens.to("cpu")
    for layer in decoder_layers(model):
        layer.to(device)
        hs = stream_layer(layer, hs, kw, device, micro=micro)
        layer.to("cpu")
        sweep()
    ppl = stream_head(model, hs, ids_list, device, offload=True)
    del hs
    sweep()
    return ppl


# %% [markdown]
# ## Calibration pass — Hessians, Cholesky, whitened SVD
#
# One streamed pass over the decoder. Per layer:
#
# 1. accumulate `H = X Xᵀ` in **fp32** per input group (q/k/v share an input,
#    gate/up share an input — 4 Hessians per layer rather than 7);
# 2. damp `H += damp·mean(diag H)·I` and Cholesky in **fp64** → `L`, `H = L Lᵀ`;
# 3. per matrix: `Ws = W L`, SVD, and `Z = Vh L⁻¹` by triangular solve.
#
# Cached artefact per matrix is `(U, S, Z)`. Then for any rank `r`
#
# ```
# W_r = U_r S_r Vh_r L⁻¹ = U_r S_r Z_r
# ```
#
# which is *exactly* the CPU scripts' `solve_triangular(Lᵀ, W_rᵀ, upper=True)ᵀ`,
# because selecting the first `r` rows commutes with right-multiplication by
# `L⁻¹`. Storing `Z` instead of `L` is 4× smaller for `down_proj`.

# %%
def svd_of(M: torch.Tensor, fast: bool = False):
    """Reduced SVD. `fast=True` goes via the Gram matrix + eigh, which is much
    quicker for very rectangular matrices but squares the condition number —
    accurate for the retained (large) singular values, noisy in the discarded
    tail. Off by default."""
    if not fast:
        return torch.linalg.svd(M, full_matrices=False)
    m, n = M.shape
    if m <= n:
        ev, U = torch.linalg.eigh((M @ M.T).double())
        idx = torch.argsort(ev, descending=True)
        S = ev[idx].clamp_min(0).sqrt().float()
        U = U[:, idx].float()
        Vh = (U.T @ M) / S.clamp_min(1e-12).unsqueeze(1)
        return U, S, Vh
    ev, V = torch.linalg.eigh((M.T @ M).double())
    idx = torch.argsort(ev, descending=True)
    S = ev[idx].clamp_min(0).sqrt().float()
    V = V[:, idx].float()
    U = (M @ V) / S.clamp_min(1e-12).unsqueeze(0)
    return U, S, V.T


def cholesky_damped(H32: torch.Tensor, damp: float, fp64: bool = True):
    """H + damp·mean(diag H)·I = L Lᵀ, escalating the damping on failure.
    Identical damping rule to the CPU scripts, done in fp64 throughout."""
    Hd = H32.double() if fp64 else H32.clone()
    del H32                      # the caller passed the only reference; free it now
    mean_diag = torch.diagonal(Hd).mean()
    if not torch.isfinite(mean_diag) or mean_diag <= 0:
        raise RuntimeError(f"degenerate Hessian: mean(diag) = {float(mean_diag)}")
    applied = 0.0
    for k in range(6):
        want = damp * (2 ** k)
        Hd.diagonal().add_((want - applied) * mean_diag)
        applied = want
        L, info = torch.linalg.cholesky_ex(Hd)
        if int(info) == 0:
            del Hd
            return L, want
        del L
    raise RuntimeError(f"Cholesky failed up to damping {applied:g}")


def _sanity_on_matrix(W: torch.Tensor, Lf: torch.Tensor, S: torch.Tensor,
                      r: int) -> Dict[str, float]:
    """Protocol check in the activation metric, at equal rank.

    ‖(W − W_r)X‖_F² = ‖(W − W_r)L‖_F², and for the activation-aware truncation
    that is exactly Σ_{j>r} σ_j² of the whitened matrix. Compare against plain
    SVD and against the Lᵀ bug. Required ordering: L < Lᵀ < plain."""
    err_act = float(S.double()[r:].pow(2).sum())

    Up, Sp, Vhp = torch.linalg.svd(W, full_matrices=False)
    Wp = (Up[:, :r] * Sp[:r]) @ Vhp[:r]
    err_plain = float(((W - Wp) @ Lf).double().pow(2).sum())
    del Up, Sp, Vhp, Wp

    Ut, St, Vht = torch.linalg.svd(W @ Lf.T, full_matrices=False)
    Yt = (Ut[:, :r] * St[:r]) @ Vht[:r]
    Wt = torch.linalg.solve_triangular(Lf, Yt.T, upper=False).T   # Y (Lᵀ)⁻¹
    err_lt = float(((W - Wt) @ Lf).double().pow(2).sum())
    del Ut, St, Vht, Yt, Wt

    return {"rank": r, "err_act_aware_L": err_act,
            "err_act_aware_LT": err_lt, "err_plain_svd": err_plain}


@torch.no_grad()
def run_calibration(cfg: Cfg, cache: Cache, calib_ids, device):
    """Build and cache (U, S, Z) for every decoder linear. Idempotent: returns
    immediately if the manifest for this exact config hash is already present."""
    if cache.has("manifest"):
        man = cache.get("manifest")
        log(f"factors already cached for {cache.key} "
            f"({len(man['shapes'])} matrices) — skipping calibration pass")
        return man

    fdt = getattr(torch, cfg.factor_dtype)
    log(f"calibration pass: {len(calib_ids)} x {cfg.seqlen} tokens, "
        f"damp={cfg.damp}, fp64 Cholesky={cfg.chol_fp64}, fast_svd={cfg.fast_svd}")
    log(f"  scratch free: {cache.disk_free_gb():.1f} GiB")

    model = load_model(cfg, device="cpu")
    kw = capture_layer_kwargs(model, calib_ids[0], device)
    hs = embed_batch(model, calib_ids, device)
    model.model.embed_tokens.to("cpu")

    shapes: Dict[str, List[int]] = {}
    svals: Dict[str, torch.Tensor] = {}
    sanity: Dict[str, Dict[str, float]] = {}
    layers = decoder_layers(model)

    for i, layer in enumerate(layers):
        t_layer = time.time()
        layer.to(device)
        lins = [(n, m) for n, m in layer.named_modules() if isinstance(m, nn.Linear)]

        H: Dict[str, torch.Tensor] = {}
        seen: Dict[str, Tuple[int, tuple]] = {}
        handles = []

        def mk(path):
            gk = group_name(path)

            def f(mod, inp, out):
                x = inp[0]
                fp = (x.data_ptr(), tuple(x.shape))
                prev = seen.get(gk)
                if prev is not None:
                    assert prev == fp, (
                        f"layer {i} group '{gk}': members received different "
                        "input tensors. SHARED_INPUT_GROUP is wrong for this "
                        "architecture — set it to identity to get one Hessian "
                        "per matrix.")
                    return
                seen[gk] = fp
                xf = x.detach().reshape(-1, x.shape[-1]).float()
                if gk not in H:
                    H[gk] = torch.zeros(xf.shape[1], xf.shape[1],
                                        dtype=torch.float32, device=xf.device)
                H[gk].addmm_(xf.T, xf)
            return f

        for path, mod in lins:
            handles.append(mod.register_forward_hook(mk(path)))

        for b in range(hs.shape[0]):
            seen.clear()
            hs[b:b + 1] = _layer_out(layer(hs[b:b + 1], **kw))
        for h in handles:
            h.remove()

        # ---- Cholesky per group, then factors per matrix ----------------------
        for gk in list(H.keys()):
            L64, used_damp = cholesky_damped(H.pop(gk), cfg.damp, cfg.chol_fp64)
            Lf = L64.float()
            del L64
            sweep()

            if cfg.cache_cholesky:
                cache.put(f"chol/{i}.{gk}",
                          {"L": Lf.to(getattr(torch, cfg.factor_dtype)).cpu(),
                           "damp": used_damp}, big=True)

            members = [(p, m) for p, m in lins if group_name(p) == gk]
            for path, mod in members:
                key = f"{i}.{path}"
                W = mod.weight.data.float()
                m_, n_ = W.shape
                assert Lf.shape[0] == n_, (
                    f"{key}: Cholesky factor is {tuple(Lf.shape)} but the matrix "
                    f"expects {n_} inputs")
                U, S, Vh = svd_of(W @ Lf, cfg.fast_svd)
                Z = torch.linalg.solve_triangular(Lf.T, Vh.T, upper=True).T
                del Vh

                if i in cfg.sanity_layers and path in cfg.sanity_paths:
                    r = max(1, int(round(cfg.sanity_ratio * min(m_, n_))))
                    sanity[key] = _sanity_on_matrix(W, Lf, S, r)
                    s = sanity[key]
                    log(f"  sanity {key} r={r}: L={s['err_act_aware_L']:.4e} "
                        f"LT={s['err_act_aware_LT']:.4e} "
                        f"plain={s['err_plain_svd']:.4e}")
                    if cfg.enforce_sanity:
                        assert s["err_act_aware_L"] <= s["err_plain_svd"], (
                            f"{key}: activation-aware does NOT beat plain SVD at "
                            "equal rank — the whitening factor is wrong")
                        assert s["err_act_aware_L"] <= s["err_act_aware_LT"], (
                            f"{key}: whitening with L is worse than with Lᵀ — "
                            "the Cholesky convention has flipped")

                cache.put(f"fac/{key}", {
                    "U": U.to(fdt).cpu(), "S": S.float().cpu(), "Z": Z.to(fdt).cpu(),
                    "m": m_, "n": n_, "damp": used_damp,
                }, big=True)
                shapes[key] = [m_, n_]
                svals[key] = S.float().cpu()
                del U, S, Z, W
                sweep()
            del Lf
            sweep()

        layer.to("cpu")
        sweep()
        log(f"  layer {i:3d}/{len(layers)} factors done "
            f"({time.time() - t_layer:.1f}s, scratch free {cache.disk_free_gb():.1f} GiB)")

    man = {"key": cache.key, "model": cfg.model_id, "shapes": shapes,
           "sanity": sanity, "cfg": asdict(cfg)}
    cache.put("svals", svals)
    cache.put("manifest", man)
    del hs, model
    sweep()
    log(f"calibration complete: {len(shapes)} matrices")
    return man


# %% [markdown]
# ## Rank allocation and the effort ratio
#
# ```
# ρ = Σᵢ min(rᵢ(mᵢ+nᵢ), mᵢnᵢ) / Σᵢ mᵢnᵢ
# ```
#
# The `min` is the break-even check: a rank-`r` factorisation only saves FLOPs
# when `r < mn/(m+n)`. Above that the adversary keeps the matrix dense and pays
# `mn`. Note the allocator still *spends* `r(m+n)` from its budget for such a
# matrix, exactly as `exp1a_prime.py` did, so the realised `ρ` can land slightly
# under the target.

# %%
def uniform_ranks(shapes: Dict[str, List[int]], ratio: float) -> Dict[str, int]:
    return {k: max(1, int(round(ratio * min(m_, n_)))) for k, (m_, n_) in shapes.items()}


class Allocation(NamedTuple):
    """What the allocator decided, and whether it could honour the budget.

    The floor is spent before any budget check, so when `floor_cost > budget`
    every greedy increment is skipped and `ranks` is the bare floor — an
    allocation that costs MORE than was asked for. That is not an operating
    point at the requested effort and must not be reported as one, so the fact
    travels with the result instead of being recomputed (or forgotten) by the
    caller."""
    ranks: Dict[str, int]
    spent: float            # flops the returned allocation actually costs
    floor_cost: float       # flops the per-matrix floor alone costs
    budget: float           # the budget it was asked to respect

    @property
    def feasible(self) -> bool:
        return self.floor_cost <= self.budget

    @property
    def floor_budget_frac(self) -> float:
        """Floor cost as a fraction of budget. > 1 means infeasible."""
        return self.floor_cost / self.budget if self.budget else float("inf")


def greedy_ranks(svals: Dict[str, torch.Tensor], shapes: Dict[str, List[int]],
                 budget: float, floor: float = 0.0) -> Allocation:
    """Exact greedy for the separable surrogate.

    error_i(r_i) = Σ_{j>r_i} σ_ij²      flops_i(r_i) = r_i (m_i + n_i)
    minimise Σ error_i  s.t.  Σ flops_i ≤ B

    Each rank increment at matrix i costs (m_i+n_i) and removes σ_ij², so
    sorting all increments by σ_ij²/(m_i+n_i) descending is optimal. `floor` is
    a per-matrix minimum rank: without it the greedy starves whole matrices to
    rank ~0, which is catastrophic end-to-end because error compounds through
    depth even when the layer-local error removed is small.

    A floor large enough to exceed `budget` on its own is NOT clamped or
    rescaled — doing so would invent an operating point the allocator never
    chose. The bare floor is returned and flagged infeasible via the result, so
    a sweep can walk past it and record where feasibility breaks."""
    rank = {k: max(1, int(round(floor * min(m_, n_)))) for k, (m_, n_) in shapes.items()}
    spent = sum(rank[k] * (shapes[k][0] + shapes[k][1]) for k in shapes)
    floor_cost = spent          # what the floor alone costs, before any greedy
    cand = []
    for k, (m_, n_) in shapes.items():
        w = m_ + n_
        sv2 = (svals[k].double() ** 2).tolist()
        for j in range(rank[k], len(sv2)):
            cand.append((sv2[j] / w, k, j))
    cand.sort(reverse=True)
    for _gain, k, _j in cand:
        w = shapes[k][0] + shapes[k][1]
        if spent + w > budget:
            continue
        rank[k] += 1
        spent += w
    return Allocation(rank, spent, floor_cost, budget)


def rho_of(ranks: Dict[str, int], shapes: Dict[str, List[int]]) -> float:
    cost = sum(min(ranks[k] * (m_ + n_), m_ * n_) for k, (m_, n_) in shapes.items())
    dense = sum(m_ * n_ for m_, n_ in shapes.values())
    return cost / dense


def dense_params(shapes) -> int:
    return sum(m_ * n_ for m_, n_ in shapes.values())


# %% [markdown]
# ## Materialising the truncation
#
# A truncated matrix becomes two real `nn.Linear`s, so the FLOP saving counted
# by `ρ` is the FLOP saving the model actually performs. When the break-even
# check fails the matrix stays dense and keeps its **original** weight
# (`dense_fallback="original"`, the `exp1a_prime.py` convention). Serving a
# truncated-but-dense matrix at full price, as `exp1a_actaware.py` did, is
# strictly dominated; set `dense_fallback="truncated"` to reproduce it.

# %%
class LowRankLinear(nn.Module):
    """W ≈ A B with A: (m, r), B: (r, n). Cost r(m+n) instead of mn."""

    def __init__(self, A: torch.Tensor, B: torch.Tensor,
                 bias: Optional[torch.Tensor], dtype):
        super().__init__()
        m_, r = A.shape
        r2, n_ = B.shape
        assert r == r2, (A.shape, B.shape)
        self.in_features, self.out_features, self.rank = n_, m_, r
        self.down = nn.Linear(n_, r, bias=False, dtype=dtype)
        self.up = nn.Linear(r, m_, bias=bias is not None, dtype=dtype)
        self.down.weight.data.copy_(B.to(dtype))
        self.up.weight.data.copy_(A.to(dtype))
        if bias is not None:
            self.up.bias.data.copy_(bias.detach().to(dtype))

    def forward(self, x):
        return self.up(self.down(x))

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, rank={self.rank}"


def plain_factors(key: str, W_cpu: torch.Tensor, cache: Cache, cfg: Cfg, device):
    name = f"plain/{key}"
    if cfg.cache_plain and cache.has(name):
        d = cache.get(name)
        return d["U"].float(), d["S"].float(), d["Vh"].float()
    Wg = W_cpu.to(device).float()
    U, S, Vh = svd_of(Wg, cfg.fast_svd)
    U, S, Vh = U.cpu().float(), S.cpu().float(), Vh.cpu().float()
    del Wg
    sweep()
    if cfg.cache_plain:
        fdt = getattr(torch, cfg.factor_dtype)
        cache.put(name, {"U": U.to(fdt), "S": S, "Vh": Vh.to(fdt)}, big=True)
    return U, S, Vh


def factors_at(key: str, r: int, method: str, mod: nn.Linear,
               cache: Cache, cfg: Cfg, device):
    """(A, B) in fp32 on CPU such that W_r = A @ B at rank r."""
    if method == "plain_svd":
        U, S, Vh = plain_factors(key, mod.weight.data, cache, cfg, device)
        return U[:, :r].contiguous(), (S[:r].unsqueeze(1) * Vh[:r]).contiguous()
    d = cache.get(f"fac/{key}")
    U, S, Z = d["U"].float(), d["S"].float(), d["Z"].float()
    return U[:, :r].contiguous(), (S[:r].unsqueeze(1) * Z[:r]).contiguous()


@torch.no_grad()
def build_truncated(cfg: Cfg, cache: Cache, shapes, ranks: Dict[str, int],
                    method: str, device, model=None):
    """Returns (model, rho, stats). The model is left on CPU.

    Modules are fetched lazily rather than held in a list: keeping references to
    the originals while replacing them would pin a second copy of every decoder
    weight in RAM, which a 7B cannot afford."""
    if model is None:
        model = load_model(cfg, device="cpu")
    meta = [(key, i, path, mod.out_features, mod.in_features)
            for key, i, path, mod in target_linears(model)]
    total_dense = dense_params(shapes)
    cost = 0
    n_factored = n_kept_dense = 0
    keep_fracs = []

    for key, i, path, m_, n_ in meta:
        layer = decoder_layers(model)[i]
        mod = get_submodule(layer, path)
        r = max(1, min(int(ranks[key]), min(m_, n_)))
        keep_fracs.append(r / min(m_, n_))

        if r * (m_ + n_) >= m_ * n_:                    # factorisation not worth it
            cost += m_ * n_
            n_kept_dense += 1
            if cfg.dense_fallback == "truncated":
                A, B = factors_at(key, r, method, mod, cache, cfg, device)
                Wr = (A.to(device) @ B.to(device)).to(mod.weight.dtype).cpu()
                mod.weight.data.copy_(Wr)
                del A, B, Wr
                sweep()
            del mod
            continue

        A, B = factors_at(key, r, method, mod, cache, cfg, device)
        bias = mod.bias.detach().clone() if mod.bias is not None else None
        del mod
        set_submodule(layer, path, LowRankLinear(A, B, bias, DTYPE))
        cost += r * (m_ + n_)
        n_factored += 1
        del A, B, bias
        sweep()

    rho = cost / total_dense
    stats = {
        "n_factored": n_factored, "n_kept_dense": n_kept_dense,
        "keep_frac_min": min(keep_fracs), "keep_frac_max": max(keep_fracs),
        "keep_frac_mean": sum(keep_fracs) / len(keep_fracs),
    }
    return model, rho, stats


def truncated_linear_names(model) -> List[str]:
    """Fully-qualified names of every nn.Linear inside the decoder after
    truncation — the LoRA target set."""
    names = []
    for i, layer in enumerate(decoder_layers(model)):
        for n, m in layer.named_modules():
            if isinstance(m, nn.Linear):
                names.append(f"model.layers.{i}.{n}")
    return names


# %% [markdown]
# ## Exp 1b — LoRA distillation against the full model
#
# The truncated model is the student; the untruncated model at the same
# checkpoint is the teacher. Objective is KL on logits.
#
# **Why the teacher's logits are cached.** At 7B, teacher and student cannot be
# resident together in 15 GB (2 × 15.2 GB of weights before anything else). The
# teacher is therefore run **once**, layer-streamed, over the fixed pool of
# distillation sequences, and its output distribution is stored as the top-`K`
# logits plus the exact full-vocabulary `logsumexp`. The training KL is then
# taken over the partition {`K` individual classes} ∪ {everything else, lumped}:
#
# ```
# KL = Σ_{i∈topK} p_i log(p_i/q_i)  +  p_tail log(p_tail/q_tail)
# ```
#
# This is a genuine KL between coarsened distributions and a lower bound on the
# full-vocabulary KL (log-sum inequality). `validate_topk_kl` below measures the
# gap against exact full-vocab KL on a handful of batches.
#
# **ρ is unaffected by distillation.** LoRA adapters sit on the two factor
# matrices; merging them changes no shape. Evaluation runs with adapters active
# and dropout off, which is exactly equal to evaluating the merged weights.

# %%
def teacher_cache_key(cfg: Cfg) -> str:
    h = hashlib.sha1(json.dumps({
        "model": cfg.model_id, "ds": cfg.dataset, "cfg": cfg.dataset_config,
        "ndistil": cfg.ndistil, "L": cfg.distil_seqlen, "K": cfg.topk,
        "ncalib": cfg.ncalib, "seqlen": cfg.seqlen,   # fixes the disjoint offset
    }, sort_keys=True).encode()).hexdigest()[:12]
    return f"teacher.{cfg.model_id.replace('/', '__')}.{h}"


@torch.no_grad()
def build_teacher_cache(cfg: Cfg, distil_ids, device, cache: Optional[Cache] = None):
    cache = cache or Cache(cfg, key=teacher_cache_key(cfg))
    if cache.has("topk"):
        log("teacher logits already cached")
        return cache.get("topk")

    log(f"teacher pass: {len(distil_ids)} x {cfg.distil_seqlen} tokens, top-{cfg.topk}")
    model = load_model(cfg, device="cpu")
    kw = capture_layer_kwargs(model, distil_ids[0], device)
    hs = embed_batch(model, distil_ids, device)
    model.model.embed_tokens.to("cpu")
    for i, layer in enumerate(decoder_layers(model)):
        layer.to(device)
        for b in range(hs.shape[0]):
            hs[b:b + 1] = _layer_out(layer(hs[b:b + 1], **kw))
        layer.to("cpu")
        sweep()

    norm = model.model.norm.to(device)
    head = model.lm_head.to(device)
    N, L = hs.shape[0], hs.shape[1]
    vals = torch.empty(N, L, cfg.topk, dtype=torch.float16)
    idxs = torch.empty(N, L, cfg.topk, dtype=torch.int32)
    lse = torch.empty(N, L, dtype=torch.float32)
    for b in range(N):
        logits = head(norm(hs[b:b + 1])).float()[0]        # (L, V)
        lse[b] = torch.logsumexp(logits, dim=-1).cpu()
        v, ix = torch.topk(logits, cfg.topk, dim=-1)
        vals[b] = v.half().cpu()
        idxs[b] = ix.int().cpu()
        del logits, v, ix
    norm.to("cpu")
    head.to("cpu")
    del hs, model
    sweep()

    obj = {"vals": vals, "idx": idxs, "lse": lse, "topk": cfg.topk,
           "seqlen": cfg.distil_seqlen, "n": N}
    cache.put("topk", obj, big=True)
    log(f"teacher cache: {vals.numel() * 2 / 2**20:.0f} MiB values + "
        f"{idxs.numel() * 4 / 2**20:.0f} MiB indices")
    return obj


def topk_kl(student_logits: torch.Tensor, t_idx: torch.Tensor,
            t_val: torch.Tensor, t_lse: torch.Tensor, eps: float = 1e-9):
    """KL(teacher ‖ student) over {top-K classes} ∪ {lumped tail}."""
    sl = student_logits.float()
    s_lse = torch.logsumexp(sl, dim=-1)                       # (B, L)
    s_top = torch.gather(sl, -1, t_idx.long())                # (B, L, K)
    p = torch.exp(t_val.float() - t_lse.unsqueeze(-1))
    q = torch.exp(s_top - s_lse.unsqueeze(-1))
    p_tail = (1.0 - p.sum(-1)).clamp_min(eps)
    q_tail = (1.0 - q.sum(-1)).clamp_min(eps)
    kl = (p * (p.clamp_min(eps).log() - q.clamp_min(eps).log())).sum(-1)
    kl = kl + p_tail * (p_tail.log() - q_tail.log())
    return kl.mean()


def exact_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor):
    return F.kl_div(F.log_softmax(student_logits.float(), -1),
                    F.log_softmax(teacher_logits.float(), -1),
                    log_target=True, reduction="batchmean") / student_logits.shape[1]


@torch.no_grad()
def validate_topk_kl(cfg: Cfg, cache: Cache, shapes, svals, distil_ids,
                     device, nbatch: int = 4, ratio: float = 0.5):
    """How much does the top-K coarsening cost? Compare top-K KL against exact
    full-vocab KL between a truncated student and the full teacher, on a model
    small enough for both to be resident."""
    teacher = load_model(cfg, device=device)
    ranks = uniform_ranks(shapes, ratio)
    student, rho, _ = build_truncated(cfg, cache, shapes, ranks, "act_aware_uniform", device)
    student.to(device)
    tc = build_teacher_cache(cfg, distil_ids, device)
    rows = []
    for b in range(min(nbatch, len(distil_ids))):
        ids = distil_ids[b].unsqueeze(0).to(device)
        s_logits = student(ids).logits
        t_logits = teacher(ids).logits
        approx = float(topk_kl(s_logits,
                               tc["idx"][b:b + 1].to(device),
                               tc["vals"][b:b + 1].to(device),
                               tc["lse"][b:b + 1].to(device)))
        exact = float(exact_kl(s_logits, t_logits))
        rows.append({"batch": b, "topk_kl": approx, "exact_kl": exact,
                     "ratio": approx / exact if exact else float("nan")})
        log(f"  batch {b}: top-{cfg.topk} KL {approx:.4f} vs exact {exact:.4f} "
            f"({100 * approx / exact:.1f}%)")
        del s_logits, t_logits
        sweep()
    del teacher, student
    sweep()
    return {"rho": rho, "ratio": ratio, "batches": rows}


def adapter_root(cfg: Cfg) -> Optional[str]:
    """Where adapters go. Drive, never scratch: scratch dies with the runtime,
    and an adapter that does not outlive the session is worth nothing."""
    if cfg.adapter_dir:
        return cfg.adapter_dir
    d = mount_drive(cfg)
    return os.path.join(d, "adapters") if d else None


def save_adapter_checkpoint(pm, step: int, cfg: Cfg, cache: Cache, method: str,
                            ratio: float, floor: Optional[float],
                            ranks: Dict[str, int], rho: float) -> Optional[str]:
    """Save the LoRA adapter at one B_calib checkpoint, with the manifest needed
    to rebuild the model around it.

    An adapter alone reconstructs nothing. It sits on top of a *truncated* base,
    which is determined by the factor cache (`calib_key`) and the exact rank
    assignment. Both are recorded here, because in three weeks nobody remembers
    which arm an orphaned adapter came from.

    `pm.save_pretrained` writes adapter tensors only -- ~40 MB against ~3 GB for
    merged weights.
    """
    root = adapter_root(cfg)
    if root is None:
        log("  adapter checkpoint skipped: no Drive mounted")
        return None
    fl = "none" if floor is None else f"{floor}"
    path = os.path.join(root, cache.key,
                        f"{method}_keep{ratio}_floor{fl}", f"B{step}")
    os.makedirs(path, exist_ok=True)
    pm.save_pretrained(path)
    manifest = {
        "calib_key": cache.key,
        "model_id": cfg.model_id,
        "method": method,
        "keep_ratio": ratio,
        "floor": floor,
        "B_calib": step,
        "rho": rho,
        "ranks": {k: int(v) for k, v in ranks.items()},
        "lora": {"r": cfg.lora_r, "alpha": cfg.lora_alpha,
                 "dropout": cfg.lora_dropout, "lr": cfg.lr},
        "distil": {"ndistil": cfg.ndistil, "distil_seqlen": cfg.distil_seqlen,
                   "topk": cfg.topk},
        "saved": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": ("LoRA adapter only. Rebuild the truncated base from the factor "
                 "cache under calib_key using this exact ranks dict, then apply "
                 "this adapter. The adapter is meaningless on the dense model."),
    }
    with open(os.path.join(path, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    log(f"    adapter checkpoint -> {path}")
    return path


def lora_distil(cfg: Cfg, student, teacher_obj, distil_ids, eval_ids, device,
                eval_at: List[int], ckpt: Optional[Dict] = None):
    """Distil `student` against the cached teacher. Returns {steps: ppl}, with
    perplexity measured at each checkpoint in `eval_at` (adapters active)."""
    from peft import LoraConfig, get_peft_model

    student.config.use_cache = False
    if cfg.grad_checkpointing:
        student.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        student.enable_input_require_grads()
    student.to(device)

    targets = truncated_linear_names(student)
    lcfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                      lora_dropout=cfg.lora_dropout, bias="none",
                      target_modules=targets, task_type="CAUSAL_LM")
    pm = get_peft_model(student, lcfg)
    trainable = [p for p in pm.parameters() if p.requires_grad]
    log(f"  LoRA: {len(targets)} target modules, "
        f"{sum(p.numel() for p in trainable) / 1e6:.1f}M trainable params")

    opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=0.0)
    max_steps = max(eval_at)
    warm = max(1, int(0.05 * max_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm))

    t_val, t_idx, t_lse = teacher_obj["vals"], teacher_obj["idx"], teacher_obj["lse"]
    n_seq = min(len(distil_ids), t_val.shape[0])
    out: Dict[int, float] = {}
    pm.train()
    step = 0
    t_start = time.time()
    running = 0.0
    while step < max_steps:
        b = step % n_seq
        ids = distil_ids[b].unsqueeze(0).to(device)
        logits = pm(input_ids=ids).logits
        loss = topk_kl(logits, t_idx[b:b + 1].to(device),
                       t_val[b:b + 1].to(device), t_lse[b:b + 1].to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        running += float(loss)
        del logits, loss
        step += 1

        if step % 50 == 0:
            log(f"    step {step:4d}/{max_steps}  KL {running / 50:.4f}  "
                f"({(time.time() - t_start) / step:.2f}s/step)")
            running = 0.0
        if step in eval_at:
            pm.eval()
            with torch.no_grad():
                out[step] = perplexity(pm, eval_ids, device, streamed=False)
            log(f"    B_calib={step}: ppl {out[step]:.4f}")
            if ckpt is not None:
                save_adapter_checkpoint(pm, step, **ckpt)
            pm.train()
            sweep()

    pm.eval()
    return out, pm


# %% [markdown]
# ## Drivers
#
# `run_model` does the whole thing for one model: calibration pass, Exp 1a
# (uniform) for both adversaries, Exp 1a′ (greedy, at ρ matched to the uniform
# arm), then Exp 1b (LoRA distillation) for the selected methods. Results are
# accumulated into a flat row list and written to Drive after every row, so a
# disconnect never loses completed work.

# %%
def fit_resident(model, device, reserve_gb: float = 3.0) -> bool:
    """Move the model onto the GPU if it comfortably fits; otherwise leave it on
    CPU and let the streamed path handle it."""
    if device == "cpu":
        return False
    nbytes = sum(p.numel() * p.element_size() for p in model.parameters())
    free, _total = torch.cuda.mem_get_info()
    if nbytes + reserve_gb * 2**30 < free:
        model.to(device)
        return True
    log(f"  model is {nbytes / 2**30:.1f} GiB vs {free / 2**30:.1f} GiB free — "
        "using the layer-streamed path")
    return False


def preflight(cfg: Cfg):
    """Rough resource estimate before committing to a long run."""
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(cfg.model_id)
    h, ffn, nl = c.hidden_size, c.intermediate_size, c.num_hidden_layers
    kvh = getattr(c, "num_key_value_heads", c.num_attention_heads)
    hd = h // c.num_attention_heads
    mats = [(h, h), (kvh * hd, h), (kvh * hd, h), (h, h), (ffn, h), (ffn, h), (h, ffn)]
    fac = sum((m * min(m, n) + min(m, n) * n) for m, n in mats) * nl * 4 / 2**30
    dense = sum(m * n for m, n in mats) * nl
    log(f"preflight {cfg.model_id}: {nl} layers, hidden {h}, ffn {ffn}")
    log(f"  decoder linear params (ρ denominator): {dense / 1e9:.2f} B")
    log(f"  (U,S,Z) factor cache at fp32: ~{fac:.1f} GiB")
    log(f"  scratch free now: {shutil.disk_usage(cfg.scratch if os.path.isdir(cfg.scratch) else '/').free / 2**30:.1f} GiB")
    if fac > shutil.disk_usage("/").free / 2**30:
        log("  WARNING: factor cache may not fit on local disk. Reduce the run, "
            "or set cfg.factor_dtype='float16' (knowingly lossy), or accept "
            "recomputation.")
    return {"dense_params": dense, "factor_cache_gb": fac}


def run_model(cfg: Cfg, results: Dict, out_path: str, do_1b: bool = True):
    log("=" * 78)
    log(f"MODEL {cfg.model_id}")
    log("=" * 78)
    os.makedirs(cfg.scratch, exist_ok=True)
    preflight(cfg)

    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    calib_ids, eval_ids, distil_ids = make_splits(tok, cfg)
    cache = Cache(cfg)

    man = run_calibration(cfg, cache, calib_ids, DEV)
    shapes = man["shapes"]
    svals = cache.get("svals")

    entry = results.setdefault(cfg.model_id, {})
    entry["config"] = asdict(cfg)
    entry["sanity_matrix"] = man["sanity"]
    entry["dense_params"] = dense_params(shapes)
    rows: List[Dict] = entry.setdefault("rows", [])

    def flush():
        results["_meta"] = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "device": DEV, "weight_dtype": str(DTYPE),
                            "bf16_supported": BF16_OK}
        with open(out_path, "w") as f:
            json.dump(results, f, indent=1)

    # ---- baseline ------------------------------------------------------------
    base_model = load_model(cfg, device="cpu")
    fit_resident(base_model, DEV)
    base = perplexity(base_model, eval_ids, DEV)
    entry["base_ppl"] = base
    log(f"baseline ppl = {base:.4f}")
    del base_model
    sweep()
    flush()

    # ---- Exp 1a: uniform arms -----------------------------------------------
    plain_ratios = cfg.plain_svd_ratios if cfg.plain_svd_ratios is not None else cfg.ratios
    uniform_rho: Dict[float, float] = {}
    # keyed (method, ratio, floor); floor is None for the uniform arms
    ranks_by_cfg: Dict[Tuple[str, float, Optional[float]], Dict[str, int]] = {}

    for method in cfg.methods:
        if method == "act_aware_greedy":
            continue
        use = cfg.ratios if method != "plain_svd" else plain_ratios
        for ratio in use:
            ranks = uniform_ranks(shapes, ratio)
            model, rho, stats = build_truncated(cfg, cache, shapes, ranks, method, DEV)
            fit_resident(model, DEV)
            ppl = perplexity(model, eval_ids, DEV)
            if method == "act_aware_uniform":
                uniform_rho[ratio] = rho
            ranks_by_cfg[(method, ratio, None)] = ranks
            rows.append({
                "model": cfg.model_id, "method": method, "B_calib": 0,
                "keep_ratio": ratio, "target_rho": None, "rho": rho,
                "ppl": ppl, "base_ppl": base, "d_ppl_pct": 100 * (ppl - base) / base,
                "floor": None, **stats,
            })
            log(f"{method:18s} keep={ratio:<5} rho={rho:.4f} ppl={ppl:12.3f} "
                f"dppl={rows[-1]['d_ppl_pct']:+10.2f}%  dense_kept={stats['n_kept_dense']}")
            del model
            sweep()
            flush()

    # ---- protocol check: act-aware must dominate plain SVD at equal rank -----
    checks = []
    for ratio in plain_ratios:
        a = next((r for r in rows if r["method"] == "act_aware_uniform"
                  and r["keep_ratio"] == ratio and r["B_calib"] == 0), None)
        p = next((r for r in rows if r["method"] == "plain_svd"
                  and r["keep_ratio"] == ratio and r["B_calib"] == 0), None)
        if a and p:
            ok = a["ppl"] <= p["ppl"]
            checks.append({"keep_ratio": ratio, "act_aware_ppl": a["ppl"],
                           "plain_svd_ppl": p["ppl"], "ok": ok})
            log(f"protocol check keep={ratio}: act-aware {a['ppl']:.3f} vs "
                f"plain SVD {p['ppl']:.3f} -> {'OK' if ok else 'FAILED'}")
            if cfg.enforce_sanity and not ok and a["rho"] < 0.999:
                raise AssertionError(
                    f"activation-aware ({a['ppl']:.3f}) does not beat plain SVD "
                    f"({p['ppl']:.3f}) at keep={ratio}. The whitening factor is "
                    "wrong — H = L Lᵀ, whiten with L, not Lᵀ.")
    entry["sanity_end_to_end"] = checks
    flush()

    # ---- Exp 1a': greedy at ρ matched to the uniform arm ---------------------
    if "act_aware_greedy" in cfg.methods:
        dense_tot = dense_params(shapes)
        # `floor` is outside calib_key(), so sweeping it reuses the cached
        # factors: each extra floor costs one re-truncation plus one eval.
        floors = cfg.floors if cfg.floors else (cfg.floor,)
        if cfg.floors and cfg.floor not in cfg.floors:
            raise ValueError(
                f"cfg.floor={cfg.floor} is not in cfg.floors={cfg.floors}; Exp 1b "
                "distils the cfg.floor arm and would have no ranks to start from")
        for ratio in cfg.ratios:
            target = uniform_rho.get(ratio)
            if target is None:
                target = rho_of(uniform_ranks(shapes, ratio), shapes)
            for fl in floors:
                alloc = greedy_ranks(svals, shapes, target * dense_tot, floor=fl)
                ranks = alloc.ranks
                model, rho, stats = build_truncated(
                    cfg, cache, shapes, ranks, "act_aware_greedy", DEV)
                fit_resident(model, DEV)
                ppl = perplexity(model, eval_ids, DEV)
                ranks_by_cfg[("act_aware_greedy", ratio, fl)] = ranks
                rows.append({
                    "model": cfg.model_id, "method": "act_aware_greedy", "B_calib": 0,
                    "keep_ratio": ratio, "target_rho": target, "rho": rho,
                    "ppl": ppl, "base_ppl": base, "d_ppl_pct": 100 * (ppl - base) / base,
                    "floor": fl, "ranks": {k: int(v) for k, v in ranks.items()},
                    # the floor is spent before any budget check, so a large
                    # enough floor returns an allocation that BREAKS the budget
                    "budget_feasible": alloc.feasible,
                    "floor_budget_frac": alloc.floor_budget_frac,
                    **stats,
                })
                log(f"{'act_aware_greedy':18s} floor={fl:<5} target_rho={target:.4f} "
                    f"rho={rho:.4f} ppl={ppl:12.3f} "
                    f"dppl={rows[-1]['d_ppl_pct']:+10.2f}%  "
                    f"keep_frac mean={stats['keep_frac_mean']:.2f} "
                    f"min={stats['keep_frac_min']:.2f}")
                if not alloc.feasible:
                    log(f"  INFEASIBLE: floor={fl} alone costs "
                        f"{alloc.floor_budget_frac:.1%} of the budget for "
                        f"target_rho={target:.4f}; every greedy increment was "
                        f"skipped and this row is NOT an operating point at "
                        f"keep={ratio}")
                del model
                sweep()
                flush()

    # ---- Exp 1b: LoRA distillation ------------------------------------------
    eval_at = sorted(s for s in cfg.b_calib if s > 0)
    if do_1b and eval_at:
        teacher = build_teacher_cache(cfg, distil_ids, DEV)
        d_ratios = cfg.distil_ratios if cfg.distil_ratios is not None else cfg.ratios
        for method in cfg.distil_methods:
            for ratio in d_ratios:
                # a floor sweep produces one greedy arm per floor; distillation
                # always runs the primary cfg.floor arm so 1b stays comparable
                fl = cfg.floor if method == "act_aware_greedy" else None
                ranks = ranks_by_cfg.get((method, ratio, fl))
                if ranks is None:
                    log(f"  no rank assignment for ({method}, {ratio}, floor={fl}); skipping")
                    continue
                model, rho, stats = build_truncated(cfg, cache, shapes, ranks, method, DEV)
                log(f"distil {method} keep={ratio} rho={rho:.4f} "
                    f"for {max(eval_at)} steps")
                try:
                    ckpt = ({"cfg": cfg, "cache": cache, "method": method,
                             "ratio": ratio, "floor": fl, "ranks": ranks,
                             "rho": rho} if cfg.save_adapters else None)
                    ppls, pm = lora_distil(cfg, model, teacher, distil_ids,
                                           eval_ids, DEV, eval_at, ckpt=ckpt)
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower():
                        raise
                    log(f"  OOM during distillation at rho={rho:.3f}: {e}")
                    log("  -> lower cfg.distil_seqlen, enable cfg.grad_checkpointing, "
                        "or restrict cfg.distil_ratios to the smaller rho values")
                    del model
                    sweep()
                    continue
                for steps, ppl in sorted(ppls.items()):
                    rows.append({
                        "model": cfg.model_id, "method": method, "B_calib": steps,
                        "keep_ratio": ratio, "target_rho": uniform_rho.get(ratio),
                        "rho": rho, "ppl": ppl, "base_ppl": base,
                        "d_ppl_pct": 100 * (ppl - base) / base,
                        "floor": cfg.floor if method == "act_aware_greedy" else None,
                        "lora_r": cfg.lora_r, "lr": cfg.lr,
                        "distil_seqlen": cfg.distil_seqlen, "topk": cfg.topk,
                        **stats,
                    })
                    log(f"{method:18s} B_calib={steps:<4} rho={rho:.4f} "
                        f"ppl={ppl:12.3f} dppl={rows[-1]['d_ppl_pct']:+10.2f}%")
                del pm, model
                sweep()
                flush()

    flush()
    log(f"model done -> {out_path}")
    return results


# %% [markdown]
# ## Reporting

# %%
def row_infeasible(r: Dict) -> bool:
    """True when a row is not an operating point at its stated keep ratio.

    Primary signal is the allocator's own verdict. Rows written before
    `budget_feasible` existed do not carry it, so fall back to the observable
    symptom: effort actually spent exceeding the effort requested."""
    if r.get("budget_feasible") is False:
        return True
    target = r.get("target_rho")
    return target is not None and r.get("rho", 0.0) > target + 1e-9


def print_table(results: Dict):
    for model_id, entry in results.items():
        if model_id.startswith("_") or not isinstance(entry, dict) \
                or "rows" not in entry:
            continue
        base = entry.get("base_ppl")
        print(f"\n### {model_id}   baseline ppl = {base:.4f}")
        rows = entry.get("rows", [])
        keyf = lambda r: (r["method"], r["B_calib"],
                          r.get("floor") if r.get("floor") is not None else -1.0,
                          -r["rho"])
        print(f"{'method':20s} {'B_calib':>7s} {'keep':>6s} {'floor':>6s} "
              f"{'rho':>8s} {'ppl':>13s} {'d_ppl %':>12s} {'dense kept':>11s}")
        print("-" * 91)
        n_infeasible = 0
        for r in sorted(rows, key=keyf):
            fl = r.get("floor")
            bad = row_infeasible(r)
            n_infeasible += bad
            frac = r.get("floor_budget_frac")
            flag = ""
            if bad:
                over = f" floor={frac:.0%} of budget" if frac is not None else ""
                flag = f"   << NOT AN OPERATING POINT:{over or ' rho exceeds target'}"
            print(f"{r['method']:20s} {r['B_calib']:>7d} "
                  f"{(r.get('keep_ratio') or 0):>6.2f} "
                  f"{(f'{fl:.2f}' if fl is not None else '-'):>6s} "
                  f"{r['rho']:>8.4f} "
                  f"{r['ppl']:>13.3f} {r['d_ppl_pct']:>+12.2f} "
                  f"{r.get('n_kept_dense', 0):>11d}{flag}")
        if n_infeasible:
            print(f"\n  {n_infeasible} row(s) flagged: the rank floor alone "
                  "exceeded the FLOP budget, so the allocator returned the bare "
                  "floor and spent MORE than the target rho. Those rows are not "
                  "results at their stated keep ratio and must not be read as "
                  "operating points.")
        for c in entry.get("sanity_end_to_end", []):
            print(f"  protocol check keep={c['keep_ratio']}: act-aware "
                  f"{c['act_aware_ppl']:.3f} vs plain SVD {c['plain_svd_ppl']:.3f} "
                  f"-> {'OK' if c['ok'] else 'FAILED'}")


def markdown_table(results: Dict) -> str:
    out = []
    for model_id, entry in results.items():
        if model_id.startswith("_") or not isinstance(entry, dict) \
                or "rows" not in entry:
            continue
        out.append(f"\n**{model_id}** — baseline ppl {entry['base_ppl']:.4f}\n")
        out.append("| method | B_calib | keep | floor | rho | ppl | Δ ppl | budget |")
        out.append("|---|---|---|---|---|---|---|---|")
        keyf = lambda r: (r["method"], r["B_calib"],
                          r.get("floor") if r.get("floor") is not None else -1.0,
                          -r["rho"])
        any_bad = False
        for r in sorted(entry["rows"], key=keyf):
            fl = r.get("floor")
            bad = row_infeasible(r)
            any_bad |= bad
            frac = r.get("floor_budget_frac")
            if bad:
                note = (f"**INFEASIBLE** (floor = {frac:.0%} of budget)"
                        if frac is not None else "**INFEASIBLE** (rho > target)")
            else:
                note = "ok"
            out.append(f"| {r['method']} | {r['B_calib']} | "
                       f"{(r.get('keep_ratio') or 0):.2f} | "
                       f"{f'{fl:.2f}' if fl is not None else '—'} | {r['rho']:.4f} | "
                       f"{r['ppl']:.3f} | {r['d_ppl_pct']:+.2f}% | {note} |")
        if any_bad:
            out.append("")
            out.append("> Rows marked **INFEASIBLE** are not operating points at "
                       "their stated keep ratio: the per-matrix rank floor alone "
                       "exceeded the FLOP budget, so the allocator returned the "
                       "bare floor and spent more effort than the target ρ.")
    return "\n".join(out)


# %% [markdown]
# ## Smoke test — run this first
#
# Qwen2.5-0.5B, tiny budgets, every code path exercised (calibration, both
# adversaries, greedy allocation, both protocol checks, teacher cache, LoRA
# distillation). A few minutes on any GPU. If this passes, the real runs differ
# only in scale.

# %%
SMOKE = Cfg(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    seqlen=256, ncalib=8, neval=4, ndistil=16, distil_seqlen=128,
    ratios=(0.75, 0.5), b_calib=(0, 4, 8), topk=64,
    distil_ratios=(0.5,),
    scratch=DEFAULT_SMOKE_SCRATCH, drive=None,
    out_name="smoke.json",
)

if __name__ == "__main__":
    smoke_results: Dict = {}
    run_model(SMOKE, smoke_results,
              os.path.join(SMOKE.scratch, SMOKE.out_name), do_1b=True)
    print_table(smoke_results)


# %% [markdown]
# ## Optional: how much does the top-K KL coarsening cost?
#
# Only meaningful where teacher and student both fit resident, i.e. 1.5B or
# smaller. Reports top-K KL as a fraction of exact full-vocabulary KL.

# %%
def run_topk_validation(cfg: Cfg):
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    calib_ids, eval_ids, distil_ids = make_splits(tok, cfg)
    cache = Cache(cfg)
    man = run_calibration(cfg, cache, calib_ids, DEV)
    return validate_topk_kl(cfg, cache, man["shapes"], cache.get("svals"),
                            distil_ids, DEV)


# run_topk_validation(SMOKE)

# %% [markdown]
# ## Run 1 — Qwen2.5-1.5B-Instruct
#
# Comfortable on a 15 GB card: both models fit resident, so distillation uses
# the fast path and perplexity is measured without streaming.

# %%
RESULTS: Dict = {}
OUT = os.path.join(DEFAULT_DRIVE, "exp1_gpu_results.json") if DEFAULT_DRIVE \
    else os.path.join(DEFAULT_SCRATCH, "exp1_gpu_results.json")

CFG_1_5B = Cfg(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    seqlen=512, ncalib=128, neval=64,
    ndistil=512, distil_seqlen=256,
    ratios=(0.9, 0.75, 0.5, 0.3),
    floor=0.25,
    # Closes the floor-sensitivity caveat in findings-exp1a-prime-nonuniform.md:
    # a single arbitrary floor does not give a minimum of Phi(rho) over floors.
    # `floor` is outside calib_key(), so this reuses the cached factors and costs
    # only re-truncation + eval (4 ratios x 2 floors = 8 greedy arms, not 4).
    #
    # 0.1 was dropped after the 0.5B sweep: it starves matrices to near-zero rank
    # and was catastrophic at every rho tested -- +41101% d_ppl at rho=0.6321 and
    # +723% at rho=0.8042, against +973%/+295% for floor 0.25 at the same targets.
    # It was never the minimum, so it would cost 4 greedy arms of T4 time to
    # re-confirm a result already established at 0.5B.
    floors=(0.25, 0.4),
    b_calib=(0, 100, 300, 500),
    distil_methods=("act_aware_greedy",),
    distil_ratios=(0.75, 0.5, 0.3),
    cache_plain=True,
    # Drive mirroring off: the factor cache is 12.2 GiB with cache_plain=True,
    # which does not fit a 15 GB free Drive and would write at a few MB/s.
    # Results still go to Drive; only the regenerable factors stay local.
    drive_factors=False,
)

if __name__ == "__main__":
    if mount_drive(CFG_1_5B):
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
    else:                                   # no Drive (not on Colab) -> local file
        OUT = os.path.join(CFG_1_5B.scratch, CFG_1_5B.out_name)
        os.makedirs(CFG_1_5B.scratch, exist_ok=True)

    run_model(CFG_1_5B, RESULTS, OUT)
    print_table(RESULTS)

# %% [markdown]
# ## Run 2 — Qwen2.5-7B-Instruct
#
# Requirements, honestly: **≥ 16 GB system RAM** (the model lives on CPU between
# streamed layers — a free Colab T4 runtime at 12.7 GB cannot load it), ~10 GB
# VRAM for the streamed passes, and ~33 GB of local scratch for the factor
# cache. `drive_factors=False` because 33 GB will not fit a 15 GB Drive; results
# and singular values still go to Drive, so only the calibration pass is lost on
# a runtime restart.
#
# Distillation is restricted to the ρ values where the student is small enough
# to sit resident alongside the optimiser state. If it OOMs, the driver says so
# and moves on rather than dying.

# %%
CFG_7B = Cfg(
    model_id="Qwen/Qwen2.5-7B-Instruct",
    seqlen=512, ncalib=128, neval=64,
    ndistil=512, distil_seqlen=256,
    ratios=(0.9, 0.75, 0.5, 0.3),
    floor=0.25,
    # No floor sweep at 7B: each extra greedy arm re-reads ~31 GiB of factors
    # from disk and re-streams the eval, so 8 extra arms would add hours.
    floors=(),
    b_calib=(0, 100, 300, 500),
    plain_svd_ratios=(0.5,),      # baseline + protocol check only; SVD is costly here
    distil_methods=("act_aware_greedy",),
    distil_ratios=(0.5, 0.3),
    cache_plain=False,
    drive_factors=False,
    grad_checkpointing=True,
)

if __name__ == "__main__":
    run_model(CFG_7B, RESULTS, OUT)
    print_table(RESULTS)

# %% [markdown]
# ## Run 3 — improvement session (floor 0.4, adapters saved)
#
# Two jobs in one session, both on the arm Run 1 never distilled. Exp 1b only
# ever distils `cfg.floor`, which was 0.25, so floor 0.4 — **8.36× better in
# Δ ppl at `keep = 0.9`** (+28.68% vs +239.67%, at essentially the same `ρ`) —
# has never been combined with the distillation lever. Those are the two
# strongest levers in the experiment and they have never been composed.
#
# `CFG_1_5B` is untouched: it produced `results/exp1_gpu_results.json` and stays
# reproducible. These configs share every field that enters `calib_key()`, so
# they reuse the same factor cache, and `drive_factors=True` finally puts that
# cache somewhere a later `auditor.py` session can find it — `cache_plain=False`
# drops it from 12.2 to 6.1 GiB, which is what makes it fit a free Drive.
#
# **Which arm gets the convergence run.** `keep = 0.5` (`ρ = 0.6245`), because
# H5 as pre-registered can only be falsified at `ρ ≤ 0.7` and that is the only
# floor-0.4 arm in the region. `keep = 0.9` would look far more dramatic — an
# 11× improvement on +28.68% lands near +2.6% — but at `ρ = 0.8949` it cannot
# satisfy the criterion whatever it reaches.
#
# That asymmetry is itself the finding: **the `ρ` threshold, not the quality
# threshold, is doing all the work in H5.** No quality improvement at high `ρ`
# can falsify it, so the only live falsification path is improving quality at
# low `ρ` — which is exactly what the convergence run tests. Worth a paragraph
# in the write-up.
#
# Note that with `methods=("act_aware_greedy",)` the end-to-end act-aware-vs-
# plain-SVD protocol check records nothing (it needs both arms present). The
# matrix-level `L` vs `Lᵀ` check and `synthetic_whitening_check()` still run, so
# the whitening convention is still verified; the end-to-end assertion is not.

# %%
_IMPROVE_KW = dict(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    seqlen=512, ncalib=128, neval=64,      # identical to CFG_1_5B => same
    ndistil=512, distil_seqlen=256,        # calib_key => factors computed once
    floor=0.4, floors=(0.4,),
    methods=("act_aware_greedy",),         # uniform and plain SVD already measured
    distil_methods=("act_aware_greedy",),
    cache_plain=False,                     # 12.2 -> 6.1 GiB of factors
    drive_factors=True,                    # which then fits a free Drive
    save_adapters=True,                    # the point of the session
)

# Job A — the missing floor-0.4 distillation cells.
CFG_IMPROVE = Cfg(
    ratios=(0.9, 0.75),
    distil_ratios=(0.9, 0.75),
    b_calib=(0, 100, 300, 500),
    out_name="exp1_improve_results.json",
    **_IMPROVE_KW,
)

# Job B — the convergence question, on the one arm where H5 can be falsified.
# keep=0.5 is deliberately absent from Job A: this grid already contains 500, so
# running it in both would repeat 500 steps for nothing.
CFG_CONVERGE = Cfg(
    ratios=(0.5,),
    distil_ratios=(0.5,),
    b_calib=(0, 250, 500, 1000, 2000, 3000),
    out_name="exp1_converge_results.json",
    **_IMPROVE_KW,
)

# %%
if __name__ == "__main__":
    def _clear_stale_manifest(c: Cfg) -> None:
        """Run 1 wrote `manifest` and `svals` to Drive but NOT the factors
        (`drive_factors=False`, so `fac/*` is `big=True` and was gated out).
        These configs share Run 1's calib_key, so `run_calibration` would find
        that manifest, log "factors already cached -- skipping calibration
        pass", and then die in `build_truncated` on the first missing
        `fac/*.pt`. Drop the stale manifest so the pass actually re-runs; this
        time `drive_factors=True` mirrors the factors with it.
        """
        cache = Cache(c)
        if not cache.has("manifest"):
            return
        probe = next(iter(cache.get("manifest")["shapes"]))
        if cache.has(f"fac/{probe}"):
            log("factor cache is complete -- calibration will be skipped")
            return
        for base in (cache.scratch, cache.drive):
            if not base:
                continue
            p = os.path.join(base, Cache._fn("manifest"))
            if os.path.exists(p):
                os.remove(p)
                log(f"removed stale manifest (factors absent): {p}")

    def _out_for(c: Cfg) -> str:
        d = mount_drive(c)
        base = d if d else c.scratch
        os.makedirs(base, exist_ok=True)
        return os.path.join(base, c.out_name)

    # Separate results dicts: `run_model` keys its entry by model_id, and both
    # configs are the same model, so one shared dict would silently overwrite.
    _clear_stale_manifest(CFG_IMPROVE)

    IMPROVE: Dict = {}
    run_model(CFG_IMPROVE, IMPROVE, _out_for(CFG_IMPROVE))
    print_table(IMPROVE)

    CONVERGE: Dict = {}
    run_model(CFG_CONVERGE, CONVERGE, _out_for(CFG_CONVERGE))
    print_table(CONVERGE)

# %% [markdown]
# ## Final output

# %%
# The requested deliverable: one flat row per (model, method, B_calib, rho).
if __name__ == "__main__":
    RESULTS["table"] = [
        {"model": r["model"], "method": r["method"], "B_calib": r["B_calib"],
         "keep_ratio": r.get("keep_ratio"), "rho": r["rho"],
         "ppl": r["ppl"], "base_ppl": r["base_ppl"], "d_ppl_pct": r["d_ppl_pct"],
         "floor": r.get("floor"), "n_kept_dense": r.get("n_kept_dense")}
        for model_id, entry in RESULTS.items()
        if not model_id.startswith("_") and isinstance(entry, dict)
        for r in entry.get("rows", [])
    ]
    with open(OUT, "w") as f:
        json.dump(RESULTS, f, indent=1)
    print(markdown_table(RESULTS))
    print(f"\n{len(RESULTS['table'])} rows written to {OUT}")
