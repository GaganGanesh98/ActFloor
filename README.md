# Effort-gap experiments

Thesis code for the effort-gap experiments (Exp 1a / 1a′ / 1b). The main
artefact is `exp1_gpu.ipynb`, paired with `exp1_gpu_src.py`.

## The two files are one program

`exp1_gpu.ipynb` and `exp1_gpu_src.py` hold the same code and are **paired with
jupytext** (`ipynb,_src.py:percent`, recorded in the notebook metadata). Edit
either one, then run:

```sh
.venv/bin/python -m jupytext --sync exp1_gpu_src.py
```

jupytext rewrites whichever file is older. Never hand-edit both — sync instead.

## Reproducing the smoke test (CPU, no GPU needed)

The default `python3` on this machine is 3.14 and PyTorch has no 3.14 wheels, so
the notebook runs in a Python 3.12 virtualenv registered as a Jupyter kernel.

```sh
cd "Thesis 2027"

# 1. venv on Python 3.12 (uv; use `brew install python@3.12` + `python3.12 -m venv .venv` if you have no uv)
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt

# 2. register it as a Jupyter kernel — the notebook already selects this kernel
.venv/bin/python -m ipykernel install --user \
    --name effortgap-venv --display-name "Python 3.12 (effortgap .venv)"

# 3a. run the smoke test as a notebook: open exp1_gpu.ipynb in VS Code and pick
#     the kernel "Python 3.12 (effortgap .venv)" (registered user-wide in step 2,
#     and already recorded in the notebook metadata).
#     For a browser frontend instead: uv pip install --python .venv/bin/python jupyterlab

# 3b. …or headless, straight from the paired script (stops before the 1.5B cell)
.venv/bin/python - <<'EOF'
import pathlib
src = pathlib.Path("exp1_gpu_src.py").read_text().split("\n")
cut = next(i for i, l in enumerate(src) if l.startswith("# ## Run 1 — Qwen2.5-1.5B"))
hdr = max(i for i in range(cut) if src[i].startswith("# %% [markdown]"))
exec(compile("\n".join(src[:hdr]), "exp1_gpu_src.py", "exec"))
EOF
```

Run the cells top-to-bottom through **“Smoke test — run this first”**. Do not
run the **Run 1 (1.5B)** or **Run 2 (7B)** cells on a machine without a CUDA
GPU — they are sized for a 15 GB card.

On this machine (M-series Air, CPU, float32) the smoke test takes **~4 minutes**
end to end, or ~2.5 minutes when the factor cache is already warm.

There is no test suite; the smoke test *is* the check. It asserts its own
correctness and will stop rather than report a wrong number:

* the synthetic whitening check (`act-aware(L) < act-aware(Lᵀ) < plain SVD`),
* the per-matrix check in the activation metric on layer 0,
* the end-to-end check that activation-aware beats plain SVD at equal rank.

All of these stay enabled (`enforce_sanity=True`). If one fires, the numbers it
prints are the bug report — do not loosen the tolerance.

Last verified run (Qwen2.5-0.5B, CPU, baseline ppl 23.7125):

```
method               B_calib   keep      rho           ppl      d_ppl %  dense kept
------------------------------------------------------------------------------------
act_aware_greedy           0   0.75   0.8117        93.746      +295.34         111
act_aware_greedy           0   0.50   0.6350       254.421      +972.94          23
act_aware_greedy           4   0.50   0.6350       198.691      +737.92          23
act_aware_greedy           8   0.50   0.6350       174.431      +635.61          23
act_aware_uniform          0   0.75   0.8997        50.050      +111.07          48
act_aware_uniform          0   0.50   0.6357       399.081     +1583.00          48
plain_svd                  0   0.75   0.8997     68061.755   +286929.37          48
plain_svd                  0   0.50   0.6357    463753.708  +1955637.65          48
  protocol check keep=0.75: act-aware 50.050 vs plain SVD 68061.755 -> OK
  protocol check keep=0.5: act-aware 399.081 vs plain SVD 463753.708 -> OK
```

The truncation rows are deterministic and should reproduce exactly. The
`B_calib>0` rows move by a percent or so between runs because LoRA dropout is
unseeded.

### Where the cache goes

Scratch paths resolve at import, in this order:

1. `$EFFORTGAP_SCRATCH` if set — use this to put the cache on an external disk:
   `EFFORTGAP_SCRATCH=/Volumes/ext/effortgap .venv/bin/python …`
2. `/content` when it exists and is writable — unchanged Colab behaviour;
   Google Drive mirroring also only switches on there,
3. otherwise a folder next to `exp1_gpu_src.py`, falling back to `~/.cache`.

**Disk is the binding constraint on a laptop.** The `(U, S, Z)` factor cache is
about **1.8 GiB for Qwen2.5-0.5B** (≈6.5 GiB at 1.5B, ≈33 GiB at 7B). The smoke
config therefore sets `cache_plain=False`, which stops the plain-SVD factors
being cached as well and roughly halves the footprint; plain SVD is recomputed
on demand instead. That is a caching decision only — it changes no number.
Keep ~2.5 GiB free before starting.

### Library-version fixes (no maths changed)

Two things in the original notebook broke against current wheels:

* `torch.logspace(..., generator=None)` — `logspace` is deterministic and never
  used the generator; torch 2.14 dropped the argument. Removed.
* `load_dataset("wikitext", ...)` — HuggingFace retired the bare, loading-script
  `wikitext` repo, so the hub now rejects the un-namespaced id. Changed to
  `Salesforce/wikitext`, which is the same corpus with the same
  `wikitext-2-raw-v1` config.

### CPU / no CUDA

With no CUDA device the notebook falls back to `DEV="cpu"` and float32, and says
so. This is deliberate and there is **no MPS path**: MPS has no float64, so
`cfg.chol_fp64=True` would silently downgrade the Cholesky and change the
numerics of the method. Hessians, Cholesky factors and SVDs stay in fp32/fp64
regardless of the weight dtype.
