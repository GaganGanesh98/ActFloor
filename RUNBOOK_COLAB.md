# Runbook — Run 3 on a free Colab T4

Run 3 (`CFG_IMPROVE` + `CFG_CONVERGE`) is the floor-0.4 arm with distillation,
the run `findings-exp1-scale.md` names as the one most likely to falsify H5. It
needs a GPU, so it cannot run on the laptop. Everything below follows the
round-trip rule in `README.md`: code goes **to** the runtime, only
`results/*.json` comes **back**.

## 1. Runtime

colab.research.google.com → **New notebook** → Runtime → Change runtime type →
**T4 GPU**.

## 2. Code

Files pane (folder icon, left) → upload `exp1_gpu_src.py` from this repo. It
lands in `/content`. Do not upload or open `exp1_gpu.ipynb` in Colab.

## 3. Cells

```python
# cell 1 — deps (same spec as the file's own deps cell)
!pip -q install "transformers>=4.44" datasets accelerate peft safetensors
```

```python
# cell 2 — Drive, then Run 3. Results and adapters are flushed to Drive after
# every row/checkpoint, so a disconnect loses at most the row in progress.
from google.colab import drive
drive.mount("/content/drive")

import sys; sys.path.insert(0, "/content")
import exp1_gpu_src as E
E.run_improvement()
```

Importing the module runs no experiment cells (they are behind `__main__`);
`run_improvement()` runs Job A then Job B and nothing else.

The first thing it does is drop Run 1's stale factor manifest on Drive and
redo the calibration pass, because Run 1 never mirrored the factors. That is
expected and logged (`removed stale manifest (factors absent)`).

## 4. What to expect

* Disk: ~6.1 GiB of factors on Drive (`cache_plain=False`). Needs that much
  free in a 15 GB Drive.
* Time: Run 1 was ~80 min. Job A is 2 greedy arms + 2 × 500 distillation
  steps; Job B is one arm × 3 000 steps. Budget **3–4 hours**. Keep the tab
  open; free Colab disconnects idle sessions.
* If the session dies mid-Job B, re-run cell 2. Calibration is skipped (factors
  are on Drive now) and Job A re-runs quickly; Job B restarts from step 0 —
  the driver does not resume distillation from a checkpoint.

## 5. Bring results back

From Drive, `MyDrive/effortgap/`:

* `exp1_improve_results.json`
* `exp1_converge_results.json`

Copy both into `results/` here and commit them **unmodified, before any
analysis** (the same discipline as `a59182d`). Adapters stay on Drive under
`MyDrive/effortgap/adapters/` — `auditor.py` finds them there.

Never download the Colab notebook back over `exp1_gpu.ipynb`.

## Afterwards

* `auditor.run(E.CFG_IMPROVE, out="results/auditor_oracle_1.5B.json")` on the
  same Drive gives the oracle bound at 1.5B with the saved adapters. Budget
  several hours; restrict `methods`/`floors` if needed.
* 7B (Run 2) needs a high-RAM runtime (≥16 GB system RAM, ~33 GB scratch);
  free T4 cannot load it.
