#!/usr/bin/env python
"""Targeted check of the Cfg.floors sweep. Runs on CPU, no GPU needed.

Why this exists
---------------
The floor sweep added in the `floors` commit is verified statically but has
never been executed. Running the ordinary smoke test to exercise it needs
~3.6 GiB of factor cache (act-aware + plain SVD), which does not fit on a
laptop with ~2 GiB free.

This script restricts `methods` to ("act_aware_greedy",). That drops the
plain-SVD arms entirely, so nothing new is written to the factor cache: the
act-aware (U, S, Z) factors already on disk are reused, because `floor` and
`methods` are both outside `Cfg.calib_key()`. Cost is a few hundred MB of
transient model memory and a couple of minutes.

What it does and does not check
-------------------------------
Checks:
  * the floors loop produces one greedy arm per (ratio, floor)
  * ranks_by_cfg is keyed (method, ratio, floor) consistently
  * Exp 1b resolves to the cfg.floor arm and no other
  * the synthetic whitening protocol check (module-level, always runs)

Does NOT check:
  * the end-to-end act-aware vs plain-SVD protocol check, which needs the
    plain_svd rows this script deliberately omits. That check is exercised
    by the ordinary smoke test and by Run 1 on a GPU.
  * the per-matrix L vs L^T check, which lives in the calibration pass and
    is skipped whenever the factors are already cached.
  * the guard rejecting a floors tuple that omits cfg.floor. Firing it needs
    a full calibration and baseline before run_model reaches the greedy
    block, which is not worth the time; it is checked by construction only.

Usage
-----
    .venv/bin/python validate_floors.py
    .venv/bin/python validate_floors.py --floors 0.4,0.55,0.7,0.85
    .venv/bin/python validate_floors.py --floors 0.4,0.55,0.7 --skip-1b

--floors   comma-separated rank floors to sweep   (default 0.1,0.25,0.4)
--ratios   comma-separated keep ratios            (default 0.75,0.5)
--floor    the floor Exp 1b distils; must be one of --floors. Defaults to
           0.25 when that is in the sweep, otherwise the first floor given.
--skip-1b  skip the LoRA distillation arm (~25 s) when only probing floors.

Unknown arguments are an error, not ignored.
"""
import argparse
import os
import pathlib


def _csv_floats(s: str):
    try:
        vals = tuple(float(x) for x in s.split(",") if x.strip())
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a comma-separated float list: {s!r}")
    if not vals:
        raise argparse.ArgumentTypeError("empty list")
    return vals


_ap = argparse.ArgumentParser(description="Check the Cfg.floors sweep on CPU.")
_ap.add_argument("--floors", type=_csv_floats, default=(0.1, 0.25, 0.4))
_ap.add_argument("--ratios", type=_csv_floats, default=(0.75, 0.5))
_ap.add_argument("--floor", type=float, default=None,
                 help="floor Exp 1b distils; must be in --floors")
_ap.add_argument("--skip-1b", action="store_true", dest="skip_1b")
ARGS = _ap.parse_args()          # unknown args -> SystemExit(2), never ignored

FLOORS = ARGS.floors
RATIOS = ARGS.ratios
if ARGS.floor is not None:
    PRIMARY = ARGS.floor
    if PRIMARY not in FLOORS:
        _ap.error(f"--floor {PRIMARY} is not in --floors {FLOORS}; Exp 1b would "
                  "have no ranks to start from")
else:
    PRIMARY = 0.25 if 0.25 in FLOORS else FLOORS[0]
print(f"floors={FLOORS}  ratios={RATIOS}  primary floor (Exp 1b)={PRIMARY}"
      f"{'  [1b skipped]' if ARGS.skip_1b else ''}\n")

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "exp1_gpu_src.py"

# Slice the paired script up to (not including) the smoke-test cell, so we get
# every definition without running the smoke config itself.
lines = SRC.read_text().split("\n")
cut = next(i for i, l in enumerate(lines) if l.startswith("# ## Smoke test"))
hdr = max(i for i in range(cut) if lines[i].startswith("# %% [markdown]"))
g = {"__name__": "effortgap_defs", "__file__": str(SRC)}
exec(compile("\n".join(lines[:hdr]), str(SRC), "exec"), g)

Cfg = g["Cfg"]
run_model, print_table = g["run_model"], g["print_table"]
DEFAULT_SMOKE_SCRATCH = g["DEFAULT_SMOKE_SCRATCH"]

# --- 1. guard, by construction only ------------------------------------------
# Not executed: reaching the guard inside run_model costs a full calibration and
# baseline pass. This only confirms the condition it tests is the intended one.
bad = Cfg(model_id="Qwen/Qwen2.5-0.5B-Instruct", floor=0.25, floors=(0.1, 0.4))
assert bad.floor not in bad.floors, "guard precondition is not what was intended"
assert "cfg.floor not in cfg.floors" in SRC.read_text(), "guard missing from source"
print("guard present in source; condition confirmed (not executed here)\n")

# --- 2. the real sweep -------------------------------------------------------
# seqlen/ncalib/damp/chol_fp64/fast_svd/factor_dtype match SMOKE exactly, so
# calib_key() resolves to the cached factors and calibration is skipped.
CFG = Cfg(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    seqlen=256, ncalib=8, neval=4, ndistil=16, distil_seqlen=128,
    ratios=RATIOS, b_calib=(0,) if ARGS.skip_1b else (0, 4), topk=64,
    methods=("act_aware_greedy",),          # no plain SVD -> no new cache writes
    floor=PRIMARY, floors=FLOORS,
    distil_methods=("act_aware_greedy",), distil_ratios=(0.5,),
    scratch=DEFAULT_SMOKE_SCRATCH, drive=None,
    out_name="floors_check.json",
)
print(f"calib_key = {CFG.calib_key()}")
cached = os.path.join(CFG.scratch, CFG.calib_key())
print(f"factors cached: {os.path.isdir(cached)}  ({cached})\n")

res = {}
run_model(CFG, res, os.path.join(CFG.scratch, CFG.out_name), do_1b=not ARGS.skip_1b)
print_table(res)

# --- 3. assert the shape of what came back -----------------------------------
rows = res[CFG.model_id]["rows"]
greedy0 = [r for r in rows if r["method"] == "act_aware_greedy" and r["B_calib"] == 0]
want = {(ratio, fl) for ratio in CFG.ratios for fl in CFG.floors}
got = {(r["keep_ratio"], r["floor"]) for r in greedy0}
assert got == want, f"\n  expected arms {sorted(want)}\n  got           {sorted(got)}"
print(f"\n{len(greedy0)} greedy arms = {len(CFG.ratios)} ratios x {len(CFG.floors)} floors  OK")

assert len({(r["keep_ratio"], r["floor"]) for r in greedy0}) == len(greedy0), \
    "duplicate (ratio, floor) arms — ranks_by_cfg keying is wrong"
print("every (ratio, floor) arm is distinct                        OK")

distil = [r for r in rows if r["B_calib"] > 0]
if ARGS.skip_1b:
    assert not distil, "--skip-1b was given but Exp 1b rows appeared"
    print("Exp 1b skipped by request                                   --")
else:
    assert distil, "Exp 1b produced no rows — the ranks_by_cfg lookup missed"
    stray = [r for r in distil if r["floor"] != CFG.floor]
    assert not stray, f"Exp 1b distilled a non-primary floor: {[r['floor'] for r in stray]}"
    print(f"Exp 1b distilled only floor={CFG.floor} ({len(distil)} rows)          OK")

# monotonicity is NOT asserted: a higher floor is not guaranteed to be better at
# every rho, and that is one of the things the sweep exists to measure.
by_floor = {}
for r in greedy0:
    by_floor.setdefault(r["floor"], []).append((r["rho"], r["d_ppl_pct"]))
print("\nd_ppl % by floor (this is the measurement, not an assertion):")
for fl in sorted(by_floor):
    pts = "  ".join(f"rho={rho:.4f} {d:+.1f}%" for rho, d in sorted(by_floor[fl]))
    print(f"  floor {fl:<5} {pts}")

print("\nFLOOR SWEEP VALIDATED")
