#!/usr/bin/env bash
# Stage-1 re-run, 2026-08-18.  lam -> a -> b, unattended.
#
# Runs the three phases in order and re-anchors the design point from the
# calibration instead of leaving X0 at its shipped ANALYTIC values -- the step
# whose omission cost the first attempt (X0's lambda_tso=0.9 gives rho_emp_p95
# = 2.487 against a 1.5 limit, so the design point was infeasible and the
# filter would have stayed empty).
#
#   docs/tuning/RUNBOOK_rerun_2026-08.md
#   docs/daily_log/08_2026/2026-08-18_dso4_voltage_relief.md
set -u -o pipefail

cd "Z:/Python_Projekte/qOFO_GH" || exit 1

PY=F:/python_environments/qOFO_clean/python.exe
OUT=results/tuning_mc/stage1
LIMITS=tuning_mc/configs/limits_mc_v2_tier1.json

export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
# BLAS=1 per worker, and W = 20 = the physical core count.
#
# NOT more.  Workers are single-threaded (BLAS pinned to 1, Gurobi Threads=1),
# so wall time is  ceil(N/W) * T * max(1, W/20):  past W=20 every worker just
# runs proportionally slower without reducing the batch count.  For phase A
# (N=29):  W=20 -> 2.00 T,  W=24 -> 2.40 T,  W=29 -> 1.45 T.  So 24 is *slower*
# than 20, and only matching W to the task count (29) would help -- which costs
# more cores than this run is allowed.  For phase B (16 points/poll) anything
# from 16 to 20 is a single batch; 24 is again 1.2x slower.
# Measured 2026-08-18 at W=18: CPU/wall = 0.99 per worker, i.e. no contention
# below the core count.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

COMMON=(--scenario-set tier1
        --ds-criterion guard --filter-ds
        --limits "$LIMITS"
        --search-dso-v-authority 20
        --workers 20)

say() { echo; echo "############ $(date '+%H:%M:%S')  $*"; echo; }

# ---------------------------------------------------------------- phase lam
# --rho-margin 0.031: the arg's own help says "at 0 the calibration does not
# transfer out of sample", and the archived 2026-08-14 calibration ran at 0
# (its `calibration` block is null) -- which is the lambda*-transfer defect in
# docs/tuning/METHOD_weight_selection.md.
say "PHASE LAM  (6 lambda values, ~1 batch)"
"$PY" -m tuning_mc.stage_1_search --phase lam "${COMMON[@]}" \
      --rho-target 1.5 --rho-margin 0.031 || { echo "phase lam FAILED"; exit 1; }

LAMSTAR=$("$PY" -c "
import json,sys
d=json.load(open(r'$OUT/phase_lambda.json'))
v=d.get('lambda_star')
print('' if v is None else repr(float(v)))
")
if [ -z "$LAMSTAR" ]; then
  echo "ABORT: phase lam found no feasible lambda* -- every swept value missed"
  echo "       the contraction target. Widen --lam-values or revisit the limits;"
  echo "       do NOT run phase a at an infeasible anchor."
  exit 1
fi
say "lambda* = $LAMSTAR   (re-anchoring lambda_tso; lambda_dso stays at X0)"

# Only lambda_tso is re-anchored.  Measured on this bank 2026-08-18: g3 sits at
# exactly +0.987 for every probe of lambda_dso / tau / engage_* / dso_g_v_ratio
# and moves ONLY with lambda_tso, so contraction is a one-coordinate property
# and detuning the DSO loop as well would cost f_ts for nothing.
# ---------------------------------------------------------------- phase a
say "PHASE A  (29 evaluations, ~2 batches)"
"$PY" -m tuning_mc.stage_1_search --phase a "${COMMON[@]}" \
      --x0 "lambda_tso=$LAMSTAR" || { echo "phase a FAILED"; exit 1; }

# ------------------------------------------------- gate: is the anchor sound?
# The whole point of the re-anchor.  If the design point is STILL infeasible,
# phase B starts from an infeasible incumbent, `filter_accepts` rejects
# everything, and the night is wasted producing an empty filter.  Fail loudly
# here instead.
GATE=$("$PY" -c "
import json
d = json.load(open(r'$OUT/phase_a.json'))
b = d.get('base') or {}
hard = b.get('hard') or {}
bad = [k for k, v in hard.items() if float(v) > 0]
print(('OK' if b.get('feasible') else 'BAD') + '|' + ','.join(bad) +
      '|' + ','.join(d.get('live') or []))
")
GATE_STATUS=${GATE%%|*}
GATE_REST=${GATE#*|}
GATE_VIOL=${GATE_REST%%|*}
GATE_LIVE=${GATE_REST#*|}
say "design point feasible? $GATE_STATUS   violated=[$GATE_VIOL]   live=[$GATE_LIVE]"
if [ "$GATE_STATUS" != "OK" ]; then
  echo "ABORT: the re-anchored design point is still infeasible on [$GATE_VIOL]."
  echo "       Phase B from an infeasible incumbent yields an empty filter."
  echo "       Look at that constraint before spending a night on phase B."
  exit 1
fi

# ---------------------------------------------------------------- phase b
# phase_b.json is rewritten after EVERY poll, so an unfinished phase B still
# leaves a valid incumbent + filter on disk.
say "PHASE B  (compass search; ~16 points per poll, 1 batch each)"
"$PY" -m tuning_mc.stage_1_search --phase b "${COMMON[@]}" \
      --x0 "lambda_tso=$LAMSTAR" || { echo "phase b FAILED"; exit 1; }

# ---------------------------------------------------------------- summary
say "RE-RUN COMPLETE -- summary"
"$PY" -c "
import json
o = r'$OUT'
lam = json.load(open(o + '/phase_lambda.json'))
a   = json.load(open(o + '/phase_a.json'))
b   = json.load(open(o + '/phase_b.json'))
print('lambda*            :', lam.get('lambda_star'))
print('live directions    :', a.get('live'))
print('dead directions    :', a.get('dead'))
base, best = a.get('base') or {}, b.get('best') or {}
for name, digits in (('f_ts', 6), ('f_q', 6), ('f_ds', 8)):
    lo, hi = base.get(name), best.get(name)
    if lo is None or hi is None:
        print(f'{name:5s} base -> best : n/a')
    else:
        print(f'{name:5s} base -> best : {lo:.{digits}f} -> {hi:.{digits}f}'
              f'  ({100 * (hi / lo - 1):+.2f} %)' if lo else '')
print('filter size        :', len(b.get('filter', [])))
print('polls run          :', len(b.get('history', [])))
print()
print('incumbent knobs:')
for k, v in sorted((b.get('incumbent') or {}).items()):
    print(f'   {k:20s} {v:.6g}')
"
