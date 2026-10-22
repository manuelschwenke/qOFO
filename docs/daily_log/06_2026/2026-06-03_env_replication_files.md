# 2026-06-03 — Environment replication artifacts

**What changed**
- Added `requirements-lock.txt` — fully pinned `pip list --format=freeze` of the
  `qOFO_clean` env (Python 3.12.12, ~95 packages, exact `==` pins).
- Added `environment.yml` — conda recipe for full rebuild: conda provides
  `python=3.12`, `ipopt`, `blas=*=openblas`; pip section installs
  `-r requirements-lock.txt`.
- Left existing `requirements.txt` (hand-curated, loose pins) untouched.

**Method / reasoning**
- Env is hybrid. `conda env export --from-history` showed conda-installed:
  python, ipopt, pyomo, scipy, matplotlib, blas[openblas], gurobi=12.0.1.
  Only `ipopt` (and the OpenBLAS BLAS) genuinely require conda; the rest resolve
  fine as pip wheels, so they are left to the pip lock to avoid double-management
  and conda/pip version drift.
- Dropped the conda `gurobi` package from the recipe: `gurobipy==12.0.1` is
  already a pip wheel in the env; installing both would conflict. Documented the
  `grbgetkey` academic-license step instead.

**Why**
- Need reproducible setup on a second machine.

**Replicate**
- Preferred: `conda env create -f environment.yml && conda activate qOFO_clean`
- Pip-only (into existing 3.12 env): install ipopt separately, then
  `pip install -r requirements-lock.txt`.
