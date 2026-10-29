"""Would Amendment 1 have changed the Stage-II path of run 0001?

    python -X utf8 analysis/tap_engagement/replay_path.py <amendment run dir>

Stage II-b centres each round's box on the winner of the selection rule
(without the settling check) over everything evaluated so far, and keeps the
span factor f if a new candidate wins (halves it otherwise). If the amended
rule picks the same incumbent after II-a and after rounds 1-3 as the frozen
rule did, an application of the amended procedure from scratch draws the same
seeded Sobol points around the same centres with the same f, so it evaluates
the same 242 candidates; with bit-exact simulations it then differs from run
0001 only in the final selection. Per-candidate feasibility does not depend on
the evaluated set (same limits, same Stage-I baseline), so it is read from the
final report of run 0001.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tuning.stage2_search import rank_lexicographic  # noqa: E402

R = ROOT / "results/ch_09_parameter_selection/0001_2026-10-25_215433_weight_selection_driver_band"
A = Path(sys.argv[1])
amend = json.loads((A / "selection/theta_sts_min.json").read_text("utf-8"))
theta, bound = amend["theta_sts_min_pu"], amend["theta_max_pu"]
feas = {r["key"]: r for r in json.loads(
    (R / "selection/selection_2b_round4.json").read_text("utf-8"))["ranked"]}
stages = [("2a", R / "stage2a/candidates")] + [
    (f"2b_round{r}", R / f"stage2b/round{r}/candidates") for r in range(1, 5)]
cum = set()
same_all = True
for tag, d in stages:
    cum |= {p.stem[5:] for p in d.glob("cand_*.json")}
    logged = json.loads((R / f"selection/selection_{tag}.json").read_text("utf-8"))["ranked"][0]["key"]
    wf = rank_lexicographic([feas[k] for k in cum if k in feas])[0]["key"]
    wa = rank_lexicographic([feas[k] for k in cum if k in feas and theta[k] <= bound])[0]["key"]
    same = wf == wa
    if tag != "2b_round4":
        same_all &= same
    print(f"after {tag:10s}: evaluated {len(cum):3d}, feasible {sum(k in feas for k in cum):3d}, "
          f"admitted {sum(k in feas and theta[k] <= bound for k in cum):3d} | frozen {wf} "
          f"(run-0001 log {logged}) | amended {wa} (theta {theta[wa]:.4f}) | same {same}")
print(f"\nincumbents after II-a and rounds 1-3 identical: {same_all}")
