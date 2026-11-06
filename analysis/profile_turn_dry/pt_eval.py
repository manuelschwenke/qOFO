"""Dry evaluation of the profile-turn exemption draft (parameters fixed before
this was run) on the II-a rows of run 0008.  Read-only; nothing is re-judged."""
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SCR = Path(__file__).parent
RUN = Path(r"F:\qOFO_res\results\ch_09_parameter_selection\0008_2026-11-04_133847_ch_9_14_procedure_v4")
PROM_FRAC = 0.05      # turn prominence: 5 % of the area's DER capacity on each side
SIDE_S = 3600.0       # ... within 1 h (the dwell horizon)
LAG_S = 900.0         # turn may precede the first move by one minimum TS tap hold
BANK_TOL_S = 60.0     # a move within 60 s of any bank move = bank-coupled (counted)

R = json.loads((SCR / "pt_residuals.json").read_text("utf-8"))


def turns(r, cap):
    r = np.asarray(r, float)
    n, h, d = len(r), int(SIDE_S / 60), PROM_FRAC * cap
    out = []
    for i in range(n):
        lo, hi = max(0, i - h), min(n, i + h + 1)
        seg = r[lo:hi]
        left, right = r[lo:i + 1], r[i:hi]
        if r[i] == seg.min() and np.argmin(seg) == i - lo:
            if left.max() - r[i] >= d and right.max() - r[i] >= d:
                out.append((i * 60.0, "injection peak"))
        elif r[i] == seg.max() and np.argmax(seg) == i - lo:
            if r[i] - left.min() >= d and r[i] - right.min() >= d:
                out.append((i * 60.0, "injection trough"))
    return out


T = {w: {a: turns(v["r_mw"], v["der_cap_mw"]) for a, v in d["areas"].items()} for w, d in R.items()}
print("turns per window and area (h into window):")
for w, d in T.items():
    print(f"  {w}: " + "; ".join(f"{a} {[round(t / 3600, 2) for t, _ in v]}" for a, v in d.items() if v))


def area_of(dev):
    if dev.startswith("DSO_"):
        return dev.split("|")[0]
    if dev.startswith("z"):
        return "zone" + dev[1:].split("[")[0]
    return "zone" + dev.split("[")[0]


def classify(w, row, h, used):
    is_bank = h["transformer"].startswith("z")       # a bank's own moves do not couple it
    if not is_bank and any(abs(m[0] - t) <= BANK_TOL_S for m in row.get("shunt_moves", [])
                           for t in (h["t0_s"], h["t1_s"])):
        return "bank-coupled"
    for t, _ in T[w].get(area_of(h["transformer"]), []):
        if h["t0_s"] - LAG_S <= t <= h["t1_s"] and (h["transformer"], t) not in used:
            used.add((h["transformer"], t))
            return "profile-turn"
    return "counted"


rows = [json.load(open(f, encoding="utf-8")) for f in glob.glob(str(RUN / "stage2a/windows/*.json"))]
rows += [json.loads(l) for l in open(SCR / "v4_2a_remote_rows.jsonl", encoding="utf-8-sig") if l.strip()]
by = defaultdict(dict)
for r in rows:
    by[r["key"]][r["window"]] = r["row"]

for with_banks in (False, True):
    kinds = defaultdict(Counter)
    released, still, clean = [], [], []
    for k, pw in by.items():
        left = 0
        any_rev = False
        for w, row in pw.items():
            used = set()
            dets = list(row.get("hunting_detail", []))
            if with_banks:
                dets += [dict(x, transformer=x.get("bank", x.get("transformer", "")))
                         for x in row.get("shunt_hunting_detail", [])]
            elif row.get("n_shunt_hunting"):
                left += int(row["n_shunt_hunting"])
                kinds[w]["bank (not exempt)"] += int(row["n_shunt_hunting"])
                any_rev = True
            for h in dets:
                any_rev = True
                c = classify(w, row, h, used)
                kinds[w][c] += 1
                left += c != "profile-turn"
        if not any_rev:
            clean.append(k)
        elif left == 0:
            released.append(k)
        else:
            still.append(k)
    print(f"\n=== exemption on taps{' and banks' if with_banks else ' only'} ===")
    for w in sorted(kinds):
        print(f"  {w}: {dict(kinds[w])}")
    print(f"  candidates: {len(by)}; no reversal so far {len(clean)}; all reversals exempt "
          f"(would continue, re-simulation needed) {len(released)}; still rejected {len(still)}")

# shape of a shunt hunting detail, for the bank variant
ex = next((x for r in rows for x in r["row"].get("shunt_hunting_detail", [])), None)
print("\nshunt_hunting_detail example:", ex)
