"""
experiments/reference_period/characterise_reference_period.py
=============================================================

One-off characterisation of the REFERENCE PERIOD of Part III (thesis
``ch:setup:scenarios:reference``): 05 January 2016, 08:00, 360 min, with the
six scheduled disturbances of ``make_config()``.  Chapters 9 and 10 show their
detailed (single-run) results on this period; the Monte-Carlo ensemble is
drawn elsewhere in the year (``experiments/part3_mc``).

What is computed, and from what
-------------------------------
* Exogenous inputs, from the network and the SimBench profiles, without a
  power flow: load and DER/IBG infeed per level (transmission area, each
  sub-transmission system), and their percentile rank in the whole profile
  year (15-min samples).  Loads at nominal voltage (the plant uses an
  anchored ZIP model, so the simulated load differs by its voltage factor).
* Power-flow quantities, from the L2 run of the reference campaign (local
  control only, so no coordinated controller shapes them): synchronous-
  machine dispatch per area, the network equivalent (slack machine), the
  active and reactive exchange at the coupling transformers of each
  sub-transmission system, and the range of the monitored transmission
  voltages.
* The disturbance list with the affected element, its area and its size.

Outputs
-------
``<out>/reference_period_inputs.csv``   15-min exogenous series
``<out>/reference_period_pf.csv``       1-min power-flow series (L2 run)
``<out>/reference_period_summary.json`` numbers quoted in the thesis
``<out>/reference_period_summary.md``   the same, readable
``<fig-dir>/reference_period.pdf``      thesis figure fig:setup:reference

Run
---
    python -m experiments.reference_period.characterise_reference_period \
        --run-dir results/THESIS_case1_svr/20261006_K1 \
        --out results/THESIS_reference_period --fig-dir <thesis>/graphics/setup

Author: Manuel Schwenke / Claude Code (2026-10-07)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from core.profiles import DEFAULT_PROFILES_CSV, load_profiles, snapshot_base_values
from experiments.case1_svr.run_case1 import resolved
from network.ieee39 import add_hv_networks, build_ieee39_net
from network.ieee39.dso_overrides import apply_dso_overrides
from network.zone_partition import fixed_zone_partition_ieee39

CM = 1 / 2.54
WIDTH = 12.0 * CM
HORIZON_MIN = 360

#: TU Darmstadt corporate colours as defined in the thesis preamble
#: (ClassicThesis.tex) and in qOFO configs/color_config.py.
TU = {
    "darkblue": "#004E8A",  # 1c  tudarkblue -- other machines + equivalent
    "midblue": "#00689D",   # 2c  tumidblue  -- network equivalent alone
    "olive": "#7FAB16",     # 4c  tuolive    -- wind
    "gold": "#D7AC00",      # 6c  tugold     -- photovoltaic
    "red": "#B90F22",       # 9c  tured      -- the tripped machine G_4
    "magenta": "#951169",   # 10c tumagenta  -- exchange at the couplers
}


def _setup() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Palatino Linotype", "Book Antiqua", "DejaVu Serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Palatino Linotype",
        "mathtext.it": "Palatino Linotype:italic",
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.6, "lines.linewidth": 1.0,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


# ---------------------------------------------------------------------------
#  Network, built as run_multi_tso_dso builds it (steps 1-3 + overrides)
# ---------------------------------------------------------------------------
def build(cfg):
    net, meta = build_ieee39_net(ext_grid_vm_pu=1.03, scenario=cfg.scenario,
                                 verbose=False)
    _, bus_zone = fixed_zone_partition_ieee39(net, verbose=False)
    meta = add_hv_networks(net, meta, install_tso_tertiary_shunts=False,
                           verbose=False)
    apply_dso_overrides(
        net, meta.hv_networks,
        dso_der_scale=cfg.dso_der_scale or None,
        dso_load_p_scale=cfg.dso_load_p_scale or None,
        dso_load_q_profile_base_mvar=cfg.dso_load_q_profile_base_mvar or None,
        dso_line_std_type=cfg.dso_line_std_type or None,
    )
    snapshot_base_values(net)
    return net, meta, dict(bus_zone)


def classify_buses(net, meta, bus_zone) -> Dict[int, str]:
    """Bus -> 'STS:<id>' for the 110 kV networks (their line-connected
    components), else 'TS:<area>'.  Buses created after the zone partition
    inherit the area of the nearest partitioned bus."""
    import networkx as nx
    import pandapower.topology as top
    g_lines = top.create_nxgraph(net, include_trafos=False,
                                 include_trafo3ws=False, include_impedances=False)
    level: Dict[int, str] = {}
    for hv in meta.hv_networks:
        for b in hv.bus_indices:
            for c in nx.node_connected_component(g_lines, int(b)):
                level[int(c)] = f"STS:{hv.net_id}"
    g_ts = top.create_nxgraph(net, include_trafo3ws=False)
    g_ts = g_ts.subgraph([b for b in net.bus.index if int(b) not in level])
    for b in net.bus.index:
        b = int(b)
        if b in level:
            continue
        if b in bus_zone:
            level[b] = f"TS:{int(bus_zone[b])}"
            continue
        dist = nx.single_source_shortest_path_length(g_ts, b)
        cand = sorted((d, n) for n, d in dist.items() if int(n) in bus_zone)
        level[b] = f"TS:{int(bus_zone[cand[0][1]])}" if cand else "TS:?"
    return level


# ---------------------------------------------------------------------------
#  Exogenous series over the whole profile year (15-min)
# ---------------------------------------------------------------------------
def exogenous_year(net, level: Dict[int, str]) -> pd.DataFrame:
    prof = load_profiles(DEFAULT_PROFILES_CSV, timestep_min=15)
    out: Dict[str, np.ndarray] = {}

    def add(key: str, v: np.ndarray) -> None:
        out[key] = out.get(key, 0.0) + v

    n = len(prof)
    for i in net.load.index:
        if not bool(net.load.at[i, "in_service"]):
            continue                         # dormant contingency loads
        grp = level[int(net.load.at[i, "bus"])]
        for q, col, base in (("P", "profile_p", "base_p_mw"),
                             ("Q", "profile_q", "base_q_mvar")):
            pc = net.load.at[i, col] if col in net.load.columns else None
            b = float(net.load.at[i, base])
            f = (prof[pc].to_numpy(float) if isinstance(pc, str) and pc in prof
                 else np.ones(n))
            add(f"load_{q}|{grp}", b * f)
    for s in net.sgen.index:
        if str(net.sgen.at[s, "name"]).startswith("BOUND_"):
            continue
        if not bool(net.sgen.at[s, "in_service"]):
            continue
        grp = level[int(net.sgen.at[s, "bus"])]
        pc = net.sgen.at[s, "profile"]
        b = float(net.sgen.at[s, "base_p_mw"])
        f = prof[pc].to_numpy(float) if isinstance(pc, str) and pc in prof else np.ones(n)
        add(f"der_P_{pc}|{grp}", b * f)
    df = pd.DataFrame(out, index=prof.index)
    # Level totals.
    for q in ("load_P", "load_Q"):
        df[f"{q}|TS"] = df[[c for c in df if c.startswith(q + "|TS:")]].sum(axis=1)
        df[f"{q}|STS"] = df[[c for c in df if c.startswith(q + "|STS:")]].sum(axis=1)
    for lv in ("TS", "STS"):
        df[f"der_P|{lv}"] = df[[c for c in df if c.startswith("der_P_")
                                and f"|{lv}:" in c]].sum(axis=1)
        for typ in ("PV3", "WP7", "WP10"):
            cols = [c for c in df if c.startswith(f"der_P_{typ}|{lv}:")]
            if cols:
                df[f"der_P_{typ}|{lv}"] = df[cols].sum(axis=1)
    sts_ids = sorted({c.split("|STS:")[1] for c in df if "|STS:" in c})
    for sid in sts_ids:
        df[f"der_P|STS:{sid}"] = df[[c for c in df if c.startswith("der_P_")
                                     and c.endswith(f"|STS:{sid}")]].sum(axis=1)
        df[f"net_P|STS:{sid}"] = df[f"load_P|STS:{sid}"] - df[f"der_P|STS:{sid}"]
    df["net_P|STS"] = df["load_P|STS"] - df["der_P|STS"]
    return df


def pct_rank(year: pd.Series, v: float) -> float:
    a = year.to_numpy(float)
    return float(100.0 * np.mean(a <= v))


# ---------------------------------------------------------------------------
#  Power-flow series from the L2 log
# ---------------------------------------------------------------------------
def pf_series(log: list) -> pd.DataFrame:
    rows = []
    for r in log:
        row = {"t_min": r.time_s / 60.0,
               "ne_P": float(r.slack_p_mw), "ne_Q": float(r.slack_q_mvar)}
        for z, p in (r.zone_balance_gen_p_mw or {}).items():
            row[f"sg_P|{z}"] = float(p)
        for z, q in (r.zone_balance_gen_q_mvar or {}).items():
            row[f"sg_Q|{z}"] = float(q)
        ex_p, ex_q = defaultdict(float), defaultdict(float)
        for k, p in (r.dso_trafo_p_actual_mw or {}).items():
            ex_p[r.dso_trafo_group.get(k, k.split("|")[0])] += float(p)
        for k, q in (r.dso_trafo_q_actual_mvar or {}).items():
            ex_q[r.dso_trafo_group.get(k, k.split("|")[0])] += float(q)
        for g in ex_p:
            row[f"exch_P|{g}"] = ex_p[g]
            row[f"exch_Q|{g}"] = ex_q[g]
        row["v_min"] = min((r.zone_v_min or {1: np.nan}).values())
        row["v_max"] = max((r.zone_v_max or {1: np.nan}).values())
        for z, v in (r.zone_v_min or {}).items():
            row[f"v_min|{z}"] = float(v)
        for z, v in (r.zone_v_max or {}).items():
            row[f"v_max|{z}"] = float(v)
        row["load_P_total"] = float(r.total_load_p_mw)
        row["gen_P_total"] = float(r.total_gen_p_mw)
        row["sgen_P_total"] = float(r.total_sgen_p_mw)
        rows.append(row)
    df = pd.DataFrame(rows)
    # Machines without the network equivalent: zone 1 carries the slack machine.
    if "sg_P|1" in df:
        df["sg_P|1_excl_ne"] = df["sg_P|1"] - df["ne_P"]
    df["sg_P_excl_ne"] = (df[[c for c in df if c.startswith("sg_P|")
                              and c[5:].isdigit()]].sum(axis=1) - df["ne_P"])
    df["exch_P|STS"] = df[[c for c in df if c.startswith("exch_P|")]].sum(axis=1)
    df["exch_Q|STS"] = df[[c for c in df if c.startswith("exch_Q|")]].sum(axis=1)
    df["losses_P"] = df["gen_P_total"] + df["sgen_P_total"] - df["load_P_total"]
    return df


def tripped_gen_series(log: list, minute: float) -> np.ndarray:
    """Active power of the machine that trips at ``minute``, over the run.

    ``zone_p_gen`` holds the machines of each area in a fixed order; the
    tripped one is the entry that drops to zero at the trip."""
    t = np.array([r.time_s / 60.0 for r in log])
    i = int(np.argmin(np.abs(t - minute)))
    before, after = log[max(i - 1, 0)], log[min(i + 1, len(log) - 1)]
    for z, a in (before.zone_p_gen or {}).items():
        b = (after.zone_p_gen or {}).get(z)
        if b is None or len(a) != len(b):
            continue
        for j in range(len(a)):
            if abs(float(b[j])) < 1e-3 < abs(float(a[j])):
                return np.array([float(np.asarray(r.zone_p_gen[z], float)[j])
                                 if z in (r.zone_p_gen or {}) else np.nan
                                 for r in log])
    raise ValueError(f"no machine trips at {minute} min")


def tripped_gen_dispatch(log: list, minute: float) -> Dict[str, float]:
    """Dispatched P of the machine that drops to zero at ``minute``."""
    t = np.array([r.time_s / 60.0 for r in log])
    i = int(np.argmin(np.abs(t - minute)))
    before, after = log[max(i - 1, 0)], log[min(i + 1, len(log) - 1)]
    best = {}
    for z, a in (before.zone_p_gen or {}).items():
        b = (after.zone_p_gen or {}).get(z)
        if b is None or len(a) != len(b):
            continue
        a, b = np.asarray(a, float), np.asarray(b, float)
        for j in range(len(a)):
            if abs(b[j]) < 1e-3 < abs(a[j]):
                best = {"area": int(z), "p_mw_before": float(a[j])}
    return best


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="K1 dir holding L2/log.pkl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fig-dir", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = resolved("L2", HORIZON_MIN)
    t0 = cfg.start_time
    t1 = t0 + _dt.timedelta(minutes=HORIZON_MIN)
    net, meta, bus_zone = build(cfg)
    level = classify_buses(net, meta, bus_zone)

    year = exogenous_year(net, level)
    win = year.loc[t0:t1]
    win.to_csv(out / "reference_period_inputs.csv")

    with open(Path(args.run_dir) / "L2" / "log.pkl", "rb") as f:
        log = pickle.load(f)
    pf = pf_series(log)
    trips = [e.minute for e in cfg.contingencies
             if e.element_type == "gen" and e.action == "trip"]
    if trips:
        # The tripped machine alone, and everything else that generates
        # synchronously (the other machines plus the network equivalent):
        # the second takes over what the first loses.
        pf["trip_gen_P"] = tripped_gen_series(log, trips[0])
        pf["other_sg_incl_ne_P"] = pf["gen_P_total"] - pf["trip_gen_P"]
    pf.iloc[::3].to_csv(out / "reference_period_pf.csv", index=False)

    # -- inventory --------------------------------------------------------
    inv: Dict[str, Dict[str, float]] = defaultdict(dict)
    for i in net.load.index:
        if bool(net.load.at[i, "in_service"]):
            g = level[int(net.load.at[i, "bus"])]
            inv[g]["load_base_P"] = inv[g].get("load_base_P", 0.0) + float(net.load.at[i, "base_p_mw"])
    for s in net.sgen.index:
        if str(net.sgen.at[s, "name"]).startswith("BOUND_") or not bool(net.sgen.at[s, "in_service"]):
            continue
        g = level[int(net.sgen.at[s, "bus"])]
        k = f"der_{net.sgen.at[s, 'profile']}_inst"
        inv[g][k] = inv[g].get(k, 0.0) + float(net.sgen.at[s, "sn_mva"])
    for gi in net.gen.index:
        g = level[int(net.gen.at[gi, "bus"])]
        key = "ne_sn" if bool(net.gen.at[gi, "slack"]) else "sg_sn"
        inv[g][key] = inv[g].get(key, 0.0) + float(net.gen.at[gi, "sn_mva"])
        if not bool(net.gen.at[gi, "slack"]):
            inv[g]["n_sg"] = inv[g].get("n_sg", 0) + 1

    # -- window statistics and annual ranks --------------------------------
    keys = (["load_P|TS", "load_Q|TS", "der_P|TS", "load_P|STS", "load_Q|STS",
             "der_P|STS", "der_P_PV3|STS", "der_P_WP7|STS", "der_P_WP10|STS",
             "net_P|STS"]
            + [c for c in year if c.startswith(("load_P|TS:", "net_P|STS:",
                                                "der_P|STS:", "load_P|STS:"))])
    stats = {}
    for k in keys:
        if k not in win:
            continue
        w = win[k]
        stats[k] = {"start": float(w.iloc[0]), "min": float(w.min()),
                    "mean": float(w.mean()), "max": float(w.max()),
                    "end": float(w.iloc[-1]),
                    "pct_min": pct_rank(year[k], float(w.min())),
                    "pct_mean": pct_rank(year[k], float(w.mean())),
                    "pct_max": pct_rank(year[k], float(w.max()))}

    pfk = (["ne_P", "sg_P_excl_ne", "exch_P|STS", "exch_Q|STS", "losses_P",
            "v_min", "v_max"]
           + [c for c in pf if c.startswith(("sg_P|", "exch_P|DSO", "exch_Q|DSO",
                                             "v_min|", "v_max|"))])
    pfstats = {k: {"start": float(pf[k].iloc[0]), "min": float(pf[k].min()),
                   "mean": float(pf[k].mean()), "max": float(pf[k].max())}
               for k in pfk if k in pf}

    # -- disturbances --------------------------------------------------------
    events: List[Dict] = []
    for e in sorted(cfg.contingencies, key=lambda e: e.minute):
        row = {"minute": int(e.minute), "type": e.element_type,
               "action": e.action}
        if e.element_type == "gen":
            gi = int(e.element_index)
            row.update(element=str(net.gen.at[gi, "name"]),
                       bus_ieee=int(net.bus.at[int(net.gen.at[gi, "bus"]), "name"])
                       if str(net.bus.at[int(net.gen.at[gi, "bus"]), "name"]).isdigit()
                       else None,
                       area=level[int(net.gen.at[gi, "bus"])],
                       sn_mva=float(net.gen.at[gi, "sn_mva"]),
                       q_range_mvar=[float(net.gen.at[gi, "min_q_mvar"]),
                                     float(net.gen.at[gi, "max_q_mvar"])])
            if e.action == "trip":
                row["dispatch"] = tripped_gen_dispatch(log, e.minute)
        elif e.element_type == "line":
            li = int(e.element_index)
            fb, tb = int(net.line.at[li, "from_bus"]), int(net.line.at[li, "to_bus"])
            row.update(element=f"line {li}",
                       ends_ieee=[str(net.bus.at[fb, "name"]), str(net.bus.at[tb, "name"])],
                       areas=[level[fb], level[tb]],
                       length_km=float(net.line.at[li, "length_km"]))
        else:
            row.update(bus=int(e.bus), bus_ieee=str(net.bus.at[int(e.bus), "name"]),
                       area=level[int(e.bus)], p_mw=float(e.p_mw),
                       q_mvar=float(e.q_mvar))
        events.append(row)

    summary = {"start": t0.isoformat(), "end": t1.isoformat(),
               "horizon_min": HORIZON_MIN, "inventory": inv,
               "exogenous": stats, "power_flow_L2": pfstats, "events": events,
               "note": ("exogenous = profiles x base values at nominal voltage; "
                        "power_flow_L2 = local control only (variant L2); "
                        "exchange positive = import into the STS")}
    (out / "reference_period_summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8")

    md = [f"# Reference period {t0:%d %b %Y %H:%M} + {HORIZON_MIN} min", ""]
    md.append("## Exogenous (min / mean / max, annual percentile of the mean)")
    for k, s in stats.items():
        md.append(f"- {k}: {s['min']:.0f} / {s['mean']:.0f} / {s['max']:.0f}  "
                  f"(P{s['pct_mean']:.0f}; range P{s['pct_min']:.0f}-P{s['pct_max']:.0f})")
    md += ["", "## Power flow, L2 (min / mean / max)"]
    for k, s in pfstats.items():
        f = ".3f" if k.startswith("v_") else ".0f"
        md.append(f"- {k}: {s['min']:{f}} / {s['mean']:{f}} / {s['max']:{f}}")
    md += ["", "## Inventory"] + [f"- {g}: {v}" for g, v in sorted(inv.items())]
    md += ["", "## Disturbances"] + [f"- {json.dumps(e, default=float)}" for e in events]
    (out / "reference_period_summary.md").write_text("\n".join(md) + "\n",
                                                     encoding="utf-8")
    print("\n".join(md))

    if args.fig_dir:
        figure(win, pf, events, t0, Path(args.fig_dir))


def figure(win: pd.DataFrame, pf: pd.DataFrame, events: List[Dict],
           t0: _dt.datetime, fig_dir: Path) -> None:
    _setup()
    fig_dir.mkdir(parents=True, exist_ok=True)
    tm = (win.index - pd.Timestamp(t0)).total_seconds() / 60.0
    fig, ax = plt.subplots(2, 1, figsize=(WIDTH, 10.8 * CM), sharex=True,
                           gridspec_kw=dict(hspace=0.62))
    a = ax[0]
    a.plot(tm, win["load_P|TS"], color="black", lw=1.1,
           label="Transmission load")
    if "other_sg_incl_ne_P" in pf:
        a.plot(pf["t_min"], pf["other_sg_incl_ne_P"], color=TU["darkblue"], lw=1.0,
               label="Other machines + equivalent")
    a.plot(pf["t_min"], pf["ne_P"], color=TU["midblue"], lw=1.0, ls="--",
           label="Network equivalent")
    if "trip_gen_P" in pf:
        a.plot(pf["t_min"], pf["trip_gen_P"], color=TU["red"], lw=1.1,
               label="Machine $G_4$")
    a.plot(tm, win["der_P|TS"], color=TU["olive"], lw=1.0,
           label="Transmission wind")
    a.set_ylabel("Active power [MW]")
    top = float(np.nanmax(win["load_P|TS"]))
    a.set_ylim(0.0, top + 300.0)
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3,
             frameon=False, handlelength=1.6, columnspacing=0.9)
    a.set_title("(a) Transmission level", loc="left", pad=10)

    b = ax[1]
    b.plot(tm, win["load_P|STS"], color="black", lw=1.1, label="Load")
    b.plot(tm, win["der_P_WP10|STS"] + win.get("der_P_WP7|STS", 0.0),
           color=TU["olive"], lw=1.0, label="Wind")
    b.plot(tm, win["der_P_PV3|STS"], color=TU["gold"], lw=1.0,
           label="Photovoltaic")
    b.plot(pf["t_min"], pf["exch_P|STS"], color=TU["magenta"], lw=1.0, ls="--",
           label="Exchange (import $>0$)")
    b.axhline(0.0, color="0.6", lw=0.5)
    b.set_ylabel("Active power [MW]")
    b.set_xlabel("Time since 08:00 [min]")
    b.set_xlim(0, HORIZON_MIN)
    b.set_xticks(range(0, HORIZON_MIN + 1, 60))
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=4,
             frameon=False, handlelength=1.6, columnspacing=1.0)
    b.set_title("(b) Sub-transmission systems, aggregated", loc="left")

    # Disturbances: dotted lines, numbered above panel (a) in time order as
    # in tab:setup:reference:events.
    for k, e in enumerate(sorted(events, key=lambda e: e["minute"]), start=1):
        for x in ax:
            x.axvline(e["minute"], color="0.55", lw=0.6, ls=":")
        ax[0].annotate(str(k), xy=(e["minute"], 1.0),
                       xycoords=("data", "axes fraction"),
                       xytext=(0, 1.5), textcoords="offset points",
                       va="bottom", ha="center", fontsize=6.5, color="0.25")
    for x in ax:
        x.grid(True, lw=0.3, color="0.85")
    fig.savefig(fig_dir / "reference_period.pdf")
    fig.savefig(fig_dir / "reference_period.png", dpi=200)
    plt.close(fig)
    print(f"figure: {fig_dir / 'reference_period.pdf'}")


if __name__ == "__main__":
    main()
