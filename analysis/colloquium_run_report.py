"""Audit completed four-hour colloquium runs without modifying their records."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import numpy as np


def scalar(log, name, key):
    return np.array([getattr(r, name).get(key, np.nan) for r in log], float)


def movements(log, field):
    previous, changes, travel = {}, 0, 0.0
    for r in log:
        for key, value in getattr(r, field).items():
            a = np.asarray(value, float).ravel()
            if key in previous:
                if a.shape != previous[key].shape:
                    raise ValueError(f"{field}: channel count changed for {key}")
                delta = np.abs(a - previous[key])
                changes += int(np.count_nonzero(delta > 0.5))
                travel += float(delta.sum())
            previous[key] = a
    return {"device_position_changes": changes, "total_step_travel": travel}


def audit(folder):
    folder = Path(folder)
    with (folder / "records.pkl").open("rb") as stream:
        log = pickle.load(stream)
    with (folder / "config.pkl").open("rb") as stream:
        cfg = pickle.load(stream)
    times = np.array([r.time_s for r in log])
    if len(log) != 720 or not np.array_equal(times, np.arange(1, 721) * 20.0):
        raise ValueError("Run does not contain the complete requested 4-hour timeline")
    report = {"records": len(log), "last_time_s": float(times[-1]), "ts_areas": {}}
    for z in (1, 2, 3):
        low, high = scalar(log, "zone_v_min", z), scalar(log, "zone_v_max", z)
        rms = scalar(log, "zone_v_rms_err_pu", z)
        if not np.isfinite(low).all() or not np.isfinite(high).all():
            raise ValueError(f"Non-finite TS plant voltage in area {z}")
        lo_limit = (cfg.zone_v_min_pu or {}).get(z, cfg.v_min_pu)
        hi_limit = (cfg.zone_v_max_pu or {}).get(z, cfg.v_max_pu)
        report["ts_areas"][z] = {
            "plant_voltage_min_pu": float(low.min()),
            "plant_voltage_max_pu": float(high.max()),
            "steps_outside_configured_voltage_bounds": int(np.sum((low < lo_limit) | (high > hi_limit))),
            "time_mean_spatial_rms_error_pu": float(np.nanmean(rms)),
            "rms_sample_count": int(np.isfinite(rms).sum()),
        }
    report["ts_machine_taps"] = movements(log, "zone_oltc_taps")
    report["ts_shunts"] = movements(log, "zone_tso_shunt_states")
    if cfg.tso_mode == "ofo":
        if cfg.shunt_t_dwell_s != 10800.0:
            raise ValueError("The presentation OFO case must use exactly 3 h shunt dwell")
        prev, times_by_bank = {}, {}
        for r in log:
            for z, value in r.zone_tso_shunt_states.items():
                a = np.asarray(value)
                if z in prev:
                    for k in np.flatnonzero(a != prev[z]):
                        times_by_bank.setdefault(f"zone{z}_bank{k + 1}", []).append(r.time_s)
                prev[z] = a
        for bank, times in times_by_bank.items():
            if len(times) > 1 and np.min(np.diff(times)) < cfg.shunt_t_dwell_s - 1e-8:
                raise ValueError(f"Shunt dwell violated in recorded trajectory: {bank}")
        report["shunt_dwell_audit"] = {
            "configured_s": cfg.shunt_t_dwell_s,
            "switch_times_s_by_bank": times_by_bank,
            "all_observed_inter_switch_intervals_compliant": True,
            "first_switch": "Eligible immediately; dwell applies between switches of the same bank",
        }
    report["sts_coupler_taps"] = movements(log, "dso_trafo_tap_pos")
    report["solver_statuses"] = {
        "tso": dict(Counter(str(s) for r in log for s in r.zone_tso_status.values())),
        "dso": dict(Counter(str(s) for r in log for s in r.dso_status.values())),
    }
    report["sts_groups"] = {}
    groups = sorted(set().union(*(r.dso_group_v_meas_min_pu.keys() for r in log)))
    for g in groups:
        lo = scalar(log, "dso_group_v_min_pu", g)
        hi = scalar(log, "dso_group_v_max_pu", g)
        report["sts_groups"][g] = {
            "plant_voltage_min_pu": float(np.nanmin(lo)),
            "plant_voltage_max_pu": float(np.nanmax(hi)),
        }
    report["interfaces"] = {}
    keys = sorted(set().union(*(r.dso_trafo_q_set_mvar.keys() for r in log)))
    for key in keys:
        command = scalar(log, "dso_trafo_q_set_mvar", key)
        metered = scalar(log, "dso_trafo_q_actual_mvar", key)
        finite = np.isfinite(command) & np.isfinite(metered)
        if not finite.any():
            raise ValueError(f"No commanded/metered sample pairs for {key}")
        error = metered[finite] - command[finite]
        report["interfaces"][key] = {
            "sample_pairs": int(finite.sum()),
            "achieved_command_rmse_mvar": float(np.sqrt(np.mean(error ** 2))),
            "max_abs_achieved_command_error_mvar": float(np.abs(error).max()),
            "timing": "same recorded step; post-power-flow achieved Q versus current command",
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    report = {case: audit(args.run_dir / case) for case in ("pi", "ofo")}
    report["interpretation"] = (
        "Architecture comparison, not an isolated control-law effect. "
        "Shunt position changes are reported; the corrected OFO run enforces 3 h dwell per bank. "
        "Metrics describe this trajectory, not a statistical robustness claim."
    )
    path = args.run_dir / "run_validation.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
