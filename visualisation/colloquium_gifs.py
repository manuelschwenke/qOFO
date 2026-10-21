"""Render seven presentation GIFs from completed colloquium QSS records.

Layout and aggregation follow the live plots. Both cases explicitly use
post-power-flow system telemetry, because the PI runner does not record the
new noisy OFO measurement snapshot channels. Commands and discrete positions
come from the corresponding recorded actuator states. No fallback is used.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import timedelta
import json
from pathlib import Path
import pickle
import zipfile

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image

COLORS = ["#004E8A", "#CC4C03", "#009D81", "#721085", "#009CDA",
          "#A60084", "#E6A800", "#596300"]
EVENTS = [(30, "Generator 2 trip"), (120, "+250 Mvar load at bus 11"),
          (180, "Generator 2 restored"), (210, "Line 25 trip")]
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "axes.titlesize": 14, "axes.labelsize": 12, "xtick.labelsize": 11,
    "ytick.labelsize": 11, "legend.fontsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#98A4AE", "axes.labelcolor": "#243746",
    "text.color": "#243746", "xtick.color": "#465B6B", "ytick.color": "#465B6B",
    "savefig.facecolor": "white",
})


def events_from_config(cfg, horizon_min=240):
    """Event markers/labels from the run's configured contingencies."""
    events = []
    for e in sorted(cfg.contingencies, key=lambda e: e.minute):
        if e.minute > horizon_min:
            continue
        if e.element_type == "gen":
            label = f"Generator {e.element_index} " + ("trip" if e.action == "trip" else "restored")
        elif e.element_type == "line":
            label = f"Line {e.element_index} " + ("trip" if e.action == "trip" else "restored")
        elif e.element_type == "load" and e.action == "connect":
            label = f"+{e.q_mvar:.0f} Mvar load at bus {e.bus}"
        else:
            label = f"{e.element_type} {e.element_index} {e.action}"
        events.append((e.minute, label))
    return events


def scalar(log, field_name, key):
    return np.array([getattr(r, field_name).get(key, np.nan) for r in log], float)


def matrix(log, field_name, key):
    rows = [np.asarray(getattr(r, field_name).get(key, []), float).ravel() for r in log]
    n = max(map(len, rows), default=0)
    if not n:
        raise ValueError(f"No recorded channels: {field_name}[{key}]")
    if any(len(r) not in (0, n) for r in rows):
        raise ValueError(f"Changing channel order/length: {field_name}[{key}]")
    return np.array([r if len(r) else np.full(n, np.nan) for r in rows])


def envelope_limits(*arrays, minimum_span=0.001, margin=0.10):
    vals = np.concatenate([np.asarray(a, float).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        raise ValueError("Cannot scale an empty panel")
    lo, hi = float(vals.min()), float(vals.max())
    span = max(hi - lo, minimum_span)
    return lo - margin * span, hi + margin * span


def band_polygons(x, lo, hi):
    valid = np.isfinite(lo) & np.isfinite(hi)
    # Preserve measurement gaps rather than interpolating through them.
    segments = np.split(np.flatnonzero(valid),
                        np.flatnonzero(np.diff(np.flatnonzero(valid)) > 1) + 1)
    return [np.column_stack((np.r_[x[s], x[s][::-1]],
                             np.r_[lo[s], hi[s][::-1]]))
            for s in segments if len(s) >= 2]


@dataclass
class Panel:
    ax: object
    lines: list = field(default_factory=list)
    bands: list = field(default_factory=list)

    def line(self, y, *, label=None, color=COLORS[0], style="-", step=False, lw=1.5):
        if not np.isfinite(y).any():
            raise ValueError(f"No finite samples for {label}")
        artist, = self.ax.plot([], [], color=color, ls=style, lw=lw, label=label,
                               drawstyle="steps-post" if step else "default")
        self.lines.append((artist, np.asarray(y)))
        return artist

    def band(self, lo, hi, color, alpha=0.14):
        artist = PolyCollection([], facecolors=color, alpha=alpha, edgecolors="none")
        self.ax.add_collection(artist)
        self.bands.append((artist, np.asarray(lo), np.asarray(hi)))

    def update(self, x, n):
        for artist, y in self.lines:
            artist.set_data(x[:n], y[:n])
        for artist, lo, hi in self.bands:
            artist.set_verts(band_polygons(x[:n], lo[:n], hi[:n]))


class AnimationFigure:
    def __init__(self, log, title, subtitle, rows, cols, footer):
        self.log = log
        self.time = np.array([r.time_s / 60 for r in log])
        self.fig, axs = plt.subplots(rows, cols, figsize=(16, 9), dpi=100, squeeze=False)
        self.axs = axs
        self.panels = []
        self.events = []
        self.fig.subplots_adjust(left=0.075, right=0.985, bottom=0.10,
                                 top=0.825, hspace=0.40, wspace=0.28)
        self.fig.text(0.035, 0.961, title, fontsize=22, weight="bold", va="top")
        self.fig.text(0.035, 0.913, subtitle, fontsize=11, va="top", color="#516979")
        self.clock = self.fig.text(0.965, 0.863, "", fontsize=14, ha="right",
                                   family="DejaVu Sans Mono", weight="bold")
        self.event_text = self.fig.text(0.035, 0.863, "", fontsize=11, color="#8B3F12")
        self.fig.text(0.035, 0.025, footer, fontsize=10, color="#516979")
        for i, row in enumerate(axs):
            for ax in row:
                ax.set_xlim(0, 240)
                ax.set_xticks([0, 60, 120, 180, 240])
                ax.grid(True, alpha=0.18)
                ax.set_axisbelow(True)
                if i == rows - 1:
                    ax.set_xlabel("Simulation time / min")
                else:
                    ax.tick_params(labelbottom=False)
                for t, _ in EVENTS:
                    marker = ax.axvline(t, color="#B1784B", lw=0.8, ls=":", visible=False)
                    self.events.append((marker, t))

    def panel(self, row, col, title, ylabel, limits=None, integer=False):
        ax = self.axs[row, col]
        ax.set_title(title, loc="left", pad=6)
        ax.set_ylabel(ylabel)
        if limits:
            if integer:
                limits = (float(np.floor(limits[0])), float(np.ceil(limits[1])))
            ax.set_ylim(*limits)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=integer))
        p = Panel(ax)
        self.panels.append(p)
        return p

    def update(self, minute):
        n = int(np.searchsorted(self.time, minute, side="right"))
        for p in self.panels:
            p.update(self.time, n)
        for artist, t in self.events:
            artist.set_visible(t <= minute)
        self.clock.set_text(f"t = {str(timedelta(seconds=int(round(minute * 60))))} / 4:00:00")
        past = [(t, label) for t, label in EVENTS if t <= minute]
        self.event_text.set_text(
            f"Latest event: {past[-1][1]}  ({past[-1][0]} min)" if past else "Profile-driven operation"
        )

    def save(self, path, frame_seconds=40, fps=10, preview_only=False):
        path = Path(path)
        self.update(240)
        self.fig.savefig(path.with_suffix(".png"), dpi=100)
        if preview_only:
            plt.close(self.fig)
            return None
        # Build one palette from the fully revealed figure. It includes all
        # trace colours and keeps colours stable across animation frames.
        self.fig.canvas.draw()
        palette = Image.fromarray(
            np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3].copy()
        ).quantize(colors=192, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        frames = []
        minutes = np.r_[np.arange(0, 14400, frame_seconds) / 60.0, 240.0]
        for i, minute in enumerate(minutes):
            self.update(float(minute))
            self.fig.canvas.draw()
            im = Image.fromarray(np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3].copy())
            frames.append(im.quantize(palette=palette, dither=Image.Dither.NONE))
            if i % 90 == 0:
                print(f"{path.name}: frame {i + 1}/{len(minutes)}", flush=True)
        durations = [int(round(1000 / fps))] * len(frames)
        durations[-1] += 2000
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations,
                       loop=0, optimize=False, disposal=1)
        plt.close(self.fig)
        with Image.open(path) as check:
            total_ms = 0
            for i in range(check.n_frames):
                check.seek(i)
                check.load()
                total_ms += check.info.get("duration", 0)
            info = {"file": str(path.name), "frames": check.n_frames,
                    "size": check.size, "duration_ms": total_ms,
                    "bytes": path.stat().st_size, "last_simulation_time_s": 14400}
        if info["frames"] != len(minutes) or info["size"] != (1600, 900):
            raise ValueError(f"GIF verification failed: {info}")
        path.with_suffix(".gif.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        print(f"Verified {path}: {info['frames']} frames, {total_ms / 1000:.1f} s", flush=True)
        return info


def legend(p, ncol=3):
    p.ax.legend(loc="upper left", ncol=ncol, frameon=True,
                facecolor="white", edgecolor="none", framealpha=0.85,
                handlelength=1.6, columnspacing=0.8, borderaxespad=0.3)


def voltage_figure(log, cfg, case, limits):
    name = "PI pilot-node control" if case == "pi" else "Proposed TS + STS OFO cascade"
    f = AnimationFigure(log, f"{name} | TS voltage tracking",
        "One subplot per TS area | mean and minimum-to-maximum system voltage | selected fixed parameters",
        3, 1, f"QSS | rural_700, doubled STS 3 | TS voltage limits: {cfg.v_min_pu:.2f}-{cfg.v_max_pu:.2f} p.u. | Post-power-flow values; all samples retained")
    for j, z in enumerate((1, 2, 3)):
        p = f.panel(j, 0, f"TS area {z}", "Voltage / p.u.", limits)
        lo, mn, hi = [scalar(log, f"zone_v_{s}", z) for s in ("min", "mean", "max")]
        c = COLORS[j]
        p.band(lo, hi, c)
        p.line(mn, label="Area mean", color=c, lw=2)
        ref = cfg.zone_v_setpoints_pu[z]
        p.ax.axhline(ref, color="#243746", ls="--", lw=1.2, label=f"Reference {ref:.2f} p.u.")
        handles, labels = p.ax.get_legend_handles_labels()
        handles += [Patch(facecolor=c, alpha=.16)]
        labels += ["Area min-max"]
        if case == "pi":
            # The diagnostic error is reference minus the pilot measurement.
            pilot = ref - scalar(log, "zone_svr_pilot_error_pu", z)
            artist = p.line(pilot, color="#721085", style=":", lw=1.7)
            handles += [artist]
            labels += [f"Pilot bus {cfg.svr_pilot_buses[z]}"]
        p.ax.legend(handles, labels, ncol=4, frameon=False, loc="upper left")
    return f


def actuator_figure(log, cfg, case, metadata):
    pi = case == "pi"
    rows = 4 if pi else 5
    title = "PI pilot-node control" if pi else "Proposed TS + STS OFO cascade"
    f = AnimationFigure(log, f"{title} | TS actuators",
        "One column per TS area | reactive infeed per device | recorded commands and discrete positions",
        rows, 3,
        ("QSS | PI with voltage-priority safeguard | MT taps held in the established S1 reference"
         if pi else "QSS | MT = machine transformer | Shunts: + reactor (MSR), - capacitor (MSC) | Minimum dwell: 3 h per shunt bank"))
    for j, z in enumerate((1, 2, 3)):
        zone = metadata["zones"][str(z)]
        if pi:
            ref = cfg.zone_v_setpoints_pu[z]
            pilot = ref - scalar(log, "zone_svr_pilot_error_pu", z)
            lim = envelope_limits(pilot, [ref], minimum_span=.01, margin=.25)
            p = f.panel(0, j, f"TS area {z} | pilot bus {cfg.svr_pilot_buses[z]}", "Pilot V / p.u.", lim)
            p.line(np.full(len(log), ref), label="Setpoint", color="#243746", style="--", step=True)
            p.line(pilot, label="Pilot voltage", color=COLORS[j])
            legend(p, 2)
        else:
            values = matrix(log, "zone_v_gen", z)
            p = f.panel(0, j, f"TS area {z} | AVR setpoints", "AVR V / p.u.",
                        envelope_limits(values, minimum_span=.005, margin=.20))
            for k in range(values.shape[1]):
                p.line(values[:, k], label=f"G{zone['generators'][k]}", color=COLORS[k], step=True)
            legend(p, 4)
        for row, key, label, ids in (
            (1, "zone_q_gen", "Generator Q / Mvar", zone["generators"]),
            (2, "zone_q_der", "DER Q / Mvar", zone["der"]),
            (3, "zone_oltc_taps", "MT tap / step", zone["details"]["oltc_trafo_indices"]),
        ):
            values = matrix(log, key, z)
            p = f.panel(row, j, "", label,
                        envelope_limits(values, minimum_span=2 if row == 3 else 10, margin=.15),
                        integer=row == 3)
            for k in range(values.shape[1]):
                prefix = "MT" if row == 3 else ("G" if row == 1 else "DER")
                p.line(values[:, k], label=f"{prefix}{ids[k]}", color=COLORS[k],
                       step=row == 3, lw=1.1)
            if values.shape[1] <= 5:
                legend(p, min(values.shape[1], 4))
            else:
                p.ax.text(.02, .92, f"{values.shape[1]} devices, one trace each",
                          transform=p.ax.transAxes, fontsize=9, va="top")
        if not pi:
            # Integrator banks live at STS coupling tertiaries. TS area 1
            # has no subordinate STS and therefore no installed banks.
            # Distinguish this structural absence from missing measurements.
            has_banks = any(np.asarray(r.zone_tso_shunt_states.get(z, [])).size for r in log)
            has_sts = z in metadata["dso_to_tso"].values()
            if not has_banks and not has_sts and not zone["details"]["shunt_bus_indices"]:
                p = f.panel(4, j, "", "Shunt / step", (-1, 1), integer=True)
                p.ax.text(.5, .5, "No shunt banks in this area", ha="center", va="center",
                          transform=p.ax.transAxes, fontsize=11, color="#677B88")
            else:
                values = matrix(log, "zone_tso_shunt_states", z)
                p = f.panel(4, j, "", "Shunt / step",
                            envelope_limits(values, minimum_span=2, margin=.20), integer=True)
                for k in range(values.shape[1]):
                    p.line(values[:, k], label=f"Bank {k + 1}", color=COLORS[k], step=True)
                legend(p, 4)
    return f


def sts_voltage_figure(log, cfg, groups):
    values = [scalar(log, f"dso_group_v_{s}_pu", g)
              for g in groups for s in ("min", "max")]
    limits = envelope_limits(*values, [cfg.v_setpoint_pu], minimum_span=.015)
    f = AnimationFigure(log, "Proposed TS + STS OFO cascade | STS voltages",
        "One subplot per STS zone | mean and minimum-to-maximum system voltage | common voltage scale",
        2, 2, "QSS | STS 3 DER capacity and active load doubled | Post-power-flow voltage; liveplot mean/min/max aggregation")
    for i, g in enumerate(groups):
        lo, mn, hi = [scalar(log, f"dso_group_v_{s}_pu", g) for s in ("min", "mean", "max")]
        p = f.panel(i // 2, i % 2, g.replace("DSO", "STS"), "Voltage / p.u.", limits)
        c = COLORS[i]
        p.band(lo, hi, c)
        p.line(mn, color=c, label="Zone mean", lw=2)
        p.ax.axhline(cfg.v_setpoint_pu, color="#243746", ls="--", lw=1.2,
                     label=f"Reference {cfg.v_setpoint_pu:.2f} p.u.")
        handles, labels = p.ax.get_legend_handles_labels()
        p.ax.legend(handles + [Patch(facecolor=c, alpha=.16)], labels + ["Zone min-max"],
                    ncol=3, frameon=False, loc="upper left")
    return f


def trafo_groups(log, groups):
    mapping = {}
    for r in log:
        for key, g in r.dso_trafo_group.items():
            if key in mapping and mapping[key] != g:
                raise ValueError(f"Transformer group changed: {key}")
            mapping[key] = g
    out = {g: sorted((k for k, value in mapping.items() if value == g),
                     key=lambda k: int(k.rsplit("trafo_", 1)[1])) for g in groups}
    if any(not keys for keys in out.values()):
        raise ValueError("STS zone without an interface transformer")
    return out


def sts_actuator_figure(log, groups, trafos):
    f = AnimationFigure(log, "Proposed TS + STS OFO cascade | STS actuators",
        "One column per STS zone | total actual DER reactive infeed and individual coupling-transformer taps",
        2, 4, "QSS | Shaded band: reported DER reactive capability | Tap positions are shown as integer steps")
    f.fig.subplots_adjust(left=.07, wspace=.37)
    for i, g in enumerate(groups):
        q = scalar(log, "dso_group_q_der_mvar", g)
        lo = scalar(log, "dso_group_q_der_min_mvar", g)
        hi = scalar(log, "dso_group_q_der_max_mvar", g)
        p = f.panel(0, i, g.replace("DSO", "STS"), "DER Q / Mvar",
                    envelope_limits(q, lo, hi, minimum_span=20))
        p.band(lo, hi, COLORS[i])
        p.line(q, label="Actual total", color=COLORS[i], lw=2)
        legend(p, 1)
        values = [scalar(log, "dso_trafo_tap_pos", k) for k in trafos[g]]
        p = f.panel(1, i, "Coupler OLTC positions", "Tap / step",
                    envelope_limits(*values, minimum_span=2, margin=.20), integer=True)
        for j, key in enumerate(trafos[g]):
            p.line(values[j], color=COLORS[j], label=key.split("|")[1].replace("trafo_", "T"),
                   step=True)
        legend(p, 3)
    return f


def interface_figure(log, groups, trafos):
    f = AnimationFigure(log, "Proposed TS + STS OFO cascade | Interface Q tracking",
        "One subplot per STS zone | one colour per coupling transformer | solid: achieved, dashed: commanded",
        2, 2, "Q at HV side of each coupling transformer | Positive: TS to STS | Shaded band: reported capability (clipped to plot scale)")
    for i, g in enumerate(groups):
        data = [scalar(log, field, k) for k in trafos[g]
                for field in ("dso_trafo_q_actual_mvar", "dso_trafo_q_set_mvar")]
        p = f.panel(i // 2, i % 2, g.replace("DSO", "STS"), "Interface Q / Mvar",
                    envelope_limits(*data, minimum_span=10, margin=.12))
        for j, key in enumerate(trafos[g]):
            c = COLORS[j]
            label = key.split("|")[1].replace("trafo_", "T")
            p.band(scalar(log, "dso_trafo_q_cap_min_mvar", key),
                   scalar(log, "dso_trafo_q_cap_max_mvar", key), c, alpha=.055)
            p.line(scalar(log, "dso_trafo_q_actual_mvar", key), color=c, label=label, lw=1.8)
            p.line(scalar(log, "dso_trafo_q_set_mvar", key), color=c, style="--", step=True, lw=1.2)
        legend(p, 3)
    return f


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--only", help="Comma-separated paths, e.g. pi/01_ts_voltages")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    logs, cfgs, metas = {}, {}, {}
    selected = set(args.only.split(",")) if args.only else None
    # An OFO-only selection loads only OFO records; the TS voltage scale is then
    # derived from OFO alone (shared PI/OFO scale requires selecting a pi/ figure).
    ofo_independent = selected and all(name.startswith("ofo/") for name in selected)
    for case in (("ofo",) if ofo_independent else ("pi", "ofo")):
        folder = run / case
        status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
        if status["status"] != "complete":
            raise ValueError(f"{case} is not complete")
        with (folder / "records.pkl").open("rb") as f:
            logs[case] = pickle.load(f)
        with (folder / "config.pkl").open("rb") as f:
            cfgs[case] = pickle.load(f)
        metas[case] = json.loads((folder / "network_metadata.json").read_text(encoding="utf-8"))
    events = {tuple(events_from_config(c)) for c in cfgs.values()}
    if len(events) != 1:
        raise ValueError(f"Cases have different contingency schedules: {events}")
    EVENTS[:] = events.pop()
    limits = envelope_limits(*[
        scalar(log, f"zone_v_{s}", z)
        for log in logs.values() for z in (1, 2, 3) for s in ("min", "max")
    ], [1.03], minimum_span=.02)
    groups = sorted(set().union(*(r.dso_group_v_min_pu.keys() for r in logs["ofo"])))
    if len(groups) != 4:
        raise ValueError(f"Expected frozen four-STS network, got {groups}")
    trafos = trafo_groups(logs["ofo"], groups)
    specs = []
    for case in ("pi", "ofo"):
        specs.extend([
            (f"{case}/01_ts_voltages", lambda case=case: voltage_figure(logs[case], cfgs[case], case, limits)),
            (f"{case}/02_ts_actuators", lambda case=case: actuator_figure(logs[case], cfgs[case], case, metas[case])),
        ])
    specs.extend([
        ("ofo/03_sts_voltages", lambda: sts_voltage_figure(logs["ofo"], cfgs["ofo"], groups)),
        ("ofo/04_sts_actuators", lambda: sts_actuator_figure(logs["ofo"], groups, trafos)),
        ("ofo/05_interface_q", lambda: interface_figure(logs["ofo"], groups, trafos)),
    ])
    selected = set(args.only.split(",")) if args.only else None
    report = []
    for name, builder in specs:
        if selected and name not in selected:
            continue
        info = builder().save(run / f"{name}.gif", preview_only=args.preview_only)
        if info:
            info["file"] = name + ".gif"
            report.append(info)
    if not args.preview_only:
        (run / ("gif_validation.json" if not selected else "gif_validation_partial.json")).write_text(
            json.dumps(report, indent=2), encoding="utf-8")
    if not args.preview_only and all((run / (name + ".gif")).exists() for name, _ in specs):
        complete_report = []
        for name, _ in specs:
            info = json.loads((run / (name + ".gif.json")).read_text(encoding="utf-8"))
            info["file"] = name + ".gif"
            complete_report.append(info)
        (run / "gif_validation.json").write_text(json.dumps(complete_report, indent=2), encoding="utf-8")
        with zipfile.ZipFile(run / "colloquium_gifs.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for name, _ in specs:
                for ext in (".gif", ".png"):
                    archive.write(run / (name + ext), arcname=name + ext)
            for name in ("README.md", "design.json", "gif_validation.json", "run_validation.json"):
                if (run / name).exists():
                    archive.write(run / name, arcname=name)
    print("Export complete", flush=True)


if __name__ == "__main__":
    main()
