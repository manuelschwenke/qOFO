"""Experiment-only nameplate / load / conductor overrides per DSO underlay.

Moved here from ``analysis/annual_dso_pq_characterization.py`` on 2026-07-30 so
the closed-loop runner can apply the same multipliers as the annual
characterisation without importing an analysis module (that one pulls in
matplotlib).  One implementation, two callers:

* ``analysis/annual_dso_pq_characterization.py`` -- isolated annual probes;
* ``experiments/runners/multi_tso_dso.py`` -- the multi-TSO/DSO closed loop and
  therefore the PowerFactory RMS replay, whose snapshot carries the scaled
  ratings into PowerFactory through ``pf_sync``.

These are **scenario multipliers, not builder state**: ``constants.py`` still
defines the symmetric 410/700 MW networks, and a run that scales one DSO must
record the fact.  ``apply_dso_overrides`` writes what it applied into
``net["dso_overrides"]`` for exactly that reason.

Call it directly after ``add_hv_networks`` and BEFORE any power flow, load
model, droop tagging or operating-point initialisation: it rewrites ``p_mw``,
``base_p_mw``, ``sn_mva`` and the reactive-load base, all of which those steps
read.

Author: Manuel Schwenke / Claude Code (2026-07-30)
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import pandapower as pp


def apply_dso_overrides(
    net: pp.pandapowerNet,
    hv_networks: Sequence[object],
    *,
    dso_der_scale: Mapping[str, float] | None,
    dso_load_p_scale: Mapping[str, float] | None,
    dso_load_q_profile_base_mvar: Mapping[str, float] | None,
    dso_line_std_type: Mapping[str, str] | None,
) -> None:
    """Apply experiment-only nameplate/load overrides to selected DSOs."""
    der_scale = dict(dso_der_scale or {})
    load_p_scale = dict(dso_load_p_scale or {})
    load_q_profile_base = dict(dso_load_q_profile_base_mvar or {})
    line_std_type = dict(dso_line_std_type or {})
    known_ids = {str(hv.net_id) for hv in hv_networks}

    unknown_line_ids = sorted(set(line_std_type) - known_ids)
    if unknown_line_ids:
        raise ValueError(
            f"Unknown DSO IDs in line standard type: {unknown_line_ids}; "
            f"expected one of {sorted(known_ids)}"
        )
    if any(not str(value).strip() for value in line_std_type.values()):
        raise ValueError("DSO line standard type names must not be empty")

    for label, values, allow_zero in (
        ("DER scale", der_scale, False),
        ("active-load scale", load_p_scale, False),
        ("profile-only reactive-load base", load_q_profile_base, True),
    ):
        unknown = sorted(set(values) - known_ids)
        if unknown:
            raise ValueError(
                f"Unknown DSO IDs in {label}: {unknown}; "
                f"expected one of {sorted(known_ids)}"
            )
        for dso_id, value in values.items():
            minimum_ok = value >= 0.0 if allow_zero else value > 0.0
            if not np.isfinite(value) or not minimum_ok:
                comparator = "non-negative" if allow_zero else "positive"
                raise ValueError(
                    f"{label} for {dso_id} must be finite and {comparator}"
                )

    for hv in hv_networks:
        dso_id = str(hv.net_id)
        sgen_indices = list(hv.sgen_indices)
        load_indices = list(hv.load_indices)

        if dso_id in line_std_type:
            std_type = str(line_std_type[dso_id])
            if std_type not in net.std_types["line"]:
                library = pp.create_empty_network()
                try:
                    std_type_data = library.std_types["line"][std_type]
                except KeyError as exc:
                    raise ValueError(
                        f"Unknown pandapower line standard type {std_type!r}"
                    ) from exc
                pp.create_std_type(
                    net,
                    std_type_data,
                    std_type,
                    element="line",
                    overwrite=True,
                )
            for line_idx in hv.line_indices:
                pp.change_std_type(net, int(line_idx), std_type, element="line")

        if dso_id in der_scale:
            factor = float(der_scale[dso_id])
            for column in ("p_mw", "base_p_mw", "sn_mva"):
                if column in net.sgen.columns:
                    net.sgen.loc[sgen_indices, column] = (
                        pd.to_numeric(net.sgen.loc[sgen_indices, column])
                        * factor
                    )

        load_rating_scale = 1.0
        if dso_id in load_p_scale:
            factor = float(load_p_scale[dso_id])
            for column in ("p_mw", "base_p_mw"):
                net.load.loc[load_indices, column] = (
                    pd.to_numeric(net.load.loc[load_indices, column])
                    * factor
                )
            hv.total_ref_p_mw = float(hv.total_ref_p_mw) * factor
            load_rating_scale = max(load_rating_scale, factor)

        if dso_id in load_q_profile_base:
            target_q_mvar = float(load_q_profile_base[dso_id])
            profile_q = net.load.loc[load_indices, "profile_q"]
            profiled_indices = list(profile_q.index[profile_q.notna()])
            constant_indices = list(profile_q.index[profile_q.isna()])
            if not profiled_indices:
                raise ValueError(
                    f"{dso_id} has no reactive-load profile rows"
                )
            current_profile_base = float(
                pd.to_numeric(
                    net.load.loc[profiled_indices, "base_q_mvar"]
                ).sum()
            )
            if current_profile_base == 0.0 and target_q_mvar != 0.0:
                raise ValueError(
                    f"Cannot scale {dso_id} reactive profile from zero"
                )
            factor = (
                target_q_mvar / current_profile_base
                if current_profile_base != 0.0
                else 0.0
            )
            net.load.loc[constant_indices, ["q_mvar", "base_q_mvar"]] = 0.0
            for column in ("q_mvar", "base_q_mvar"):
                net.load.loc[profiled_indices, column] = (
                    pd.to_numeric(net.load.loc[profiled_indices, column])
                    * factor
                )
            hv.total_ref_q_mvar = target_q_mvar
            load_rating_scale = max(load_rating_scale, factor)

        if load_rating_scale != 1.0 and "sn_mva" in net.load.columns:
            net.load.loc[load_indices, "sn_mva"] = (
                pd.to_numeric(net.load.loc[load_indices, "sn_mva"])
                * load_rating_scale
            )

    net["dso_overrides"] = {
        "der_scale": der_scale,
        "load_p_scale": load_p_scale,
        "load_q_profile_base_mvar": load_q_profile_base,
        "line_std_type": line_std_type,
    }
