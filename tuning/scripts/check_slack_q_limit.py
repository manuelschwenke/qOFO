"""
Diagnostic: verify the slack generator's reactive-power envelope after
``build_ieee39_net(scenario="wind_replace")`` and ``add_hv_networks``.

Hypothesis (H1) under test
--------------------------
``network/ieee39/helpers.py:147`` creates G1 at bus 38 with the placeholder
``max_q_mvar=500.0, min_q_mvar=-500.0``.  The comment at lines 137-139 of
the same file claims the nameplate loop in ``build.py:139-153`` overwrites
these to ``±0.5 * sn_mva = ±5000 Mvar`` (since ``GEN_NAMEPLATE[38] = 10 GVA``).

If the overwrite has *not* taken effect, the slack operates with a 10x
tighter Q envelope than the nameplate intends, which alone explains
intermittent Newton-Raphson divergence in the L0 scenario of
``002_M_TSO_M_DSO_COMPARE``.

This script also computes the detailed Milano-12.2.1 capability-chart
envelope at the slack's operating point, so the user can see what the
"correct" (capability-chart-based) envelope would be.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable when the script is launched directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandapower as pp

from core.actuator_bounds import GeneratorParameters, compute_generator_q_limits
from network.ieee39.build import build_ieee39_net
from network.ieee39.constants import GEN_NAMEPLATE
from network.ieee39.hv_networks import add_hv_networks


def main() -> None:
    print("=" * 72)
    print("Slack-Q-limit diagnostic for wind_replace scenario")
    print("=" * 72)

    net, meta = build_ieee39_net(
        ext_grid_vm_pu=1.03,
        scenario="wind_replace",
        verbose=False,
    )
    meta = add_hv_networks(
        net, meta,
        install_tso_tertiary_shunts=False,
        verbose=False,
    )

    pp.runpp(net, run_control=False, calculate_voltage_angles=True,
             enforce_q_lims=False)

    slack_mask = net.gen["slack"].fillna(False).astype(bool)
    slack_idx_list = list(net.gen.index[slack_mask])
    assert len(slack_idx_list) == 1, (
        f"expected exactly one slack gen, got {len(slack_idx_list)}"
    )
    slack_idx = slack_idx_list[0]

    name      = net.gen.at[slack_idx, "name"]
    bus       = int(net.gen.at[slack_idx, "bus"])
    sn_mva    = float(net.gen.at[slack_idx, "sn_mva"])
    p_mw      = float(net.gen.at[slack_idx, "p_mw"])
    max_q     = float(net.gen.at[slack_idx, "max_q_mvar"])
    min_q     = float(net.gen.at[slack_idx, "min_q_mvar"])
    vm_pu_set = float(net.gen.at[slack_idx, "vm_pu"])
    p_post    = float(net.res_gen.at[slack_idx, "p_mw"])
    q_post    = float(net.res_gen.at[slack_idx, "q_mvar"])
    vm_post   = float(net.res_gen.at[slack_idx, "vm_pu"])

    print()
    print(f"Slack gen (idx {slack_idx}): {name} at bus {bus}")
    print(f"  GEN_NAMEPLATE[{bus}] = {GEN_NAMEPLATE.get(bus, '<missing>')}")
    print(f"  net.gen sn_mva     = {sn_mva:.1f} MVA")
    print(f"  net.gen p_mw (set) = {p_mw:.1f} MW")
    print(f"  net.gen vm_pu      = {vm_pu_set:.4f}")
    print(f"  net.gen max_q_mvar = {max_q:+.1f} Mvar")
    print(f"  net.gen min_q_mvar = {min_q:+.1f} Mvar")
    print()
    print(f"Post power-flow operating point:")
    print(f"  P = {p_post:+.1f} MW")
    print(f"  Q = {q_post:+.1f} Mvar")
    print(f"  V = {vm_post:.4f} pu")

    expected_box = 0.5 * sn_mva
    nameplate_loop_applied = abs(max_q - expected_box) < 1e-3 and abs(
        min_q + expected_box) < 1e-3

    print()
    print("-" * 72)
    print(f"Nameplate-loop overwrite detected: {nameplate_loop_applied}")
    print(f"  (expected from build.py:151-152:  max_q={+expected_box:+.1f}, "
          f"min_q={-expected_box:+.1f})")

    if not nameplate_loop_applied:
        print()
        print("  >>> The helpers.py:147 placeholder (±500 Mvar) is leaking through.")
        print("  >>> The slack is running with 10x less Q envelope than its nameplate.")
        print("  >>> This is the prime suspect for L0 divergence.")

    print()
    print("-" * 72)
    print("Reference: Milano §12.2.1 detailed capability chart at operating point")
    print("(uses default GeneratorParameters: x_d=1.8, i_f_max=2.7, beta=0.15, q0=0.4)")
    params = GeneratorParameters(
        s_rated_mva=sn_mva,
        p_max_mw=sn_mva,
    )
    q_min_chart, q_max_chart = compute_generator_q_limits(
        params, p_mw=p_post, v_pu=vm_post,
    )
    print(f"  q_min (chart) = {q_min_chart:+.1f} Mvar")
    print(f"  q_max (chart) = {q_max_chart:+.1f} Mvar")

    print()
    print("-" * 72)
    print("All synchronous gens for comparison (terminal bus, sn, p, max_q, min_q):")
    print(f"{'idx':>4} {'name':<24} {'bus':>4} {'sn_mva':>9} "
          f"{'p_mw':>9} {'max_q':>9} {'min_q':>9}")
    for gi in net.gen.index:
        print(f"{gi:>4} {net.gen.at[gi, 'name']:<24} "
              f"{int(net.gen.at[gi, 'bus']):>4} "
              f"{float(net.gen.at[gi, 'sn_mva']):>9.1f} "
              f"{float(net.gen.at[gi, 'p_mw']):>9.1f} "
              f"{float(net.gen.at[gi, 'max_q_mvar']):>+9.1f} "
              f"{float(net.gen.at[gi, 'min_q_mvar']):>+9.1f}")


if __name__ == "__main__":
    main()
