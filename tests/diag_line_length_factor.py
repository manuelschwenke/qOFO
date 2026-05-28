"""
Verify that LINE_LENGTH_FACTOR actually scales total R, X, B of the IEEE 39
TN lines and that increasing it produces the expected long-line / Ferranti
voltage uplift.

Run:
    "C:/Users/Manuel Schwenke/.conda/envs/qOFO_clean/python.exe" tests/diag_line_length_factor.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandapower as pp

from network.ieee39 import constants
from network.ieee39.build import build_ieee39_net


def _line_totals(net: pp.pandapowerNet, li: int) -> tuple[float, float, float]:
    """Return (total_R_ohm, total_X_ohm, total_C_nF) for a line."""
    L = float(net.line.at[li, "length_km"])
    return (
        float(net.line.at[li, "r_ohm_per_km"]) * L,
        float(net.line.at[li, "x_ohm_per_km"]) * L,
        float(net.line.at[li, "c_nf_per_km"])  * L,
    )


def _build(factor: float):
    """Return (net, converged_at_build).

    build_ieee39_net runs an internal power flow as a sanity check, so we
    catch LoadflowNotConverged and instead return the net constructed up to
    that point by re-running the bare construction without the trailing PF.
    """
    constants.LINE_LENGTH_FACTOR = factor
    try:
        net, _meta = build_ieee39_net()
        return net, True
    except Exception:
        # Fall back: construct the bare case39 + length fix, no internal PF.
        import pandapower.networks as pn
        from network.ieee39.helpers import fix_line_lengths
        net = pn.case39()
        fix_line_lengths(net)
        return net, False


def _try_solve(net: pp.pandapowerNet) -> bool:
    try:
        pp.runpp(net)
        return True
    except Exception:
        return False


def main() -> None:
    factors = (1.0, 1.5, 2.0, 3.0, 5.0, 10.0)
    built = {f: _build(f) for f in factors}
    nets = {f: net for f, (net, _) in built.items()}
    built_ok = {f: ok for f, (_, ok) in built.items()}
    converged = {
        f: built_ok[f] and _try_solve(nets[f]) for f in factors
    }

    # Reference line: from_bus=0 (bus 1, 1-idx) → to_bus=1 (bus 2, 1-idx).
    ref_li = nets[1.0].line.index[
        (nets[1.0].line["from_bus"] == 0) & (nets[1.0].line["to_bus"] == 1)
    ][0]

    print("\n=== Per-line totals for line 1-2 (1-indexed; case39 length 275.5 km) ===")
    print(f"{'factor':>8} {'R [Ohm]':>14} {'X [Ohm]':>14} {'C [nF]':>14} {'length_km':>12}")
    for f in factors:
        R, X, C = _line_totals(nets[f], ref_li)
        L = float(nets[f].line.at[ref_li, "length_km"])
        print(f"{f:>8.1f} {R:>14.4f} {X:>14.4f} {C:>14.2f} {L:>12.1f}")

    print("\n=== Build/PF status ===")
    for f in factors:
        print(
            f"factor={f:>5.1f}  build_ok={built_ok[f]!s:>5}  "
            f"pf_converged={converged[f]!s:>5}"
        )

    print("\n=== Bus-voltage summary (TN, vm_pu) — converged cases only ===")
    print(f"{'factor':>8} {'conv':>6} {'min':>10} {'mean':>10} {'max':>10} {'#>1.05':>8} {'#<0.95':>8}")
    for f in factors:
        if not converged[f]:
            print(f"{f:>8.1f} {'NO':>6}     -          -          -        -        -")
            continue
        vm = nets[f].res_bus["vm_pu"]
        print(
            f"{f:>8.1f} {'YES':>6} {vm.min():>10.4f} {vm.mean():>10.4f} {vm.max():>10.4f} "
            f"{int((vm > 1.05).sum()):>8d} {int((vm < 0.95).sum()):>8d}"
        )

    # Sanity assertions: totals must scale linearly. Use the highest factor
    # that still built (totals come from net data, not power-flow results,
    # so divergence is fine here).
    R1, X1, C1 = _line_totals(nets[1.0], ref_li)
    print()
    for f in factors[1:]:
        Rf, Xf, Cf = _line_totals(nets[f], ref_li)
        for label, a, b in (("R", R1, Rf), ("X", X1, Xf), ("C", C1, Cf)):
            ratio = b / a
            ok = abs(ratio - f) < 1e-9
            print(
                f"[{'OK ' if ok else 'BAD'}] total {label} ratio (factor {f} / factor 1) "
                f"= {ratio:.6f}, expected {f}"
            )

    constants.LINE_LENGTH_FACTOR = 1.0


if __name__ == "__main__":
    main()
