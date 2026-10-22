"""Offline checks of the PF modal-export parser and grouping; no PowerFactory."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pf import modal_participation as mp

VARIDX = """
 Matrix column index; Model name; State variable
     1    Grid\\Power Plant 03\\AVR 03.ElmDsl                  "xr"
     2    Grid\\G 01.ElmSym                                    "phi"
     3    Grid\\WECC_WP_TSO_s0_b18\\QVPRE.ElmDsl               "x1"
     4    DSO_1\\Load 7.ElmLod                                 "x"
"""

OUTPUT = """info - Checking for state variables, that were assumed to be constant at current operating point...
info - The state variables of the following models were assumed to be constant at current operating point:
   AVR 03 [Power Plant 03]: xr
   REEC_D Electrical Control Model [WECC_X]: x2, x3
-------------------------------------------------------------------
info - Computing eigenvalues/eigenvectors...
   Ghost [Nowhere]: xz
"""


def _write_triplets(path, rows):
    path.write_text("\n".join("  ".join(repr(float(v)) if i > 1 else str(int(v))
                                        for i, v in enumerate(r)) for r in rows))


class ParserTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        (self.d / "VariableToIdx_Amat.txt").write_text(VARIDX)
        # eigenvalues: 0 (angle ref), -50 (QVPRE), -10 (load), complex pair
        _write_triplets(self.d / "EVals.mtl", [(1, 1, 0.0, 0.0), (2, 1, -50.0, 0.0),
                                               (3, 1, -0.5, 6.0), (4, 1, -0.5, -6.0)])
        # PartFacs: row = state, column = mode
        P = [(2, 1, 1.0, 0.0), (3, 2, 1.0, 0.0),
             (1, 3, 0.3, 0.1), (2, 3, 0.6, -0.1), (4, 3, 0.1, 0.0),
             (1, 4, 0.3, -0.1), (2, 4, 0.6, 0.1), (4, 4, 0.1, 0.0)]
        _write_triplets(self.d / "PartFacs.mtl", P)

    def tearDown(self):
        self.tmp.cleanup()

    def test_varidx_and_constant_states(self):
        vi = mp.read_varidx(self.d)
        self.assertEqual(list(vi["parent"]), ["Power Plant 03", "Grid",
                                              "WECC_WP_TSO_s0_b18", "DSO_1"])
        self.assertEqual(vi.loc[3, "obj_class"], "ElmLod")
        const = mp.parse_constant_states(OUTPUT.splitlines())
        self.assertEqual(const, {("Power Plant 03", "AVR 03", "xr"),
                                 ("WECC_X", "REEC_D Electrical Control Model", "x2"),
                                 ("WECC_X", "REEC_D Electrical Control Model", "x3")})
        # PF's GetContent() delivers a multi-line message as ONE element
        blob = ["info - Checking ...", "\n".join(OUTPUT.splitlines()[1:5]), "info - next"]
        self.assertEqual(mp.parse_constant_states(blob), const)

    def test_group_rules(self):
        self.assertEqual(mp.group_of("ElmSym", "", "Grid"), "rotor")
        self.assertEqual(mp.group_of("ElmDsl", "avr_IEEET1", "Power Plant 03"), "exc")
        self.assertEqual(mp.group_of("ElmDsl", "gov_IEEEG3", "Power Plant 10"), "gov")
        self.assertEqual(mp.group_of("ElmDsl", "QVPRE_qOFO", "WECC_A"), "conv")
        self.assertEqual(mp.group_of("StaVmea", "", "WECC_A"), "conv")
        self.assertEqual(mp.group_of("ElmDsl", "Protection V_f_ROCOF", "WECC_A"), "conv")
        self.assertEqual(mp.group_of("ElmDsl", "TAPCTRL_qOFO", "TAPC_T1"), "other")
        self.assertEqual(mp.group_of("ElmLod", "", "DSO_1"), "other")

    def test_mode_table(self):
        lam = mp.read_evals(self.d)
        P = mp.read_partfacs(self.d, len(lam))
        vi = mp.read_varidx(self.d)
        meta = {("Power Plant 03", "AVR 03.ElmDsl"): ("ElmDsl", "avr_IEEET1", "Power Plant 03", "ElmComp"),
                ("Grid", "G 01.ElmSym"): ("ElmSym", "", "Grid", "ElmNet"),
                ("WECC_WP_TSO_s0_b18", "QVPRE.ElmDsl"): ("ElmDsl", "QVPRE_qOFO", "WECC_WP_TSO_s0_b18", "ElmComp")}
        st = mp.state_table(vi, meta, mp.parse_constant_states(OUTPUT.splitlines()))
        self.assertFalse(st.loc[3, "resolved"])          # the load: path only
        modes, tops = mp.mode_table(lam, P, st, window="W1", droop_case="db_zero")
        self.assertEqual(len(modes), 3)                  # Im < 0 dropped
        m = modes.set_index("mode_idx")
        self.assertEqual(m.loc[1, "group"], "rotor")
        self.assertEqual(m.loc[2, "group"], "conv")
        self.assertAlmostEqual(m.loc[2, "tau_s"], 0.02)
        # complex mode: |p| = (0.316, 0.608, 0.1) normalised
        mag = np.array([abs(0.3 + 0.1j), abs(0.6 - 0.1j), 0.1])
        self.assertAlmostEqual(m.loc[3, "p_rotor"], mag[1] / mag.sum())
        self.assertAlmostEqual(m.loc[3, "p_exc"], mag[0] / mag.sum())
        self.assertAlmostEqual(m.loc[3, "p_other"], mag[2] / mag.sum())
        self.assertTrue(m.loc[3, "in_L_VQ"])
        self.assertEqual(m.loc[3, "dominant_object"], "Grid\\G 01")
        self.assertEqual(len(tops), 3 * 4)

    def test_degenerate_clusters(self):
        import pandas as pd
        modes = pd.DataFrame({"re": [-50.0, -50.0, -50.0, -1.0], "im": [0, 0, 0, 0],
                              "group": ["conv", "conv", "other", "rotor"]})
        cl = mp.degenerate_clusters(modes)
        self.assertEqual(len(cl), 1)
        self.assertEqual(cl.loc[0, "size"], 3)
        self.assertFalse(cl.loc[0, "single_group"])


if __name__ == "__main__":
    unittest.main()
