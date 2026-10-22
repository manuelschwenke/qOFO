"""Offline checks of the residue machinery on a hand-built DAE; no PowerFactory."""
import math
import unittest

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from analysis import modal_observability as mo


def _dae():
    """x1' = -x1 + y1, x2' = -x2 + y1 (repeated eigenvalue -1),
    0 = -y1 + u (input), 0 = x1 - y2 (output y2 = x1)."""
    J = np.array([[-1, 0, 1, 0],
                  [0, -1, 1, 0],
                  [0, 0, -1, 0],
                  [1, 0, 0, -1]], dtype=float)
    V = pd.DataFrame({"i": range(4), "path": ["a", "b", "c", "d"], "parent": "P",
                      "leaf": ["x1.E", "x2.E", "y1.E", "y2.E"], "var": ["x", "x", "y", "y"]})
    index = {(r.parent, r.leaf, r.var): int(r.i) for r in V.itertuples()}
    return mo.Dae(V=V, J=csr_matrix(J), n=2, index=index)


class ResidueTests(unittest.TestCase):
    def test_reduction_and_maps(self):
        red = mo.Reduced(_dae())
        np.testing.assert_allclose(red.A, -np.eye(2))
        b = np.array([0, 0, 1.0, 0])                     # dg/du on the y1 row
        np.testing.assert_allclose(red.B(b), [1.0, 1.0])
        cz = np.array([[0.0, 1.0]])                     # output y2
        C, D = red.CD(cz, b[2:][:, None])
        np.testing.assert_allclose(C, [[1.0, 0.0]])
        np.testing.assert_allclose(D, [[0.0]])

    def test_cluster_projectors_sum_to_identity(self):
        rng = np.random.default_rng(0)
        T = rng.normal(size=(4, 4))
        A = T @ np.diag([-1.0, -1.0, -2.0, -0.5]) @ np.linalg.inv(T)
        _, cl = mo.clustered_modes(A)
        self.assertEqual(sorted(len(c["idx"]) for c in cl), [1, 1, 2])
        P = sum(c["Vc"] @ c["Wt"] for c in cl)
        np.testing.assert_allclose(P, np.eye(4), atol=1e-9)
        np.testing.assert_allclose(sum(c["lam"] * (c["Vc"] @ c["Wt"]) for c in cl), A,
                                   atol=1e-9)

    def test_step_response_and_settling(self):
        red = mo.Reduced(_dae())
        out = [{"name": "y2", "family": "V_TS", "kind": "V",
                "c": np.array([0, 0, 0, 1.0]), "y0": 0.0}]
        inp = [{"name": "u", "cls": "X", "b": np.array([0, 0, 1.0, 0]), "step": 0.01,
                "step_label": "0.01"}]
        df, info = mo.analyse(red, inp, out)
        r = df.iloc[0]
        self.assertAlmostEqual(r["dy_inf"], 0.01)
        # y2(t) = 0.01 (1 - e^-t): |tail| = 0.01 e^-t <= 1e-3 from t = ln 10
        self.assertAlmostEqual(r["Ts_abs_s"], math.log(10.0), delta=0.03)
        self.assertAlmostEqual(r["tail_at_20s"], -0.01 * math.exp(-20), places=12)
        self.assertEqual(info["unstable"], 0)

    def test_settle_time_floor(self):
        self.assertEqual(mo.settle_time(np.zeros(len(mo.T_GRID)), 1.0), mo.T_MIN)


if __name__ == "__main__":
    unittest.main()
