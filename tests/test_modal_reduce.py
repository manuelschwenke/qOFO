"""Checks of the declared modal selection rule on synthetic modes; no PowerFactory."""
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

from analysis import modal_reduce as mr


def _p(**kw):
    return {g: kw.get(g, 0.0) for g in mr.GROUPS}


class LvqRuleTests(unittest.TestCase):
    def test_vq_groups_always_in(self):
        for g in ("rotor", "exc", "conv"):
            self.assertTrue(mr.in_l_vq(g, _p(**{g: 1.0})))

    def test_governor_excluded_only_below_threshold(self):
        self.assertFalse(mr.in_l_vq("gov", _p(gov=0.91, rotor=0.09)))
        self.assertTrue(mr.in_l_vq("gov", _p(gov=0.90, rotor=0.10)))
        # per-group test, not the sum: 0.06 + 0.06 stays out
        self.assertFalse(mr.in_l_vq("gov", _p(gov=0.88, rotor=0.06, exc=0.06)))

    def test_other_is_conservative_by_default(self):
        self.assertTrue(mr.in_l_vq("other", _p(other=1.0)))
        self.assertFalse(mr.in_l_vq("other", _p(other=1.0), other_is_vq=False))


class ReductionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.src = self.dir / "modes.csv"
        mr.make_synthetic().to_csv(self.src, index=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_rule_numbers(self):
        ev = mr.reduce([self.src], self.dir / "b", plot=False)
        r = ev["db_wide"]
        self.assertAlmostEqual(r["tau_exc"], 2.5)
        self.assertAlmostEqual(r["tap_term"], 12.5)
        self.assertAlmostEqual(r["tau_lvq_max"], 4.0)
        self.assertAlmostEqual(r["margin"], 7.5)
        # the 12.5 s governor mode is the slowest overall and outside L_VQ
        self.assertEqual(r["slowest_overall"]["group"], "gov")
        self.assertFalse(r["slowest_overall"]["in_L_VQ"])
        # the governor mode with p_rotor = 0.12 trips the governor clause of R1
        self.assertTrue(r["r1_triggered"])
        self.assertEqual(len(r["r1_slow_lvq"]), 0)
        self.assertEqual(len(r["r1_gov_vq"]), 2)
        self.assertEqual(r["near_zero"], 4)
        self.assertEqual(r["csv_lvq_disagreements"], 0)

    def test_slow_lvq_mode_triggers_r1(self):
        df = mr.make_synthetic()
        df.loc[(df["mode_idx"] == 3), "re"] = -0.1     # excitation, tau 10 s
        df.to_csv(self.src, index=False)
        ev = mr.reduce([self.src], self.dir / "b", plot=False)
        r = ev["db_zero"]
        self.assertTrue(any(m["group"] == "exc" for m in r["r1_slow_lvq"]))
        self.assertAlmostEqual(r["tap_term"], 35.0)
        self.assertLess(r["margin"], 0)

    def test_near_zero_never_enters_tau(self):
        ev = mr.reduce([self.src], self.dir / "b", plot=False)
        for r in ev.values():
            self.assertTrue(all(abs(m["re"]) >= mr.NEAR_ZERO_RE
                                for m in r["r1_slow_lvq"]))
            self.assertEqual(len(r["r1_pending_near_zero"]), 2)

    def test_emitter_disagreement_is_reported(self):
        df = mr.make_synthetic()
        df.loc[0, "in_L_VQ"] = False
        df.to_csv(self.src, index=False)
        ev = mr.reduce([self.src], self.dir / "b", plot=False)
        self.assertEqual(ev["db_wide"]["csv_lvq_disagreements"], 1)

    def test_table_has_one_row_per_group(self):
        mr.reduce([self.src], self.dir / "b", plot=False)
        body = (self.dir / "b" / "modal_groups.tex").read_text(encoding="utf-8")
        rows = [l for l in body.splitlines() if l.endswith("\\\\")]
        self.assertEqual([r.split(" & ")[0] for r in rows],
                         [mr.GROUP_LABEL[g] for g in mr.GROUPS])
        self.assertTrue(all(r.count(" & ") == 5 for r in rows))

    def test_duplicate_rows_rejected(self):
        df = mr.make_synthetic()
        df = df._append(df.iloc[0])
        df.to_csv(self.src, index=False)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            mr.load([self.src])


if __name__ == "__main__":
    unittest.main()
