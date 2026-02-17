import tempfile
import unittest
from pathlib import Path

import prok_v5 as p


class TestMVPSafety(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.state_path = Path(self.tmp.name) / "state.json"
        self.log_path = Path(self.tmp.name) / "log.jsonl"
        p.STATE_FILE = self.state_path
        p.LOG_FILE = self.log_path

        self.st = p.ProkState()
        p._init_defaults(self.st)
        self.st.tasks = [
            p.Task(
                id="t1",
                domain="skola",
                name="MAT",
                deadline=None,
                total_minutes=300,
                remaining_minutes=300,
                tmt_bins=p.TMTBins(),
                recent_drift_count=0,
            )
        ]
        self.st.active_task_id = "t1"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_homeostat_require_filters_when_possible(self) -> None:
        hp = p.parse_homeostat_text("require slack != neg\n")
        cands = [
            {"name": "bad", "ctx": {"slack": "neg"}},
            {"name": "ok", "ctx": {"slack": "ok"}},
        ]
        out = p.apply_homeostat(hp, cands)
        self.assertEqual([c["name"] for c in out], ["ok"])

    def test_phenom_recommendation_stays_in_allowed_set(self) -> None:
        self.st.phenomenology["mode"] = "phi"
        sig = "skola|DRIFT|other|EMVMDNGL"
        self.st.phenomenology["signatures"] = {
            sig: {
                "n": 10,
                "actions": {
                    "OUTSIDE": {"rated_helped": 10, "helped_yes": 10, "burden_n": 10, "burden_sum": 10, "tries": 10, "procrastination_events": 0, "high_burden_n": 0},
                    "ALLOWED": {"rated_helped": 10, "helped_yes": 7, "burden_n": 10, "burden_sum": 20, "tries": 10, "procrastination_events": 0, "high_burden_n": 0},
                },
            }
        }
        rec = p.phenomenology_recommendation(self.st, sig, ["ALLOWED"], current_ctx={"domain": "skola", "label": "DRIFT", "trigger": "other"})
        self.assertIsNotNone(rec)
        self.assertEqual(rec[0], "ALLOWED")

    def test_dag_sparse_data_emits_note(self) -> None:
        self.st.phenomenology["mode"] = "dag"
        self.st.phenomenology["min_samples"] = 1
        sig = "skola|DRIFT|other|EMVMDNGL"
        self.st.phenomenology["signatures"] = {
            sig: {
                "n": 2,
                "actions": {
                    "A": {"rated_helped": 2, "helped_yes": 1, "burden_n": 2, "burden_sum": 3, "tries": 2, "procrastination_events": 0, "high_burden_n": 0},
                },
            }
        }
        rec = p.phenomenology_recommendation(self.st, sig, ["A"], current_ctx={"domain": "skola", "label": "DRIFT", "trigger": "other"})
        self.assertIsNotNone(rec)
        comps = rec[3]
        self.assertIn("dag_notes", comps)
        notes = " ".join(comps.get("dag_notes") or [])
        self.assertIn("DAG fallback", notes)

    def test_allocate_minutes_budget_is_fixed_total(self) -> None:
        self.st.tasks = [
            p.Task(id="a", domain="x", name="A", deadline="2026-03-01", total_minutes=300, remaining_minutes=300, tmt_bins=p.TMTBins(), recent_drift_count=0),
            p.Task(id="b", domain="x", name="B", deadline="2026-04-01", total_minutes=300, remaining_minutes=300, tmt_bins=p.TMTBins(), recent_drift_count=0),
        ]
        alloc = p.allocate_minutes_budget(self.st, 90)
        self.assertLessEqual(sum(alloc.values()), 90)
        self.assertGreaterEqual(sum(alloc.values()), 1)

    def test_recompute_habit_from_logs(self) -> None:
        y = (p._today() - p.dt.timedelta(days=1)).isoformat()
        p.LOG_FILE.write_text(
            f'{{"ts":"{y}T10:00:00","kind":"log_minutes","minutes":30}}\n',
            encoding="utf-8",
        )
        p.recompute_habit_from_logs(self.st)
        self.assertFalse(self.st.today_min_done)
        self.assertEqual(self.st.streak, 1)

    def test_system2_switch_damping(self) -> None:
        self.st.system2["enabled"] = True
        self.st.system2["min_task_stick_min"] = 30
        recs = []
        ok, why = p.system2_switch_decision(self.st, recs, "a", "b")
        self.assertFalse(ok)
        self.assertIn("min_task_stick_min", why)


if __name__ == "__main__":
    unittest.main()
