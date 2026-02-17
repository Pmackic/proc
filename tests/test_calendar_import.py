import tempfile
import unittest
from pathlib import Path

import prok_v5 as p


class TestCalendarImport(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        p.STATE_FILE = Path(self.tmp.name) / "state.json"
        p.LOG_FILE = Path(self.tmp.name) / "log.jsonl"
        self.st = p.ProkState()
        p._init_defaults(self.st)
        self.st.plan_block_min = 50
        self.st.day_start = "08:00"
        self.st.day_end = "22:00"
        self.st.max_blocks_per_day = 20

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_ics_import_creates_datetime_blackout(self) -> None:
        ics = (
            "BEGIN:VCALENDAR\n"
            "BEGIN:VEVENT\n"
            "UID:abc-1\n"
            "SUMMARY:Lecture\n"
            "DTSTART:20260302T100000\n"
            "DTEND:20260302T120000\n"
            "END:VEVENT\n"
            "END:VCALENDAR\n"
        )
        pth = Path(self.tmp.name) / "x.ics"
        pth.write_text(ics, encoding="utf-8")
        out = p.parse_ics_to_blackouts(pth)
        self.assertEqual(len(out["datetime"]), 1)
        self.st.datetime_blackouts.extend(out["datetime"])
        d = p.dt.date(2026, 3, 2)
        # 08:00-22:00 is 14h=840m; minus 2h=120m => 720m => 14 blocks of 50
        self.assertEqual(p.day_capacity_blocks(self.st, d), 14)


if __name__ == "__main__":
    unittest.main()
