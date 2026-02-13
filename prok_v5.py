#!/usr/bin/env python3
"""PROK v5 — multi-task Ashby regulator with per-task TMT + domain recursion + algedonic alarm.

Upgrade notes (v4 -> v5):
- Single-plan fields replaced with tasks[] + active_task_id (auto-migrated on load).
- Planner allocates blocks across tasks with per-task slack summary.
- Session/logging now target a task; per-domain drift EMA and recovery stats tracked.
- Adds algedonic alarm (ok/warning/critical) from real slack/drift signals.

Stdlib-only.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import datetime as dt
import json
import math
import random
import re
import statistics
import sys
import time
import select
import zipfile

STATE_FILE = Path(".prok_state.json")
LOG_FILE = Path(".prok_log.jsonl")

# ----------------------------
# Helpers
# ----------------------------

def _to_int(v: Any, default: int) -> int:
    try:
        if v is None:
            return default
        return int(v)
    except Exception:
        return default

def _to_float(v: Any, default: float) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default

def _to_str(v: Any, default: str) -> str:
    return default if v is None else str(v)

def _today() -> dt.date:
    return dt.date.today()

def _iso_today() -> str:
    return _today().isoformat()

def _now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")

def _clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))

def _json_dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)

def _log_event(kind: str, payload: Dict[str, Any]) -> None:
    rec = {"ts": _now_iso(), "kind": kind, **payload}
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def _parse_hhmm(s: str) -> int:
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", s.strip())
    if not m:
        raise ValueError(f"Invalid time '{s}'. Use HH:MM")
    hh = int(m.group(1)); mm = int(m.group(2))
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        raise ValueError(f"Invalid time '{s}'. Use HH:MM")
    return hh * 60 + mm

def _parse_deadline(s: str) -> str:
    s = s.strip().lower()
    m = re.fullmatch(r"(\d+)\s*(d|w|mo)", s)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        base = _today()
        if unit == "d":
            d = base + dt.timedelta(days=n)
        elif unit == "w":
            d = base + dt.timedelta(days=7*n)
        else:
            y = base.year
            mo = base.month + n
            y += (mo - 1) // 12
            mo = ((mo - 1) % 12) + 1
            dim = [31, 29 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 28,
                   31, 30, 31, 30, 31, 31, 30, 31, 30, 31][mo-1]
            day = min(base.day, dim)
            d = dt.date(y, mo, day)
        return d.isoformat()
    dt.date.fromisoformat(s)
    return s

def _daterange(a: dt.date, b: dt.date) -> List[dt.date]:
    out = []
    cur = a
    while cur <= b:
        out.append(cur)
        cur += dt.timedelta(days=1)
    return out

# ----------------------------
# Blackouts
# ----------------------------

@dataclass
class WeeklyBlackout:
    weekday_start: int  # 0=Mon..6=Sun
    weekday_end: int
    start_hhmm: str
    end_hhmm: str
    label: str = ""

@dataclass
class DateBlackout:
    date: str  # YYYY-MM-DD
    label: str = ""

@dataclass
class DateRangeBlackout:
    start: str
    end: str
    label: str = ""

# ----------------------------
# Discrete TMT + Haghbin
# ----------------------------

TMT_E = ("L", "M", "H")
TMT_V = ("L", "M", "H")
TMT_D = ("N", "F")   # near/far
TMT_G = ("L", "H")   # gamma low/high

@dataclass
class TMTBins:
    E: str = "M"
    V: str = "M"
    D: str = "N"
    G: str = "L"

    def sanitize(self) -> "TMTBins":
        self.E = self.E if self.E in TMT_E else "M"
        self.V = self.V if self.V in TMT_V else "M"
        self.D = self.D if self.D in TMT_D else "N"
        self.G = self.G if self.G in TMT_G else "L"
        return self

def _bin_down(b: str, axis: Tuple[str, ...]) -> str:
    i = axis.index(b) if b in axis else 1
    return axis[max(0, i-1)]

def _bin_up(b: str, axis: Tuple[str, ...]) -> str:
    i = axis.index(b) if b in axis else 1
    return axis[min(len(axis)-1, i+1)]

@dataclass
class HaghbinFlags:
    intent: int = 1          # I
    expected_cost: int = 1   # C
    irrational_delay: int = 1# N
    negative_affect: int = 0 # F

def haghbin_label(flags: HaghbinFlags) -> str:
    if flags.intent == 1 and flags.expected_cost == 1 and flags.irrational_delay == 1:
        return "PROCRASTINATION"
    if flags.intent == 1:
        return "DRIFT"
    return "NO_INTENT"

TRIGGERS = ("confusion", "boredom", "distraction", "anxiety", "fatigue", "other")

# ----------------------------
# User fitness profile (rho) + language pack (lambda)
# ----------------------------

DEFAULT_GOALS = ["feasible", "recover_fast", "low_annoyance", "progress", "streak_up"]
DEFAULT_LANG = {
    "state.engaged": "engaged",
    "state.drift": "drifting",
    "state.procrastinate": "procrastinating",
    "state.blocked": "blocked",
    "state.paused": "paused",
    "prompt.check": "CHECK",
    "prompt.you_can_report": "You can report anytime: d=drift, p=procrastinate, b=blocked(pause), r=resume, q=quit, c=check-now",
    "trigger.confusion": "confusion",
    "trigger.boredom": "boredom",
    "trigger.distraction": "distraction",
    "trigger.anxiety": "anxiety",
    "trigger.fatigue": "fatigue",
    "trigger.other": "other",
}

def lang_get(lang: Dict[str, str], key: str, default: str) -> str:
    return str((lang or {}).get(key) or default)

def default_goal_profile() -> Dict[str, Any]:
    # Lexicographic tiers; no numeric weights.
    return {
        "tiers": [
            ["feasible"],
            ["recover_fast"],
            ["low_annoyance"],
            ["progress"],
        ],
        "annoyance_max_checks": 999,  # soft constraint; can be lowered in study
    }

def default_phenomenology() -> Dict[str, Any]:
    # Mathematical phenomenology: classify episodes by invariant structure
    # and transfer successful recoveries to homologous cases.
    return {
        "enabled": True,
        "min_samples": 3,
        "influence": 0.35,
        "signatures": {},
    }

# ----------------------------
# Genome (NK-ready policy knobs)
# ----------------------------

def default_genome() -> Dict[str, Any]:
    return {
        "version": 2,
        "g1_block_map": {
            "L,L,F,H": 10,
            "L,M,F,H": 10,
            "M,M,F,H": 25,
            "H,M,F,H": 25,
            "M,H,N,L": 50,
        },
        "g2_check_map": {
            "L,L,F,H": [2, 5],
            "L,M,F,H": [2, 5],
            "M,M,F,H": [4, 8],
            "M,M,N,L": [6, 14],
            "H,H,N,L": [8, 16],
        },
        "g3_tightening": "mild",          # none|mild|strong
        "g4_classifier": "balanced",      # lenient|balanced|strict
        "g5_recovery_routing": "trigger", # E-first|V-first|trigger|mixed
        "g6_enforce_near_reward": True,
        "g7_artifact": "5bullets",        # none|3problems|5flashcards|5bullets
        "g8_replan_threshold": 1,           # drift events before forced replan
        "g9_pause_policy": "soft",        # soft|hard
        "g10_explore": "off",             # off|low|med|high
    }

def _theta_key(theta: TMTBins) -> str:
    return f"{theta.E},{theta.V},{theta.D},{theta.G}"

def genome_block_minutes(genome: Dict[str, Any], theta: TMTBins, plan_block_min: int) -> int:
    m = (genome.get("g1_block_map") or {})
    key = _theta_key(theta)
    if key in m:
        return _to_int(m[key], plan_block_min)
    if theta.G == "H" or theta.D == "F":
        return 25
    if theta.E == "L" or theta.V == "L":
        return 35
    return plan_block_min

def genome_check_gaps(genome: Dict[str, Any], theta: TMTBins) -> Tuple[int, int]:
    m = (genome.get("g2_check_map") or {})
    key = _theta_key(theta)
    if key in m:
        arr = m[key]
        return _to_int(arr[0], 6), _to_int(arr[1], 14)
    if theta.G == "H":
        return 2, 5
    if theta.D == "F":
        return 4, 8
    return 6, 14

def genome_tighten(genome: Dict[str, Any], min_gap: int, max_gap: int) -> Tuple[int, int]:
    mode = str(genome.get("g3_tightening") or "mild").lower()
    if mode == "none":
        return min_gap, max_gap
    if mode == "strong":
        return max(2, min_gap - 2), max(max(4, min_gap + 2), max_gap - 3)
    return max(2, min_gap - 1), max(max(4, min_gap + 2), max_gap - 2)

# ----------------------------
# Fitness DSL (prokfit) — minimal parser/evaluator
# ----------------------------

class FitParseError(Exception):
    pass

Token = Tuple[str, str]

def _tokenize(src: str) -> List[Token]:
    specs = [
        ("NEWLINE", r"\n"),
        ("SKIP", r"[ \t\r]+"),
        ("INT", r"\d+"),
        ("LBRACE", r"\{"),
        ("RBRACE", r"\}"),
        ("LPAREN", r"\("),
        ("RPAREN", r"\)"),
        ("COMMA", r","),
        ("OP", r"!=|=|in\b"),
        ("KW", r"\brequire\b|\bpriority\b|\bprefer\b|\bavoid\b|\band\b|\bor\b|\bnot\b"),
        ("IDENT", r"[A-Za-z_][A-Za-z0-9_]*"),
        ("COLON", r":"),
    ]
    rx = re.compile("|".join(f"(?P<{k}>{v})" for k, v in specs))
    out: List[Token] = []
    for m in rx.finditer(src):
        kind = m.lastgroup
        val = m.group()
        if kind == "SKIP":
            continue
        if kind == "KW":
            out.append(("KW", val))
        elif kind == "OP":
            out.append(("OP", val))
        else:
            out.append((kind, val))
    out.append(("EOF", "EOF"))
    return out

@dataclass
class Expr:
    kind: str
    value: Any = None
    left: Optional["Expr"] = None
    right: Optional["Expr"] = None

@dataclass
class PreferStmt:
    want: str
    expr: Expr

@dataclass
class ProkFitProgram:
    requires: List[Expr]
    priorities: List[List[PreferStmt]]

class _Parser:
    def __init__(self, toks: List[Token]) -> None:
        self.toks = toks
        self.i = 0

    def peek(self) -> Token:
        return self.toks[self.i]

    def pop(self) -> Token:
        t = self.toks[self.i]
        self.i += 1
        return t

    def accept(self, typ: str, val: Optional[str] = None) -> bool:
        t = self.peek()
        if t[0] != typ:
            return False
        if val is not None and t[1] != val:
            return False
        self.pop()
        return True

    def expect(self, typ: str, val: Optional[str] = None) -> Token:
        t = self.peek()
        if t[0] != typ or (val is not None and t[1] != val):
            raise FitParseError(f"Expected {typ}{'='+val if val else ''}, got {t}")
        return self.pop()

    def parse(self) -> ProkFitProgram:
        requires: List[Expr] = []
        priorities: List[List[PreferStmt]] = []
        while self.peek()[0] != "EOF":
            while self.accept("NEWLINE"):
                pass
            if self.peek()[0] == "EOF":
                break
            t = self.peek()
            if t == ("KW", "require"):
                self.pop()
                requires.append(self.parse_expr())
                while self.accept("NEWLINE"):
                    pass
            elif t == ("KW", "priority"):
                self.pop()
                self.expect("INT")
                self.expect("COLON")
                if not self.accept("NEWLINE"):
                    raise FitParseError("Expected newline after 'priority N:'")
                block: List[PreferStmt] = []
                while True:
                    while self.accept("NEWLINE"):
                        pass
                    if self.peek()[0] == "EOF":
                        break
                    if self.peek() in (("KW","priority"),("KW","require")):
                        break
                    kw = self.expect("KW")[1]
                    if kw not in ("prefer","avoid"):
                        raise FitParseError("Inside priority blocks only 'prefer'/'avoid' allowed")
                    expr = self.parse_expr()
                    block.append(PreferStmt(kw, expr))
                    while self.accept("NEWLINE"):
                        pass
                priorities.append(block)
            else:
                raise FitParseError(f"Unexpected token {t}")
        return ProkFitProgram(requires, priorities)

    def parse_expr(self) -> Expr:
        node = self.parse_term()
        while True:
            if self.accept("KW","and"):
                rhs = self.parse_term()
                node = Expr("and", left=node, right=rhs)
            elif self.accept("KW","or"):
                rhs = self.parse_term()
                node = Expr("or", left=node, right=rhs)
            else:
                break
        return node

    def parse_term(self) -> Expr:
        if self.accept("KW","not"):
            return Expr("not", left=self.parse_term())
        if self.accept("LPAREN"):
            e = self.parse_expr()
            self.expect("RPAREN")
            return e
        ident = self.expect("IDENT")[1]
        if self.peek()[0] != "OP":
            return Expr("pred", value=(ident, None, None))
        op = self.expect("OP")[1]
        if op == "in":
            self.expect("LBRACE")
            vals = [self.expect("IDENT")[1]]
            while self.accept("COMMA"):
                vals.append(self.expect("IDENT")[1])
            self.expect("RBRACE")
            return Expr("pred", value=(ident, "in", vals))
        val = self.expect("IDENT")[1]
        return Expr("pred", value=(ident, op, val))

def parse_prokfit(path: Path) -> ProkFitProgram:
    toks = _tokenize(path.read_text(encoding="utf-8"))
    return _Parser(toks).parse()

def eval_expr(expr: Expr, ctx: Dict[str, Any]) -> bool:
    if expr.kind == "and":
        return eval_expr(expr.left, ctx) and eval_expr(expr.right, ctx)
    if expr.kind == "or":
        return eval_expr(expr.left, ctx) or eval_expr(expr.right, ctx)
    if expr.kind == "not":
        return not eval_expr(expr.left, ctx)
    if expr.kind == "pred":
        name, op, val = expr.value
        cur = ctx.get(name)
        if op is None:
            return bool(cur)
        if op == "=":
            return str(cur) == str(val)
        if op == "!=":
            return str(cur) != str(val)
        if op == "in":
            return str(cur) in set(map(str, val))
    return False

def rank_candidates(program: ProkFitProgram, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    scored = rank_candidates_scored(program, candidates)
    return scored[0][1]

def rank_candidates_scored(program: ProkFitProgram, candidates: List[Dict[str, Any]]) -> List[Tuple[Tuple[int, ...], Dict[str, Any]]]:
    def passes_requires(ctx: Dict[str, Any]) -> bool:
        return all(eval_expr(r, ctx) for r in program.requires)

    use_requires = bool(program.requires) and any(passes_requires(c["ctx"]) for c in candidates)

    scored: List[Tuple[Tuple[int, ...], Dict[str, Any]]] = []
    for c in candidates:
        if use_requires and not passes_requires(c["ctx"]):
            continue
        lex: List[int] = []
        for block in program.priorities:
            penalties = 0
            bonuses = 0
            for st in block:
                ok = eval_expr(st.expr, c["ctx"])
                if st.want == "avoid" and ok:
                    penalties += 1
                if st.want == "prefer" and ok:
                    bonuses += 1
            lex.extend([penalties, -bonuses])
        scored.append((tuple(lex), c))

    if not scored:
        for c in candidates:
            lex = []
            for block in program.priorities:
                penalties = 0
                bonuses = 0
                for st in block:
                    ok = eval_expr(st.expr, c["ctx"])
                    if st.want == "avoid" and ok:
                        penalties += 1
                    if st.want == "prefer" and ok:
                        bonuses += 1
                lex.extend([penalties, -bonuses])
            scored.append((tuple(lex), c))

    scored.sort(key=lambda x: x[0])
    return scored

def built_in_fitness() -> ProkFitProgram:
    src = """require slack != neg

priority 1:
avoid procrastinating
avoid (intent_gap and expected_cost)

priority 2:
prefer recover_fast
prefer reward_near
avoid annoying
"""
    return _Parser(_tokenize(src)).parse()

def fitness_from_goal_profile(base: ProkFitProgram, goal: Dict[str, Any]) -> ProkFitProgram:
    # Insert a small bias: if low_annoyance is in early tiers, move 'avoid annoying' earlier.
    tiers = goal.get("tiers") or []
    early = set()
    for i, tier in enumerate(tiers[:2]):
        for x in tier:
            early.add(str(x))
    if "low_annoyance" not in early:
        return base

    # Deep-ish copy
    req = list(base.requires)
    pr = [list(block) for block in base.priorities]
    # Ensure at least one block exists
    if not pr:
        pr = [[]]
    # Put avoid annoying into priority 1 if not already present
    found = False
    for block in pr:
        for st in block:
            if st.want == "avoid" and st.expr.kind == "pred" and st.expr.value[0] == "annoying":
                found = True
    if not found:
        pr[0].append(PreferStmt("avoid", Expr("pred", value=("annoying", None, None))))
    return ProkFitProgram(req, pr)

# ----------------------------
# CCA placeholder (OFF)
# ----------------------------

def cca_map_placeholder(features: Dict[str, Any]) -> Dict[str, Any]:
    # v4 requirement: identity/no-op.
    return {}

# ----------------------------
# PROK state
# ----------------------------

@dataclass
class Task:
    id: str
    domain: str
    name: str
    deadline: Optional[str]
    total_minutes: int
    remaining_minutes: int
    tmt_bins: TMTBins = None
    recent_drift_count: int = 0

@dataclass
class DomainState:
    preferred_block_min: Optional[int] = None
    preferred_check_min_gap: Optional[int] = None
    preferred_check_max_gap: Optional[int] = None
    drift_ema: float = 0.0
    recovery_counts: Dict[str, int] = None

@dataclass
class ProkState:
    # Planning
    tasks: List[Task] = None
    active_task_id: Optional[str] = None
    plan_block_min: int = 50
    tiny_min: int = 10

    # Day constraints
    day_start: str = "08:00"
    day_end: str = "22:00"
    max_blocks_per_day: int = 6

    # Blackouts
    weekly_blackouts: List[WeeklyBlackout] = None
    date_blackouts: List[DateBlackout] = None
    daterange_blackouts: List[DateRangeBlackout] = None

    # Habit/momentum
    today_min_done: bool = False
    last_min_date: Optional[str] = None
    streak: int = 0
    last_streak_date: Optional[str] = None

    # Control layers
    genome: Dict[str, Any] = None
    fitness_path: Optional[str] = None

    # Domain recursion
    domains: Dict[str, DomainState] = None

    # NEW v4: study + goals + language + cca placeholder
    participant_id: str = ""
    study_mode: bool = False
    goal_profile: Dict[str, Any] = None
    language_pack: Dict[str, str] = None
    phenomenology: Dict[str, Any] = None
    use_cca: bool = False
    cca_model_path: Optional[str] = None

def _init_defaults(st: ProkState) -> None:
    if st.weekly_blackouts is None: st.weekly_blackouts = []
    if st.date_blackouts is None: st.date_blackouts = []
    if st.daterange_blackouts is None: st.daterange_blackouts = []
    if st.tasks is None: st.tasks = []
    if st.genome is None: st.genome = default_genome()
    if "version" not in st.genome: st.genome = default_genome()
    if st.goal_profile is None: st.goal_profile = default_goal_profile()
    if st.language_pack is None: st.language_pack = dict(DEFAULT_LANG)
    if st.phenomenology is None: st.phenomenology = default_phenomenology()
    if not isinstance(st.phenomenology, dict): st.phenomenology = default_phenomenology()
    if "enabled" not in st.phenomenology: st.phenomenology["enabled"] = True
    if "min_samples" not in st.phenomenology: st.phenomenology["min_samples"] = 3
    if "influence" not in st.phenomenology: st.phenomenology["influence"] = 0.35
    if "signatures" not in st.phenomenology or not isinstance(st.phenomenology.get("signatures"), dict):
        st.phenomenology["signatures"] = {}
    if st.domains is None: st.domains = {}

    for t in st.tasks:
        if t.tmt_bins is None:
            t.tmt_bins = TMTBins()
        t.tmt_bins.sanitize()

    for d, ds in (st.domains or {}).items():
        if ds.recovery_counts is None:
            ds.recovery_counts = {}

def _slugify(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", (s or "").strip().lower()).strip("-")
    return s or "task"

def _make_task_id(name: str) -> str:
    base = _slugify(name)
    return f"{base}-{random.randint(1000,9999)}"

def _load_tasks(data: Dict[str, Any]) -> List[Task]:
    raw = data.get("tasks") or []
    out: List[Task] = []
    for t in raw:
        tb = TMTBins(**(t.get("tmt_bins") or {}))
        out.append(Task(
            id=_to_str(t.get("id"), _make_task_id(t.get("name") or "task")),
            domain=_to_str(t.get("domain"), "general"),
            name=_to_str(t.get("name"), "task"),
            deadline=t.get("deadline"),
            total_minutes=_to_int(t.get("total_minutes"), 0),
            remaining_minutes=_to_int(t.get("remaining_minutes"), 0),
            tmt_bins=tb,
            recent_drift_count=_to_int(t.get("recent_drift_count"), 0),
        ))
    return out

def _load_domains(data: Dict[str, Any]) -> Dict[str, DomainState]:
    raw = data.get("domains") or {}
    out: Dict[str, DomainState] = {}
    for k, v in raw.items():
        out[str(k)] = DomainState(
            preferred_block_min=v.get("preferred_block_min"),
            preferred_check_min_gap=v.get("preferred_check_min_gap"),
            preferred_check_max_gap=v.get("preferred_check_max_gap"),
            drift_ema=_to_float(v.get("drift_ema"), 0.0),
            recovery_counts=(v.get("recovery_counts") or {}),
        )
    return out

def load_state() -> ProkState:
    if not STATE_FILE.exists():
        st = ProkState()
        _init_defaults(st)
        return st
    data = json.loads(STATE_FILE.read_text(encoding="utf-8"))

    st = ProkState(
        tasks=_load_tasks(data),
        active_task_id=data.get("active_task_id"),
        plan_block_min=_to_int(data.get("plan_block_min", data.get("block_min")), 50),
        tiny_min=_to_int(data.get("tiny_min"), 10),
        day_start=_to_str(data.get("day_start"), "08:00"),
        day_end=_to_str(data.get("day_end"), "22:00"),
        max_blocks_per_day=_to_int(data.get("max_blocks_per_day"), 6),
        weekly_blackouts=[WeeklyBlackout(**x) for x in data.get("weekly_blackouts", [])],
        date_blackouts=[DateBlackout(**x) for x in data.get("date_blackouts", [])],
        daterange_blackouts=[DateRangeBlackout(**x) for x in data.get("daterange_blackouts", [])],
        today_min_done=bool(data.get("today_min_done", False)),
        last_min_date=data.get("last_min_date"),
        streak=_to_int(data.get("streak"), 0),
        last_streak_date=data.get("last_streak_date"),
        genome=(data.get("genome") or None),
        fitness_path=data.get("fitness_path"),
        domains=_load_domains(data),
        participant_id=_to_str(data.get("participant_id"), ""),
        study_mode=bool(data.get("study_mode", False)),
        goal_profile=(data.get("goal_profile") or None),
        language_pack=(data.get("language_pack") or None),
        phenomenology=(data.get("phenomenology") or None),
        use_cca=bool(data.get("use_cca", False)),
        cca_model_path=data.get("cca_model_path"),
    )
    _init_defaults(st)

    # Migration from v4 single-plan (if present)
    if not st.tasks and data.get("domain") is not None:
        domain = _to_str(data.get("domain"), "general")
        deadline = data.get("deadline")
        total_minutes = _to_int(data.get("total_minutes"), 0)
        remaining_minutes = _to_int(data.get("remaining_minutes"), 0)
        if total_minutes == 0 and data.get("total_blocks") is not None:
            total_minutes = _to_int(data.get("total_blocks"), 0) * st.plan_block_min
        if remaining_minutes == 0 and data.get("remaining_blocks") is not None:
            remaining_minutes = _to_int(data.get("remaining_blocks"), 0) * st.plan_block_min
        tmt_bins = TMTBins(**(data.get("tmt_bins") or {}))
        recent_drift = _to_int(data.get("recent_drift_count"), 0)
        task = Task(
            id="task-1",
            domain=domain,
            name=domain,
            deadline=deadline,
            total_minutes=total_minutes,
            remaining_minutes=remaining_minutes,
            tmt_bins=tmt_bins,
            recent_drift_count=recent_drift,
        )
        st.tasks = [task]
        st.active_task_id = "task-1"

    # v4 requirement: cca off unless explicitly enabled (even if old state had it)
    if st.use_cca:
        pass

    _init_defaults(st)
    return st

def save_state(st: ProkState) -> None:
    _init_defaults(st)
    STATE_FILE.write_text(_json_dump(asdict(st)), encoding="utf-8")

def ensure_today(st: ProkState) -> None:
    td = _iso_today()
    if st.last_min_date != td:
        st.today_min_done = False
        st.last_min_date = td

def get_domain_state(st: ProkState, domain: str) -> DomainState:
    d = (domain or "general").lower()
    if st.domains is None:
        st.domains = {}
    if d not in st.domains:
        st.domains[d] = DomainState()
    if st.domains[d].recovery_counts is None:
        st.domains[d].recovery_counts = {}
    return st.domains[d]

def get_task_by_id(st: ProkState, task_id: str) -> Optional[Task]:
    for t in st.tasks:
        if t.id == task_id:
            return t
    return None

def resolve_task_by_name(st: ProkState, domain: Optional[str], name_query: Optional[str]) -> Optional[Task]:
    if not name_query:
        return None
    q = name_query.strip().lower()
    matches = []
    for t in st.tasks:
        if domain and t.domain.lower() != domain.lower():
            continue
        if q in t.name.lower():
            matches.append(t)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = ", ".join(f"{m.id}:{m.name}" for m in matches)
        raise SystemExit(f"Ambiguous task name; matches: {names}")
    return None

def select_active_task(st: ProkState, task_id: Optional[str], domain: Optional[str], name_query: Optional[str]) -> Task:
    if task_id:
        t = get_task_by_id(st, task_id)
        if not t:
            raise SystemExit(f"Task not found: {task_id}")
        st.active_task_id = t.id
        return t
    if domain or name_query:
        t = resolve_task_by_name(st, domain, name_query)
        if not t:
            raise SystemExit("No task matches domain/name query.")
        st.active_task_id = t.id
        return t
    if st.active_task_id:
        t = get_task_by_id(st, st.active_task_id)
        if t:
            return t
    if len(st.tasks) == 1:
        st.active_task_id = st.tasks[0].id
        return st.tasks[0]
    raise SystemExit("No active task. Use: prok task set-active <id>")

def domain_block_minutes(st: ProkState, domain: str, theta: TMTBins) -> int:
    ds = get_domain_state(st, domain)
    if ds.preferred_block_min is not None:
        return _to_int(ds.preferred_block_min, st.plan_block_min)
    return genome_block_minutes(st.genome, theta, st.plan_block_min)

def domain_check_gaps(st: ProkState, domain: str, theta: TMTBins) -> Tuple[int, int]:
    ds = get_domain_state(st, domain)
    if ds.preferred_check_min_gap is not None and ds.preferred_check_max_gap is not None:
        return _to_int(ds.preferred_check_min_gap, 6), _to_int(ds.preferred_check_max_gap, 14)
    return genome_check_gaps(st.genome, theta)

def update_domain_drift(ds: DomainState, drifted: bool) -> None:
    alpha = 0.2
    target = 1.0 if drifted else 0.0
    ds.drift_ema = (1 - alpha) * float(ds.drift_ema) + alpha * target

# ----------------------------
# Mathematical phenomenology
# ----------------------------

def phenomenology_signature(task: Task, label: str, trigger: str) -> str:
    theta = (task.tmt_bins or TMTBins()).sanitize()
    dom = (task.domain or "general").lower()
    return f"{dom}|{label}|{trigger}|E{theta.E}V{theta.V}D{theta.D}G{theta.G}"

def _phenomenology_action_score(stats: Dict[str, Any]) -> Tuple[float, int]:
    rated = _to_int(stats.get("rated_helped"), 0)
    yes = _to_int(stats.get("helped_yes"), 0)
    burden_n = _to_int(stats.get("burden_n"), 0)
    burden_sum = _to_float(stats.get("burden_sum"), 0.0)

    success = (yes + 1.0) / (rated + 2.0)  # Laplace smoothing
    if burden_n > 0:
        burden_avg = burden_sum / max(1, burden_n)
        burden_penalty = (burden_avg - 1.0) / 4.0
    else:
        burden_penalty = 0.0
    return success - 0.2 * burden_penalty, rated

def phenomenology_recommendation(
    st: ProkState,
    signature: str,
    candidate_names: List[str],
) -> Optional[Tuple[str, float, int]]:
    ph = st.phenomenology or {}
    if not bool(ph.get("enabled", True)):
        return None
    min_samples = max(1, _to_int(ph.get("min_samples"), 3))
    entry = (ph.get("signatures") or {}).get(signature) or {}
    actions = entry.get("actions") or {}
    allowed = set(candidate_names)

    best_name = None
    best_score = -1e9
    best_n = 0
    for name, stats in actions.items():
        if name not in allowed:
            continue
        score, rated = _phenomenology_action_score(stats)
        if rated < min_samples:
            continue
        if score > best_score:
            best_name = name
            best_score = score
            best_n = rated
    if not best_name:
        return None
    return best_name, best_score, best_n

def phenomenology_record_outcome(
    st: ProkState,
    signature: str,
    chosen: str,
    ratings: Dict[str, Any],
) -> None:
    ph = st.phenomenology or {}
    sigs = ph.setdefault("signatures", {})
    entry = sigs.setdefault(signature, {"n": 0, "actions": {}})
    entry["n"] = _to_int(entry.get("n"), 0) + 1
    actions = entry.setdefault("actions", {})
    a = actions.setdefault(chosen, {
        "tries": 0,
        "rated_helped": 0,
        "helped_yes": 0,
        "burden_n": 0,
        "burden_sum": 0.0,
    })
    a["tries"] = _to_int(a.get("tries"), 0) + 1
    if ratings.get("helped") is not None:
        a["rated_helped"] = _to_int(a.get("rated_helped"), 0) + 1
        if bool(ratings.get("helped")):
            a["helped_yes"] = _to_int(a.get("helped_yes"), 0) + 1
    if ratings.get("burden") is not None:
        a["burden_n"] = _to_int(a.get("burden_n"), 0) + 1
        a["burden_sum"] = _to_float(a.get("burden_sum"), 0.0) + float(ratings.get("burden"))

def _load_log_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out

def _legacy_signature_from_event(ev: Dict[str, Any]) -> str:
    domain = str(ev.get("domain") or "unknown").lower()
    label = str(ev.get("label") or "UNKNOWN")
    trigger = str(ev.get("trigger") or "other")
    th = ev.get("theta") or ev.get("theta_next") or {}
    E = str(th.get("E") or "M")
    V = str(th.get("V") or "M")
    D = str(th.get("D") or "N")
    G = str(th.get("G") or "L")
    return f"{domain}|{label}|{trigger}|E{E}V{V}D{D}G{G}"

def _safe_rate(num: int, den: int) -> Optional[float]:
    return (float(num) / float(den)) if den > 0 else None

def _safe_avg(vals: List[float]) -> Optional[float]:
    return (sum(vals) / float(len(vals))) if vals else None

def _fmt_num(x: Optional[float], digits: int = 2) -> str:
    if x is None:
        return "N/A"
    return f"{x:.{digits}f}"

def _gini(vals: List[int]) -> Optional[float]:
    if not vals:
        return None
    xs = [float(v) for v in vals if v >= 0]
    if not xs:
        return None
    s = sum(xs)
    if s <= 0:
        return 0.0
    xs.sort()
    n = len(xs)
    wsum = 0.0
    for i, x in enumerate(xs, 1):
        wsum += i * x
    return ((2.0 * wsum) / (n * s)) - ((n + 1.0) / n)

def build_phenom_audit_markdown(st: ProkState, records: List[Dict[str, Any]], log_path: Path) -> str:
    events = [r for r in records if str(r.get("kind")) in ("course_correct", "d", "p")]
    n_events = len(events)

    pids = sorted({str(r.get("participant_id")) for r in records if r.get("participant_id") not in (None, "")})
    sess = sorted({str(r.get("session_id")) for r in records if r.get("session_id") not in (None, "")})
    ts = [str(r.get("ts")) for r in records if r.get("ts")]
    window = "N/A"
    if ts:
        ds = sorted(t[:10] for t in ts if len(t) >= 10)
        if ds:
            window = f"{ds[0]} to {ds[-1]}"

    ph = st.phenomenology or default_phenomenology()
    n_min = max(1, _to_int(ph.get("min_samples"), 3))
    tau = _to_float(ph.get("influence"), 0.35)
    sig_field_present = sum(1 for e in events if e.get("phenomenology_signature"))
    sel_field_present = sum(1 for e in events if e.get("selection_source"))

    sigs = [str(e.get("phenomenology_signature") or _legacy_signature_from_event(e)) for e in events]
    sig_counts = Counter(sigs)
    sig_vals = sorted(sig_counts.values())
    qual_events = sum(1 for s in sigs if sig_counts.get(s, 0) >= n_min)
    median_sig = statistics.median(sig_vals) if sig_vals else None
    p90_sig = sig_vals[int((len(sig_vals) - 1) * 0.9)] if sig_vals else None
    gini = _gini(sig_vals)

    helped_vals = [e.get("helped") for e in events if e.get("helped") is not None]
    helped_yes = sum(1 for x in helped_vals if bool(x))
    burden_vals = [float(e.get("burden")) for e in events if isinstance(e.get("burden"), (int, float))]

    rows_t1: Dict[str, Dict[str, Any]] = {}
    for sig in sig_counts:
        rows_t1[sig] = {"events_n": sig_counts[sig], "actions": defaultdict(lambda: {"rated_helped": 0, "helped_yes": 0, "burden": []})}
    for e in events:
        sig = str(e.get("phenomenology_signature") or _legacy_signature_from_event(e))
        action = str(e.get("chosen") or e.get("choice") or "UNKNOWN")
        a = rows_t1[sig]["actions"][action]
        if e.get("helped") is not None:
            a["rated_helped"] += 1
            if bool(e.get("helped")):
                a["helped_yes"] += 1
        if isinstance(e.get("burden"), (int, float)):
            a["burden"].append(float(e.get("burden")))

    def best_action(actions: Dict[str, Any]) -> Tuple[str, int, Optional[float], Optional[float]]:
        best = ("UNKNOWN", -1e9, 0, None, None)
        for name, stx in actions.items():
            score, rated = _phenomenology_action_score({
                "rated_helped": stx.get("rated_helped", 0),
                "helped_yes": stx.get("helped_yes", 0),
                "burden_n": len(stx.get("burden", [])),
                "burden_sum": sum(stx.get("burden", [])),
            })
            hr = _safe_rate(stx.get("helped_yes", 0), stx.get("rated_helped", 0))
            ba = _safe_avg(stx.get("burden", []))
            if score > best[1]:
                best = (name, score, rated, hr, ba)
        return best[0], best[2], best[3], best[4]

    t2 = defaultdict(lambda: {"n": 0, "helped_yes": 0, "helped_n": 0, "burden": []})
    t3 = defaultdict(lambda: {"n": 0, "helped_yes": 0, "helped_n": 0, "burden": []})
    for e in events:
        sel = str(e.get("selection_source") or "fitness (inferred, legacy)")
        t2[sel]["n"] += 1
        domain = str(e.get("domain") or "unknown").lower()
        trig = str(e.get("trigger") or "other")
        t3[(domain, trig, sel)]["n"] += 1
        if e.get("helped") is not None:
            t2[sel]["helped_n"] += 1
            t3[(domain, trig, sel)]["helped_n"] += 1
            if bool(e.get("helped")):
                t2[sel]["helped_yes"] += 1
                t3[(domain, trig, sel)]["helped_yes"] += 1
        if isinstance(e.get("burden"), (int, float)):
            b = float(e.get("burden"))
            t2[sel]["burden"].append(b)
            t3[(domain, trig, sel)]["burden"].append(b)

    phenom_n = t2.get("fitness+phenomenology", {}).get("n", 0)
    override_rate = _safe_rate(phenom_n, n_events)
    notes = []
    if sig_field_present == 0:
        notes.append("- Log events are legacy/pre-phenomenology for intervention events; no `phenomenology_signature` fields are present.")
    if sel_field_present == 0:
        notes.append("- Log events are legacy/pre-phenomenology for intervention events; no `selection_source` fields are present.")
    notes_block = "\n".join(notes) if notes else "- Phenomenology fields detected in intervention events."

    lines: List[str] = []
    lines.append("# PROK v5 Phenomenology Audit Example")
    lines.append("")
    lines.append(f"This example is pre-filled from `{log_path.name}` as of {_iso_today()}.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1) Study metadata")
    lines.append("")
    lines.append("- Study ID: `local-auto-audit`")
    lines.append(f"- Dataset window: `{window}`")
    lines.append(f"- Participants (n): `{len(pids)}`")
    lines.append(f"- Sessions (n): `{len(sess)}`")
    lines.append(f"- Course-correct events (n): `{n_events}`")
    lines.append("- Phenomenology settings at audit time (`.prok_state.json`):")
    lines.append(f"  - `enabled`: `{str(bool(ph.get('enabled', True))).lower()}`")
    lines.append(f"  - `min_samples (n_min)`: `{n_min}`")
    lines.append(f"  - `influence (tau)`: `{tau:.2f}`")
    lines.append("")
    lines.append("Note:")
    lines.append(notes_block)
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2) Core metrics")
    lines.append("")
    lines.append("### A) Coverage and sparsity")
    lines.append("")
    lines.append(f"- Signature count `|Sigma|`: `{len(sig_counts)}`")
    lines.append(f"- Events mapped to signatures: `{n_events} / {n_events} = {_fmt_num(_safe_rate(n_events, n_events))}`" if n_events > 0 else "- Events mapped to signatures: `0 / 0 = N/A`")
    lines.append(f"- Evidence-qualified coverage (`n(signature) >= n_min`): `{qual_events} / {n_events} = {_fmt_num(_safe_rate(qual_events, n_events))}`")
    lines.append(f"- Median events per signature: `{_fmt_num(median_sig, 0) if median_sig is not None else 'N/A'}`")
    lines.append(f"- P90 events per signature: `{_fmt_num(float(p90_sig), 0) if p90_sig is not None else 'N/A'}`")
    lines.append(f"- Gini over signature counts: `{_fmt_num(gini)}`")
    lines.append("")
    lines.append("### B) Override behavior")
    lines.append("")
    lines.append(f"- Candidate decisions total: `{n_events}`")
    lines.append(f"- Baseline-only selections (`selection_source=fitness`): `{t2.get('fitness', {}).get('n', 0) if sel_field_present else 'N/A (legacy schema)'}`")
    lines.append(f"- Phenomenology selections (`selection_source=fitness+phenomenology`): `{phenom_n}`")
    lines.append(f"- Override rate: `{phenom_n} / {n_events} = {_fmt_num(override_rate)}`")
    lines.append("- Confidence-gated rejection count: `N/A`")
    lines.append("")
    lines.append("### C) Outcome deltas")
    lines.append("")
    lines.append(f"- Recovery success rated events: `{len(helped_vals)}`")
    lines.append(f"- Recovery success yes: `{helped_yes}`")
    lines.append(f"- Recovery success rate (overall rated): `{_fmt_num(_safe_rate(helped_yes, len(helped_vals)))}`")
    lines.append(f"- Burden ratings count: `{len(burden_vals)}`")
    lines.append(f"- Burden mean: `{_fmt_num(_safe_avg(burden_vals))}`")
    lines.append("")
    lines.append("Delta block:")
    lines.append("- `Delta_success`: `N/A`")
    lines.append("- `Delta_burden`: `N/A`")
    lines.append("- `Delta_time`: `N/A`")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3) Required tables")
    lines.append("")
    lines.append("### T1) Signature summary")
    lines.append("")
    lines.append("| signature | events_n | qualified (`>=n_min`) | best_action | best_action_rated_n | best_action_helped_rate | best_action_burden_avg |")
    lines.append("|---|---:|---|---|---:|---:|---:|")
    for sig, row in sorted(rows_t1.items(), key=lambda kv: (-kv[1]["events_n"], kv[0])):
        bname, brated, bhr, bba = best_action(row["actions"])
        qual = "yes" if row["events_n"] >= n_min else "no"
        lines.append(f"| {sig} | {row['events_n']} | {qual} | {bname} | {brated} | {_fmt_num(bhr)} | {_fmt_num(bba)} |")
    lines.append("")
    lines.append("### T2) Action comparison by selection source")
    lines.append("")
    lines.append("| selection_source | events_n | helped_yes_n | helped_rated_n | helped_rate | burden_n | burden_avg | time_to_recover_avg |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for sel, row in sorted(t2.items(), key=lambda kv: (-kv[1]["n"], kv[0])):
        lines.append(
            f"| {sel} | {row['n']} | {row['helped_yes']} | {row['helped_n']} | {_fmt_num(_safe_rate(row['helped_yes'], row['helped_n']))} | {len(row['burden'])} | {_fmt_num(_safe_avg(row['burden']))} | N/A |"
        )
    if not t2:
        lines.append("| N/A | 0 | 0 | 0 | N/A | 0 | N/A | N/A |")
    lines.append("")
    lines.append("### T3) Trigger/domain stratification")
    lines.append("")
    lines.append("| domain | trigger | selection_source | events_n | helped_rate | burden_avg |")
    lines.append("|---|---|---|---:|---:|---:|")
    for (dom, trig, sel), row in sorted(t3.items(), key=lambda kv: (-kv[1]["n"], kv[0][0], kv[0][1], kv[0][2])):
        lines.append(
            f"| {dom} | {trig} | {sel} | {row['n']} | {_fmt_num(_safe_rate(row['helped_yes'], row['helped_n']))} | {_fmt_num(_safe_avg(row['burden']))} |"
        )
    if not t3:
        lines.append("| N/A | N/A | N/A | 0 | N/A | N/A |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4) Minimal statistical checks")
    lines.append("")
    lines.append("Not auto-run by this command.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 5) Ablation block")
    lines.append("")
    lines.append("Current result:")
    lines.append("- Baseline policy behavior is observed from available events.")
    lines.append("- Post-upgrade phenomenology-vs-fitness ablation requires new intervention data.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 6) Decision rubric")
    lines.append("")
    cov = _safe_rate(qual_events, n_events) if n_events > 0 else None
    cov_flag = "PASS" if cov is not None and cov >= 0.5 else "FAIL"
    spar_flag = "PASS" if median_sig is not None and float(median_sig) >= 3.0 else "FAIL"
    over_flag = "PASS" if override_rate is not None and 0.05 <= override_rate <= 0.5 else "WATCH"
    lines.append(f"- Coverage adequate: `{cov_flag}`")
    lines.append(f"- Sparsity acceptable: `{spar_flag}`")
    lines.append(f"- Override rate non-trivial but bounded: `{over_flag}`")
    lines.append("- `Delta_success` non-negative: `WATCH`")
    lines.append("- `Delta_burden` non-positive: `WATCH`")
    lines.append("- No feasibility regression: `WATCH`")
    lines.append("")
    lines.append("Final recommendation:")
    lines.append("- `TUNE SIGNATURE GRANULARITY` and collect more post-upgrade intervention data before increasing influence.")

    return "\n".join(lines) + "\n"

# ----------------------------
# Capacity and feasibility
# ----------------------------

def is_date_blackout(st: ProkState, d: dt.date) -> bool:
    ds = d.isoformat()
    if any(b.date == ds for b in st.date_blackouts):
        return True
    for r in st.daterange_blackouts:
        a = dt.date.fromisoformat(r.start)
        b = dt.date.fromisoformat(r.end)
        if a <= d <= b:
            return True
    return False

def _weekly_active(b: WeeklyBlackout, wd: int) -> bool:
    if b.weekday_start <= b.weekday_end:
        return b.weekday_start <= wd <= b.weekday_end
    return wd >= b.weekday_start or wd <= b.weekday_end

def minutes_blocked_by_weekly(st: ProkState, d: dt.date) -> int:
    wd = d.weekday()
    blocked = 0
    for b in st.weekly_blackouts:
        if not _weekly_active(b, wd):
            continue
        s0 = _parse_hhmm(b.start_hhmm)
        s1 = _parse_hhmm(b.end_hhmm)
        if s1 <= s0:
            blocked += max(0, _parse_hhmm(st.day_end) - _parse_hhmm(st.day_start))
        else:
            blocked += (s1 - s0)
    return blocked

def day_capacity_blocks(st: ProkState, d: dt.date) -> int:
    if is_date_blackout(st, d):
        return 0
    w0 = _parse_hhmm(st.day_start)
    w1 = _parse_hhmm(st.day_end)
    if w1 <= w0:
        return 0
    workable = w1 - w0
    blocked = minutes_blocked_by_weekly(st, d)
    free = max(0, workable - blocked)
    blocks = free // max(1, st.plan_block_min)
    return int(min(blocks, max(0, st.max_blocks_per_day)))

def horizon_days_for_deadline(st: ProkState, deadline: Optional[str]) -> List[dt.date]:
    td = _today()
    if not deadline:
        return [td]
    dl = dt.date.fromisoformat(deadline)
    if dl < td:
        return [td]
    return _daterange(td, dl)

def task_remaining_blocks_needed(st: ProkState, task: Task) -> int:
    return int(math.ceil(max(0, task.remaining_minutes) / max(1, st.plan_block_min)))

def task_capacity_blocks(st: ProkState, task: Task) -> int:
    days = horizon_days_for_deadline(st, task.deadline)
    return sum(day_capacity_blocks(st, d) for d in days)

def task_slack_blocks(st: ProkState, task: Task) -> int:
    return task_capacity_blocks(st, task) - task_remaining_blocks_needed(st, task)

def slack_band_from_blocks(s: int) -> str:
    return "neg" if s < 0 else ("zero" if s == 0 else "pos")

# ----------------------------
# Multi-task planner (blocks)
# ----------------------------

def _task_sort_key(st: ProkState, task: Task) -> Tuple[dt.date, int]:
    dl = dt.date.fromisoformat(task.deadline) if task.deadline else _today()
    slack = task_slack_blocks(st, task)
    return (dl, slack)

def replan_multitask(st: ProkState) -> Dict[str, Any]:
    tasks = [t for t in st.tasks if t.remaining_minutes > 0]
    if not tasks:
        return {"_status": 0, "_msg": "No active tasks with remaining minutes."}

    deadlines = [t.deadline for t in tasks if t.deadline]
    if not deadlines:
        return {"_status": 0, "_msg": "No task deadlines set."}

    latest_deadline = max(dt.date.fromisoformat(d) for d in deadlines)
    days = _daterange(_today(), latest_deadline)
    caps = {d.isoformat(): day_capacity_blocks(st, d) for d in days}

    need_blocks = {t.id: task_remaining_blocks_needed(st, t) for t in tasks}
    total_need = sum(need_blocks.values())
    total_cap = sum(caps.values())
    status = 1 if total_cap >= total_need else 2

    task_slack = {t.id: task_slack_blocks(st, t) for t in tasks}

    # Priority: earliest deadline, then lowest slack
    ordered = sorted(tasks, key=lambda t: _task_sort_key(st, t))

    plan: Dict[str, Any] = {
        "_status": status,
        "_peak": int(max(caps.values()) if caps else 0),
        "_need_total": int(total_need),
        "_task_need": {k: int(v) for k, v in need_blocks.items()},
        "_task_slack": {k: int(v) for k, v in task_slack.items()},
    }

    remaining = dict(need_blocks)
    for d in days:
        dkey = d.isoformat()
        cap = caps[dkey]
        if cap <= 0:
            continue
        day_alloc: Dict[str, int] = {}
        for t in ordered:
            if remaining[t.id] <= 0:
                continue
            if t.deadline and d > dt.date.fromisoformat(t.deadline):
                continue
            take = min(cap, remaining[t.id])
            if take > 0:
                day_alloc[t.id] = int(take)
                remaining[t.id] -= take
                cap -= take
            if cap <= 0:
                break
        if day_alloc:
            plan[dkey] = day_alloc

    return plan

def compute_alarm(st: ProkState) -> Tuple[str, List[str]]:
    level = "ok"
    reasons: List[str] = []

    for t in st.tasks:
        if t.deadline:
            slack = task_slack_blocks(st, t)
            if slack < 0:
                level = "critical"
                reasons.append(f"Task {t.id} infeasible (slack {slack})")

    for t in st.tasks:
        if t.recent_drift_count >= 4:
            level = "critical"
            reasons.append(f"Task {t.id} drift cluster (recent {t.recent_drift_count})")
        elif t.recent_drift_count >= 2 and level != "critical":
            level = "warning"
            reasons.append(f"Task {t.id} drift elevated (recent {t.recent_drift_count})")

    for d, ds in (st.domains or {}).items():
        if ds.drift_ema >= 0.7:
            level = "critical"
            reasons.append(f"Domain {d} drift EMA high ({ds.drift_ema:.2f})")
        elif ds.drift_ema >= 0.4 and level != "critical":
            level = "warning"
            reasons.append(f"Domain {d} drift EMA elevated ({ds.drift_ema:.2f})")

    return level, reasons

def alarm_recommendations(level: str) -> List[str]:
    if level != "critical":
        return []
    return [
        "Increase capacity: prok config set --max-blocks-per-day 8",
        "Extend day window: prok config set --day-start 07:00 --day-end 23:00",
        "Adjust scope/deadline: prok task edit <id> --hours 12 | --deadline 2026-05-01",
        "Split scope: prok task add --domain <d> --name \"part 2\" --hours 6 --deadline 2w",
    ]

# ----------------------------
# Domain skin
# ----------------------------

def tiny_start_prompt(domain: str) -> str:
    d = (domain or "general").lower()
    if d in ("skola", "school", "study", "exam"):
        return "Open the exact material. Pick the smallest subtask. Do the first step for 2 minutes (ugly/rough is fine)."
    return "Open the thing. Pick the tiniest next action. Do 2 minutes."

# ----------------------------
# Trigger + Haghbin prompts
# ----------------------------

def ask_trigger(lang: Dict[str,str]) -> str:
    print("Trigger? [c=confusion, b=boredom, d=distraction, a=anxiety, f=fatigue, o=other]")
    m = {"c":"confusion","b":"boredom","d":"distraction","a":"anxiety","f":"fatigue","o":"other"}
    while True:
        ans = input("> ").strip().lower()[:1]
        if ans in m:
            return m[ans]
        print("Please type one of: c b d a f o")

def ask_haghbin(genome: Dict[str, Any]) -> HaghbinFlags:
    strict = str(genome.get("g4_classifier") or "balanced").lower()
    print("Quick Haghbin check (y/n).")

    def yn(prompt: str, default: int) -> int:
        while True:
            s = input(prompt + (" [Y/n] " if default == 1 else " [y/N] ")).strip().lower()
            if not s:
                return default
            if s.startswith("y"):
                return 1
            if s.startswith("n"):
                return 0
            print("Type y or n.")
    if strict == "strict":
        dI, dC, dN = 1, 1, 1
    elif strict == "lenient":
        dI, dC, dN = 1, 0, 0
    else:
        dI, dC, dN = 1, 1, 0

    I = yn("Did you intend to work now?", dI)
    C = yn("Would delaying likely cost you (grades/time/stress)?", dC)
    N = yn("Does the delay feel needless/irrational (you 'should' start)?", dN)
    F = yn("Negative feelings right now (guilt/anxiety/pressure)?", 0)
    return HaghbinFlags(intent=I, expected_cost=C, irrational_delay=N, negative_affect=F)

# ----------------------------
# TMT update + recovery candidates
# ----------------------------

def update_tmt_bins(theta: TMTBins, label: str, trigger: str, recovery_type: Optional[str] = None) -> TMTBins:
    t = TMTBins(E=theta.E, V=theta.V, D=theta.D, G=theta.G).sanitize()

    if trigger in ("confusion", "anxiety"):
        t.E = _bin_down(t.E, TMT_E)
    elif trigger == "boredom":
        t.V = _bin_down(t.V, TMT_V)
    elif trigger == "distraction":
        t.G = "H"
    elif trigger == "fatigue":
        t.E = _bin_down(t.E, TMT_E)
        t.G = "H"

    if label == "PROCRASTINATION":
        t.D = "F"

    if recovery_type == "E":
        t.E = _bin_up(t.E, TMT_E); t.G = "L"; t.D = "N"
    elif recovery_type == "V":
        t.V = _bin_up(t.V, TMT_V); t.G = "L"; t.D = "N"
    elif recovery_type == "G":
        t.G = "L"; t.D = "N"

    return t.sanitize()

def _goal_early_has(goal: Dict[str,Any], item: str, tiers: int = 2) -> bool:
    tt = goal.get("tiers") or []
    for tier in tt[:tiers]:
        if item in [str(x) for x in tier]:
            return True
    return False

def build_recovery_candidates(
    st: ProkState,
    task: Task,
    label: str,
    trigger: str,
    check_gaps: Tuple[int, int],
) -> List[Dict[str, Any]]:
    g = st.genome
    theta0 = task.tmt_bins
    goal = st.goal_profile or default_goal_profile()

    def ctx_for(theta1: TMTBins, label1: str, gaps: Tuple[int,int]) -> Dict[str, Any]:
        return {
            "slack": slack_band_from_blocks(task_slack_blocks(st, task)),
            "E": theta1.E, "V": theta1.V, "D": theta1.D, "G": theta1.G,
            "trigger": trigger,
            "procrastinating": (label1 == "PROCRASTINATION"),
            "intent_gap": (label1 == "PROCRASTINATION"),
            "expected_cost": True,
            "reward_near": (theta1.D == "N"),
            "recover_fast": True,  # will be learned later from logs
            "checks": "many" if gaps[0] <= 3 else ("medium" if gaps[0] <= 6 else "few"),
            "annoying": (gaps[0] <= 2) and _goal_early_has(goal, "low_annoyance", tiers=3),
        }

    candidates: List[Dict[str, Any]] = []

    # Candidate: E-focused recovery (clarity)
    thetaE = update_tmt_bins(theta0, label, trigger, recovery_type="E")
    gapsE = genome_check_gaps(g, thetaE)
    candidates.append({"name":"E_RECOVERY","theta_next":thetaE,"gaps_next":gapsE,
                       "steps":[
                           "Clarify the very next step (make success certain).",
                           "Shrink to a 2-minute micro-action.",
                           "Do 5 minutes now (ugly/rough is fine).",
                       ],
                       "ctx": ctx_for(thetaE, "DRIFT", gapsE)})

    # Candidate: V-focused recovery (interest)
    thetaV = update_tmt_bins(theta0, label, trigger, recovery_type="V")
    gapsV = genome_check_gaps(g, thetaV)
    candidates.append({"name":"V_RECOVERY","theta_next":thetaV,"gaps_next":gapsV,
                       "steps":[
                           "Pick the most interesting subtask (practice > reread).",
                           "Define a tiny visible win (10 minutes) and finish it.",
                           "Give a small immediate reward after (tea/stretch).",
                       ],
                       "ctx": ctx_for(thetaV, "DRIFT", gapsV)})

    # Candidate: Gamma-focused recovery (distraction control)
    thetaG = update_tmt_bins(theta0, label, trigger, recovery_type="G")
    gapsG = genome_check_gaps(g, thetaG)
    candidates.append({"name":"G_RECOVERY","theta_next":thetaG,"gaps_next":gapsG,
                       "steps":[
                           "Remove the biggest distraction (phone away / close tabs).",
                           "Shrink the next step to a 2-minute action.",
                           "Restart with 5 minutes now.",
                       ],
                       "ctx": ctx_for(thetaG, "DRIFT", gapsG)})

    # Candidate: mixed (trigger-weighted)
    thetaM = update_tmt_bins(theta0, label, trigger, recovery_type=None)
    if g.get("g6_enforce_near_reward") and thetaM.D == "F":
        thetaM.D = "N"
    gapsM = genome_check_gaps(g, thetaM)
    mixed_steps = ["Remove the biggest distraction (if any)."]
    if trigger in ("confusion","anxiety"):
        mixed_steps += ["Clarify: define the next 2-minute step so you cannot fail."]
    elif trigger == "boredom":
        mixed_steps += ["Quick win: pick a short, interesting subtask with a visible finish."]
    mixed_steps += ["Do 5 minutes now."]
    candidates.append({"name":"MIXED_RECOVERY","theta_next":thetaM.sanitize(),"gaps_next":gapsM,
                       "steps":mixed_steps,
                       "ctx": ctx_for(thetaM, "DRIFT", gapsM)})

    return candidates

def load_fitness_program(st: ProkState) -> Optional[ProkFitProgram]:
    if not st.fitness_path:
        return None
    p = Path(st.fitness_path)
    if not p.exists():
        return None
    try:
        return parse_prokfit(p)
    except Exception as e:
        print(f"Warning: failed to parse fitness DSL ({p}): {e}")
        return None

# ----------------------------
# Session runner (checkclock + immediate report)
# ----------------------------

def _poll_line_nonblocking() -> Optional[str]:
    try:
        r, _, _ = select.select([sys.stdin], [], [], 0)
    except Exception:
        return None
    if not r:
        return None
    line = sys.stdin.readline()
    if not line:
        return None
    return line.strip()

def _ask_int_in_range(prompt: str, lo: int, hi: int, default: Optional[int] = None) -> Optional[int]:
    while True:
        s = input(prompt).strip()
        if not s and default is not None:
            return default
        if not s:
            return None
        try:
            v = int(s)
            if lo <= v <= hi:
                return v
        except Exception:
            pass
        print(f"Type a number {lo}-{hi} (or press Enter to skip).")

def run_session(
    st: ProkState,
    task: Task,
    minutes: int,
    checks_mode: str,
    min_gap: int,
    max_gap: int,
    ask_trigger_on: str,
    ask_haghbin_on_p: bool,
    fitness_program: Optional[ProkFitProgram],
) -> None:
    ensure_today(st)
    _init_defaults(st)

    lang = st.language_pack or dict(DEFAULT_LANG)

    task.tmt_bins.sanitize()
    block_minutes = minutes if minutes > 0 else domain_block_minutes(st, task.domain, task.tmt_bins)

    random_checks = (checks_mode in ("random","auto"))
    if checks_mode == "auto":
        min_gap, max_gap = domain_check_gaps(st, task.domain, task.tmt_bins)
    if max_gap < min_gap:
        max_gap = min_gap

    print(f"SETUP ({task.domain}): {tiny_start_prompt(task.domain)}")
    input("Press Enter when you’ve done the 2-minute tiny start... ")

    paused = False
    last_tick = time.time()
    engaged_seconds = 0.0
    seconds_left = float(block_minutes) * 60.0

    next_check = random.randint(min_gap*60, max_gap*60) if random_checks else None

    print(lang_get(lang, "state.engaged", "ENGAGED").upper() + ": focus timer started.")
    print(lang_get(lang, "prompt.you_can_report", "You can report anytime..."))

    if random_checks:
        print(f"Random checks: ON | gaps {min_gap}–{max_gap} min")
    else:
        print("Random checks: OFF")

    session_id = f"{_now_iso()}_{random.randint(1000,9999)}"
    check_count = 0

    _log_event("session_start", {
        "participant_id": st.participant_id,
        "session_id": session_id,
        "study_mode": st.study_mode,
        "task_id": task.id,
        "task_name": task.name,
        "domain": task.domain,
        "planned_minutes": block_minutes,
        "checks_mode": checks_mode,
        "gap": [min_gap, max_gap] if random_checks else None,
        "theta": asdict(task.tmt_bins),
        "slack": task_slack_blocks(st, task),
        "goal_profile": st.goal_profile,
        "genome": st.genome,
        "use_cca": st.use_cca,
    })

    drift_events = 0
    quit_early = False

    def do_pause(reason: str) -> None:
        nonlocal paused
        paused = True
        print(f"\nPAUSED ({reason}). Type r + Enter to resume.")
        _log_event("pause", {
            "participant_id": st.participant_id,
            "session_id": session_id,
            "task_id": task.id,
            "task_name": task.name,
            "domain": task.domain,
            "reason": reason,
        })

    def do_resume() -> None:
        nonlocal paused, last_tick
        if paused:
            paused = False
            last_tick = time.time()
            print("RESUMED. Back to ENGAGED.")
            _log_event("resume", {
                "participant_id": st.participant_id,
                "session_id": session_id,
                "task_id": task.id,
                "task_name": task.name,
                "domain": task.domain,
            })

    def ask_state_check() -> str:
        print(f"\n{lang_get(lang, 'prompt.check', 'CHECK')}: [e=engaged, d=drifting, p=procrastinating, b=blocked]")
        return input("> ").strip().lower()[:1]

    def study_ratings(event_kind: str) -> Dict[str, Any]:
        if not st.study_mode:
            return {}
        burden = _ask_int_in_range("(Study) Burden right now? 1(low)–5(high) [Enter skip] > ", 1, 5, default=None)
        helped = input("(Study) Did this help you restart? [y/n/Enter skip] > ").strip().lower()
        helped_val = None
        if helped.startswith("y"):
            helped_val = True
        elif helped.startswith("n"):
            helped_val = False
        return {"burden": burden, "helped": helped_val, "event_kind": event_kind}

    def course_correct(kind: str, from_check: bool) -> None:
        nonlocal drift_events, min_gap, max_gap, next_check, paused, check_count
        drift_events += 1
        task.recent_drift_count = _clamp(task.recent_drift_count + 1, 0, 5)
        ds = get_domain_state(st, task.domain)
        update_domain_drift(ds, drifted=True)

        trigger = "other"
        label = "DRIFT" if kind == "d" else ("PROCRASTINATION" if kind == "p" else "BLOCKED")

        if ask_trigger_on in ("always","dp") and kind in ("d","p"):
            trigger = ask_trigger(lang)
        elif ask_trigger_on == "p" and kind == "p":
            trigger = ask_trigger(lang)

        flags = None
        if kind == "p" and ask_haghbin_on_p:
            flags = ask_haghbin(st.genome)
            label = haghbin_label(flags)
        else:
            label = "PROCRASTINATION" if kind == "p" else ("DRIFT" if kind == "d" else "BLOCKED")

        task.tmt_bins = update_tmt_bins(task.tmt_bins, label, trigger, recovery_type=None)
        signature = phenomenology_signature(task, label, trigger)

        base_prog = fitness_program or load_fitness_program(st) or built_in_fitness()
        prog = fitness_from_goal_profile(base_prog, st.goal_profile or default_goal_profile())

        candidates = build_recovery_candidates(st, task=task, label=label, trigger=trigger, check_gaps=(min_gap, max_gap))
        ranked = rank_candidates_scored(prog, candidates)
        best = ranked[0][1]
        selection_source = "fitness"

        shortlist = [c for _, c in ranked[:3]]
        rec = phenomenology_recommendation(st, signature, [c["name"] for c in shortlist])
        if rec:
            rec_name, rec_score, rec_n = rec
            influence = _to_float((st.phenomenology or {}).get("influence"), 0.35)
            if rec_name != best["name"] and rec_score >= influence:
                alt = next((c for c in shortlist if c["name"] == rec_name), None)
                if alt is not None:
                    best = alt
                    selection_source = "fitness+phenomenology"
                    print(f"Phenomenology match: using {rec_name} (score={rec_score:.2f}, n={rec_n}).")

        task.tmt_bins = best["theta_next"]
        if random_checks:
            min_gap, max_gap = best["gaps_next"]
            min_gap, max_gap = genome_tighten(st.genome, min_gap, max_gap)
            next_check = random.randint(min_gap*60, max_gap*60)

        label_txt = lang_get(lang, "state.drift", "DRIFTING").upper() if kind == "d" else (
            lang_get(lang, "state.procrastinate", "PROCRASTINATING").upper() if kind == "p" else "BLOCKED"
        )
        print(f"\n{label_txt}: course-correct now.")
        print(f"Chosen recovery: {best['name']}  | theta -> {asdict(task.tmt_bins)}")
        for i, line in enumerate(best["steps"], 1):
            print(f"  {i}) {line}")

        if st.genome.get("g6_enforce_near_reward") and st.genome.get("g7_artifact") != "none":
            print(f"Artifact (near-reward): {st.genome.get('g7_artifact')} (do it as part of the next ~10 minutes).")

        input("Press Enter after you've restarted (≈5 min)… ")

        # Study ratings (after action)
        ratings = study_ratings(event_kind=kind)
        _log_event("course_correct", {
            "participant_id": st.participant_id,
            "session_id": session_id,
            "task_id": task.id,
            "task_name": task.name,
            "domain": task.domain,
            "kind": kind,
            "from_check": from_check,
            "trigger": trigger,
            "label": label,
            "haghbin": asdict(flags) if flags else None,
            "theta": asdict(task.tmt_bins),
            "chosen": best["name"],
            "selection_source": selection_source,
            "phenomenology_signature": signature,
            "gaps": [min_gap, max_gap] if random_checks else None,
            **ratings,
        })
        phenomenology_record_outcome(st, signature, best["name"], ratings)

        ds.recovery_counts[best["name"]] = _to_int(ds.recovery_counts.get(best["name"]), 0) + 1

        # Replan if threshold crossed
        thresh = _to_int(st.genome.get("g8_replan_threshold"), 1)
        if task.deadline and drift_events >= thresh:
            plan = replan_multitask(st)
            if plan.get("_status") == 1:
                print(f"REPLAN: feasible. Peak cap {plan.get('_peak')} block(s)/day.")
            else:
                print("REPLAN: WARNING infeasible under constraints.")
            _log_event("replan", {
                "participant_id": st.participant_id,
                "session_id": session_id,
                "status": plan.get("_status"),
                "peak": plan.get("_peak"),
                "need_blocks": plan.get("_need_total"),
                "slack": task_slack_blocks(st, task),
            })

        paused = False
        print("Back to ENGAGED.\n")

    while seconds_left > 0:
        line = _poll_line_nonblocking()
        if line:
            c0 = line.strip().lower()[:1]
            if c0 in ("d","p"):
                course_correct(c0, from_check=False)
            elif c0 == "b":
                do_pause("blocked")
            elif c0 == "r":
                if st.genome.get("g9_pause_policy") == "hard":
                    print("Hard pause policy: do a 2-minute micro-action before resuming.")
                    input("Press Enter after 2 minutes of micro-action… ")
                do_resume()
            elif c0 == "q":
                print("\nQUIT: ending session early.")
                quit_early = True
                _log_event("quit", {
                    "participant_id": st.participant_id,
                    "session_id": session_id,
                    "task_id": task.id,
                    "task_name": task.name,
                    "domain": task.domain,
                })
                break
            elif c0 == "c":
                resp = ask_state_check()
                _log_event("check", {
                    "participant_id": st.participant_id,
                    "session_id": session_id,
                    "task_id": task.id,
                    "task_name": task.name,
                    "domain": task.domain,
                    "resp": resp or "?",
                })
                check_count += 1
                if resp in ("d","p"):
                    course_correct(resp, from_check=True)
                elif resp == "b":
                    do_pause("blocked")
                else:
                    print("OK — continue.")

        now = time.time()
        dt_sec = min(now - last_tick, 1.5)
        last_tick = now

        if not paused:
            seconds_left -= dt_sec
            engaged_seconds += dt_sec

            if random_checks and next_check is not None:
                next_check -= dt_sec
                if next_check <= 0:
                    resp = ask_state_check()
                    _log_event("check", {
                        "participant_id": st.participant_id,
                        "session_id": session_id,
                        "task_id": task.id,
                        "task_name": task.name,
                        "domain": task.domain,
                        "resp": resp or "?",
                    })
                    check_count += 1
                    if resp in ("d","p"):
                        course_correct(resp, from_check=True)
                    elif resp == "b":
                        do_pause("blocked")
                    next_check = random.randint(min_gap*60, max_gap*60)

        time.sleep(0.35)

    minutes_worked = max(0, int(round(engaged_seconds / 60.0)))
    print("\nDONE: session ended.")
    print(f"Estimated focused minutes (excluding pauses): {minutes_worked}")

    end_payload = {
        "participant_id": st.participant_id,
        "session_id": session_id,
        "task_id": task.id,
        "task_name": task.name,
        "domain": task.domain,
        "quit_early": quit_early,
        "minutes_worked_est": minutes_worked,
        "theta_end": asdict(task.tmt_bins),
        "check_count": check_count,
    }

    if st.study_mode:
        sat = _ask_int_in_range("(Study) Overall session success? 1–5 [Enter skip] > ", 1, 5, default=None)
        end_payload["session_success"] = sat

    _log_event("session_end", end_payload)

    # Domain drift EMA decay if no drift occurred
    ds = get_domain_state(st, task.domain)
    if drift_events == 0:
        update_domain_drift(ds, drifted=False)

    # Minimum / streak bookkeeping
    if minutes_worked > 0 and not st.today_min_done:
        st.today_min_done = True
        td = _today().isoformat()
        if st.last_streak_date:
            prev = dt.date.fromisoformat(st.last_streak_date)
            st.streak = (st.streak + 1) if (prev + dt.timedelta(days=1) == _today()) else 1
        else:
            st.streak = 1
        st.last_streak_date = td

    # Log minutes to plan
    if task.deadline and task.remaining_minutes > 0:
        print("Log minutes worked to reduce remaining plan? [Y/n]")
        if not input("> ").strip().lower().startswith("n"):
            raw = input(f"How many minutes to log? [default {minutes_worked}] > ").strip()
            mm = minutes_worked if not raw else max(0, _to_int(raw, minutes_worked))
            task.remaining_minutes = max(0, task.remaining_minutes - mm)
            _log_event("log_minutes", {
                "participant_id": st.participant_id,
                "session_id": session_id,
                "task_id": task.id,
                "task_name": task.name,
                "domain": task.domain,
                "minutes": mm,
                "remaining_minutes": task.remaining_minutes,
            })
            print(f"Logged {mm} min. Remaining minutes: {task.remaining_minutes}")

    save_state(st)

# ----------------------------
# CLI commands
# ----------------------------

def cmd_task_add(args) -> None:
    st = load_state(); ensure_today(st); _init_defaults(st)
    if args.block_min is not None:
        st.plan_block_min = args.block_min
    if args.tiny_min is not None:
        st.tiny_min = args.tiny_min
    if args.day_start:
        st.day_start = args.day_start
    if args.day_end:
        st.day_end = args.day_end
    if args.max_blocks_per_day is not None:
        st.max_blocks_per_day = args.max_blocks_per_day

    deadline = _parse_deadline(args.deadline) if args.deadline else None
    total_minutes = int(round(float(args.total_hours) * 60.0)) if args.total_hours is not None else 0
    task = Task(
        id=_make_task_id(args.name),
        domain=args.domain,
        name=args.name,
        deadline=deadline,
        total_minutes=total_minutes,
        remaining_minutes=total_minutes,
        tmt_bins=TMTBins(),
        recent_drift_count=0,
    )
    st.tasks.append(task)
    st.active_task_id = task.id
    save_state(st)
    _log_event("task_add", {
        "participant_id": st.participant_id,
        "task_id": task.id,
        "domain": task.domain,
        "name": task.name,
        "deadline": task.deadline,
        "total_minutes": task.total_minutes,
    })
    print(f"Added task: {task.id} | {task.domain} | {task.name}")
    if task.deadline:
        print(f"Deadline: {task.deadline} | Total minutes: {task.total_minutes}")

def cmd_task_list(args) -> None:
    st = load_state(); ensure_today(st); _init_defaults(st)
    if not st.tasks:
        print("No tasks. Add one: prok task add --domain ... --name ... --hours ... --deadline ...")
        return
    print("TASKS:")
    for t in st.tasks:
        active = "*" if st.active_task_id == t.id else " "
        slack = task_slack_blocks(st, t) if t.deadline else None
        slack_txt = str(slack) if slack is not None else "-"
        print(f"{active} {t.id} | {t.domain} | {t.name} | deadline={t.deadline or '-'} | remaining={t.remaining_minutes}/{t.total_minutes} | slack={slack_txt}")

def cmd_task_set_active(args) -> None:
    st = load_state(); ensure_today(st); _init_defaults(st)
    t = get_task_by_id(st, args.task_id)
    if not t:
        raise SystemExit(f"Task not found: {args.task_id}")
    st.active_task_id = t.id
    save_state(st)
    print(f"Active task set to {t.id}: {t.name}")

def cmd_task_remove(args) -> None:
    st = load_state(); ensure_today(st); _init_defaults(st)
    t = get_task_by_id(st, args.task_id)
    if not t:
        raise SystemExit(f"Task not found: {args.task_id}")
    st.tasks = [x for x in st.tasks if x.id != t.id]
    if st.active_task_id == t.id:
        st.active_task_id = st.tasks[0].id if st.tasks else None
    save_state(st)
    _log_event("task_remove", {"participant_id": st.participant_id, "task_id": t.id})
    print(f"Removed task {t.id}: {t.name}")

def cmd_task_edit(args) -> None:
    st = load_state(); ensure_today(st); _init_defaults(st)
    t = get_task_by_id(st, args.task_id)
    if not t:
        raise SystemExit(f"Task not found: {args.task_id}")
    if args.domain:
        t.domain = args.domain
    if args.name:
        t.name = args.name
    if args.deadline:
        t.deadline = _parse_deadline(args.deadline)
    if args.total_hours is not None:
        t.total_minutes = int(round(float(args.total_hours) * 60.0))
        if t.remaining_minutes > t.total_minutes:
            t.remaining_minutes = t.total_minutes
    if args.remaining_minutes is not None:
        t.remaining_minutes = max(0, _to_int(args.remaining_minutes, t.remaining_minutes))
    save_state(st)
    print(f"Updated task {t.id}: {t.name}")

def cmd_plan(args) -> None:
    # Back-compat alias: plan -> task add
    cmd_task_add(args)
    print("Note: 'plan' is now 'task add' in v5.")

def _weekday_idx(s: str) -> int:
    m = {"mon":0,"monday":0,"tue":1,"tues":1,"tuesday":1,"wed":2,"wednesday":2,
         "thu":3,"thur":3,"thurs":3,"thursday":3,"fri":4,"friday":4,"sat":5,"saturday":5,"sun":6,"sunday":6}
    k = s.strip().lower()
    if k not in m:
        raise ValueError("Invalid weekday. Use Mon..Sun")
    return m[k]

def _parse_weekday_range(s: str) -> Tuple[int,int]:
    s = s.strip()
    if "-" not in s:
        i = _weekday_idx(s); return i,i
    a,b = s.split("-",1)
    return _weekday_idx(a), _weekday_idx(b)

def cmd_blackout(args) -> None:
    st = load_state(); ensure_today(st); _init_defaults(st)
    if args.action == "list":
        print("Weekly blackouts:")
        for b in st.weekly_blackouts:
            print(f"  {b.weekday_start}-{b.weekday_end} {b.start_hhmm}-{b.end_hhmm} {b.label}".strip())
        print("Date blackouts:")
        for b in st.date_blackouts:
            print(f"  {b.date} {b.label}".strip())
        print("Date-range blackouts:")
        for b in st.daterange_blackouts:
            print(f"  {b.start}..{b.end} {b.label}".strip())
        return
    if args.action == "clear":
        st.weekly_blackouts = []; st.date_blackouts = []; st.daterange_blackouts = []
        save_state(st); _log_event("blackout_clear", {"participant_id": st.participant_id})
        print("Cleared all blackouts.")
        return
    if args.action == "add":
        label = args.label or ""
        if args.weekday:
            ws,we = _parse_weekday_range(args.weekday)
            _parse_hhmm(args.start); _parse_hhmm(args.end)
            st.weekly_blackouts.append(WeeklyBlackout(ws,we,args.start,args.end,label))
            save_state(st); _log_event("blackout_add", {"participant_id": st.participant_id, "type":"weekly","weekday":args.weekday})
            print("Added weekly blackout.")
            return
        if args.date:
            dt.date.fromisoformat(args.date)
            st.date_blackouts.append(DateBlackout(args.date,label))
            save_state(st); _log_event("blackout_add", {"participant_id": st.participant_id, "type":"date","date":args.date})
            print("Added date blackout.")
            return
        if args.daterange:
            m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})\.\.(\d{4}-\d{2}-\d{2})", args.daterange.strip())
            if not m:
                raise SystemExit("daterange must be YYYY-MM-DD..YYYY-MM-DD")
            a,b = m.group(1), m.group(2)
            dt.date.fromisoformat(a); dt.date.fromisoformat(b)
            st.daterange_blackouts.append(DateRangeBlackout(a,b,label))
            save_state(st); _log_event("blackout_add", {"participant_id": st.participant_id, "type":"range","start":a,"end":b})
            print("Added date-range blackout.")
            return
        raise SystemExit("blackout add requires --weekday or --date or --daterange")

def cmd_today(args) -> None:
    st = load_state(); ensure_today(st); _init_defaults(st)
    if not st.tasks:
        print("No tasks set. Use: prok task add --domain ... --name ... --hours ... --deadline ...")
        return
    td = _today()
    cap = day_capacity_blocks(st, td)
    plan = replan_multitask(st)
    today_key = td.isoformat()
    today_plan = plan.get(today_key, {}) if plan.get("_status") in (1, 2) else {}
    total_planned = sum(int(v) for v in today_plan.values()) if isinstance(today_plan, dict) else 0

    level, reasons = compute_alarm(st)

    print(f"DATE: {td.isoformat()}")
    print(f"Capacity today: {cap} block(s) | Planned today: {total_planned} block(s)")
    print(f"Minimum today: {st.tiny_min} min | done: {'YES' if st.today_min_done else 'NO'} | streak: {st.streak}")
    print(f"Alarm: {level.upper()}")
    if reasons:
        for r in reasons[:3]:
            print(f"  - {r}")
    if level == "critical":
        print("STRUCTURAL FIX NEEDED: adjust constraints/capacity/scope/deadline.")
        for rec in alarm_recommendations(level):
            print(f"  - {rec}")

    if today_plan:
        print("Per-task recommendation (blocks):")
        for tid, blocks in today_plan.items():
            t = get_task_by_id(st, tid)
            name = t.name if t else tid
            print(f"  {tid} | {name} | {blocks}")

    # Suggested run
    suggested = None
    if st.active_task_id and st.active_task_id in today_plan:
        suggested = get_task_by_id(st, st.active_task_id)
    elif today_plan:
        top_id = max(today_plan.items(), key=lambda kv: kv[1])[0]
        suggested = get_task_by_id(st, top_id)
    if suggested:
        sugg_minutes = domain_block_minutes(st, suggested.domain, suggested.tmt_bins)
        gmn, gmx = domain_check_gaps(st, suggested.domain, suggested.tmt_bins)
        print(f"Suggested run: --task {suggested.id} --minutes {sugg_minutes} --checks auto (gaps {gmn}–{gmx})")

def cmd_run(args) -> None:
    st = load_state(); ensure_today(st); _init_defaults(st)
    if args.fitness:
        st.fitness_path = args.fitness
    task = select_active_task(st, args.task, args.domain, args.name)
    save_state(st)

    minutes = args.minutes or domain_block_minutes(st, task.domain, task.tmt_bins)
    checks_mode = args.checks
    if checks_mode == "random" and (args.min_gap is None or args.max_gap is None):
        mn, mx = domain_check_gaps(st, task.domain, task.tmt_bins)
    else:
        mn = _to_int(args.min_gap, 6) if args.min_gap is not None else 6
        mx = _to_int(args.max_gap, 14) if args.max_gap is not None else 14

    program = load_fitness_program(st) if st.fitness_path else None

    run_session(
        st=st,
        task=task,
        minutes=minutes,
        checks_mode=checks_mode,
        min_gap=mn,
        max_gap=mx,
        ask_trigger_on=args.ask_trigger,
        ask_haghbin_on_p=not args.no_haghbin,
        fitness_program=program,
    )

def cmd_log(args) -> None:
    st = load_state(); ensure_today(st); _init_defaults(st)
    task = select_active_task(st, args.task, args.domain, args.name)
    mm = max(0, _to_int(args.minutes, 0))
    if mm <= 0:
        raise SystemExit("Minutes must be > 0")
    task.remaining_minutes = max(0, task.remaining_minutes - mm)
    _log_event("log_minutes", {
        "participant_id": st.participant_id,
        "task_id": task.id,
        "task_name": task.name,
        "domain": task.domain,
        "minutes": mm,
        "remaining_minutes": task.remaining_minutes,
    })
    save_state(st)
    print(f"Logged {mm} min to {task.id}. Remaining minutes: {task.remaining_minutes}")

def cmd_status(args) -> None:
    st = load_state(); ensure_today(st); _init_defaults(st)
    print("STATE:")
    print(f"  participant_id: {st.participant_id or '(unset)'} | study_mode: {st.study_mode}")
    print(f"  plan_block_min: {st.plan_block_min} | tiny_min: {st.tiny_min}")
    print(f"  window: {st.day_start}-{st.day_end} | max_blocks/day: {st.max_blocks_per_day}")
    print(f"  today_min_done: {st.today_min_done} | streak: {st.streak}")
    level, reasons = compute_alarm(st)
    print(f"  alarm: {level}")
    if reasons:
        for r in reasons[:3]:
            print(f"    - {r}")
    if level == "critical":
        print("  structural fix needed: adjust constraints/capacity/scope/deadline")
        for rec in alarm_recommendations(level):
            print(f"    - {rec}")
    print("  tasks:")
    for t in st.tasks:
        slack = task_slack_blocks(st, t) if t.deadline else None
        slack_txt = str(slack) if slack is not None else "-"
        active = "*" if st.active_task_id == t.id else " "
        print(f"   {active} {t.id} | {t.domain} | {t.name} | deadline={t.deadline or '-'} | remaining={t.remaining_minutes}/{t.total_minutes} | slack={slack_txt} | drift={t.recent_drift_count} | θ={asdict(t.tmt_bins)}")
    print(f"  genome.explore: {st.genome.get('g10_explore')}")
    print(f"  goal_profile: {st.goal_profile}")
    ph = st.phenomenology or {}
    print(f"  phenomenology: enabled={bool(ph.get('enabled', True))} | signatures={len((ph.get('signatures') or {}))} | min_samples={_to_int(ph.get('min_samples'), 3)} | influence={_to_float(ph.get('influence'), 0.35):.2f}")
    print(f"  use_cca: {st.use_cca} (v4 placeholder, not used)")
    print(f"  state file: {STATE_FILE.resolve()}")
    print(f"  log file: {LOG_FILE.resolve()}")

def cmd_reset(args) -> None:
    if STATE_FILE.exists():
        STATE_FILE.unlink()
    print("Reset: deleted .prok_state.json (logs kept).")

def cmd_config(args) -> None:
    st = load_state(); ensure_today(st); _init_defaults(st)
    if args.action == "show":
        print(_json_dump({
            "plan_block_min": st.plan_block_min,
            "tiny_min": st.tiny_min,
            "day_start": st.day_start,
            "day_end": st.day_end,
            "max_blocks_per_day": st.max_blocks_per_day,
        }))
        return
    if args.action == "set":
        if args.block_min is not None:
            st.plan_block_min = args.block_min
        if args.tiny_min is not None:
            st.tiny_min = args.tiny_min
        if args.day_start:
            st.day_start = args.day_start
        if args.day_end:
            st.day_end = args.day_end
        if args.max_blocks_per_day is not None:
            st.max_blocks_per_day = args.max_blocks_per_day
        save_state(st)
        print("Config updated.")
def cmd_genome(args) -> None:
    st = load_state(); ensure_today(st); _init_defaults(st)
    if args.action == "show":
        print(_json_dump(st.genome)); return
    if args.action == "write-default":
        out = Path(args.out or "genome.default.json")
        out.write_text(_json_dump(default_genome()), encoding="utf-8")
        print(f"Wrote default genome to {out}"); return
    if args.action == "load":
        p = Path(args.path)
        st.genome = json.loads(p.read_text(encoding="utf-8"))
        _init_defaults(st); save_state(st)
        print(f"Loaded genome from {p}"); return

def cmd_fitness(args) -> None:
    st = load_state(); ensure_today(st); _init_defaults(st)
    if args.action == "show":
        print(st.fitness_path or "(none)"); return
    if args.action == "write-default":
        out = Path(args.out or "fitness.default.prokfit")
        out.write_text(
            "require slack != neg\n\n"
            "priority 1:\n"
            "  avoid procrastinating\n"
            "  avoid (intent_gap and expected_cost)\n\n"
            "priority 2:\n"
            "  prefer recover_fast\n"
            "  prefer reward_near\n"
            "  avoid annoying\n",
            encoding="utf-8",
        )
        print(f"Wrote default fitness file to {out}"); return
    if args.action == "set":
        p = Path(args.path)
        _ = parse_prokfit(p)
        st.fitness_path = str(p)
        save_state(st)
        print(f"Set fitness DSL: {p}"); return

def cmd_study(args) -> None:
    st = load_state(); ensure_today(st); _init_defaults(st)
    if args.action == "on":
        st.study_mode = True
        save_state(st)
        print("Study mode: ON")
        return
    if args.action == "off":
        st.study_mode = False
        save_state(st)
        print("Study mode: OFF")
        return
    if args.action == "id":
        st.participant_id = args.value or ""
        save_state(st)
        print(f"Participant ID set to: {st.participant_id or '(unset)'}")
        return
    if args.action == "show":
        print(_json_dump({"participant_id": st.participant_id, "study_mode": st.study_mode}))
        return

def cmd_goal(args) -> None:
    st = load_state(); ensure_today(st); _init_defaults(st)
    if args.action == "show":
        print(_json_dump(st.goal_profile)); return
    if args.action == "clear":
        st.goal_profile = default_goal_profile()
        save_state(st)
        print("Goal profile reset to default."); return
    if args.action == "wizard":
        print("Pick your priorities (higher = earlier). Choose from:")
        for i, g in enumerate(DEFAULT_GOALS, 1):
            print(f"  {i}) {g}")
        picks = []
        raw = input("Enter 3 numbers (e.g., 1 2 3) or press Enter for default > ").strip()
        if raw:
            nums = [n for n in raw.split() if n.isdigit()]
            for n in nums[:3]:
                idx = int(n)-1
                if 0 <= idx < len(DEFAULT_GOALS):
                    picks.append(DEFAULT_GOALS[idx])
        if not picks:
            picks = ["feasible","recover_fast","low_annoyance"]
        tiers = [[picks[0]],[picks[1]],[picks[2]],["progress"]]
        tol = _ask_int_in_range("Max checks per session before it feels annoying? (e.g., 3,5,10) [Enter skip] > ", 0, 999, default=None)
        st.goal_profile = {"tiers": tiers, "annoyance_max_checks": 999 if tol is None else tol}
        save_state(st)
        print("Saved goal profile."); return
    if args.action == "set":
        p = Path(args.path)
        st.goal_profile = json.loads(p.read_text(encoding="utf-8"))
        save_state(st)
        print(f"Loaded goal profile from {p}"); return

def cmd_lang(args) -> None:
    st = load_state(); ensure_today(st); _init_defaults(st)
    if args.action == "show":
        print(_json_dump(st.language_pack)); return
    if args.action == "clear":
        st.language_pack = dict(DEFAULT_LANG)
        save_state(st)
        print("Language pack reset to default."); return
    if args.action == "set":
        p = Path(args.path)
        st.language_pack = json.loads(p.read_text(encoding="utf-8"))
        save_state(st)
        print(f"Loaded language pack from {p}"); return
    if args.action == "wizard":
        lp = dict(st.language_pack or DEFAULT_LANG)
        print("Quick language wizard (press Enter to keep default):")
        drift = input(f"Word for 'drifting' [{lp['state.drift']}] > ").strip()
        if drift: lp["state.drift"] = drift
        blocked = input(f"Word for 'blocked' [{lp['state.blocked']}] > ").strip()
        if blocked: lp["state.blocked"] = blocked
        lp["prompt.you_can_report"] = input(f"Report prompt line [{lp['prompt.you_can_report']}] > ").strip() or lp["prompt.you_can_report"]
        st.language_pack = lp
        save_state(st)
        print("Saved language pack."); return

def cmd_phenom(args) -> None:
    st = load_state(); ensure_today(st); _init_defaults(st)
    ph = st.phenomenology or default_phenomenology()
    sigs = ph.get("signatures") or {}

    if args.action == "show":
        print(_json_dump({
            "enabled": bool(ph.get("enabled", True)),
            "min_samples": _to_int(ph.get("min_samples"), 3),
            "influence": _to_float(ph.get("influence"), 0.35),
            "signatures": len(sigs),
        }))
        if not sigs:
            print("No phenomenology signatures learned yet.")
            return
        print("Top signatures:")
        top = sorted(sigs.items(), key=lambda kv: _to_int((kv[1] or {}).get("n"), 0), reverse=True)[:10]
        for sig, entry in top:
            actions = (entry or {}).get("actions") or {}
            best_name = "-"
            best_score = -1e9
            best_n = 0
            for name, stats in actions.items():
                score, rated = _phenomenology_action_score(stats or {})
                if rated > 0 and score > best_score:
                    best_name = name
                    best_score = score
                    best_n = rated
            if best_name == "-":
                print(f"  {sig} | n={_to_int(entry.get('n'), 0)} | best=-")
            else:
                print(f"  {sig} | n={_to_int(entry.get('n'), 0)} | best={best_name} score={best_score:.2f} rated={best_n}")
        return

    if args.action == "audit":
        log_path = Path(args.log or str(LOG_FILE))
        records = _load_log_records(log_path)
        md = build_phenom_audit_markdown(st, records, log_path)
        out = Path(args.out or "PHENOM_AUDIT_EXAMPLE.md")
        out.write_text(md, encoding="utf-8")
        print(f"Wrote phenomenology audit: {out.resolve()}")
        return

    if args.action == "set":
        if args.enabled is not None:
            ph["enabled"] = (str(args.enabled).lower() == "on")
        if args.min_samples is not None:
            ph["min_samples"] = max(1, _to_int(args.min_samples, _to_int(ph.get("min_samples"), 3)))
        if args.influence is not None:
            ph["influence"] = max(0.0, min(1.0, _to_float(args.influence, _to_float(ph.get("influence"), 0.35))))
        st.phenomenology = ph
        save_state(st)
        print("Phenomenology settings updated.")
        return

    if args.action == "reset":
        keep = {
            "enabled": bool(ph.get("enabled", True)),
            "min_samples": max(1, _to_int(ph.get("min_samples"), 3)),
            "influence": max(0.0, min(1.0, _to_float(ph.get("influence"), 0.35))),
            "signatures": {},
        }
        st.phenomenology = keep
        save_state(st)
        print("Phenomenology memory reset (settings kept).")
        return

def cmd_export(args) -> None:
    st = load_state(); ensure_today(st); _init_defaults(st)
    out = Path(args.out or f"prok_export_{_iso_today()}.zip")
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        if STATE_FILE.exists():
            z.write(STATE_FILE, arcname=STATE_FILE.name)
        if LOG_FILE.exists():
            z.write(LOG_FILE, arcname=LOG_FILE.name)
    print(f"Exported bundle: {out.resolve()}")

def cmd_cca(args) -> None:
    # v4 requirement: placeholder only
    st = load_state(); ensure_today(st); _init_defaults(st)
    print("CCA is a placeholder in v4 and is NOT used.")
    print(_json_dump({"use_cca": st.use_cca, "cca_model_path": st.cca_model_path}))

def cmd_ethics(args) -> None:
    st = load_state(); ensure_today(st); _init_defaults(st)
    ph = st.phenomenology or {}
    print("ETHICAL CLARIFICATION (Aristotelian, operational):")
    print("- Physics: move from delayed potential to enacted timely work.")
    print("- Categories: task/domain/time/quality/relation structure decisions.")
    print("- Ethics: target is flow-capable engaged mean, not maximal pressure.")
    print("- Politics: local-first, non-coercive, user pause/quit rights preserved.")
    print("")
    print("UNEQUAL DOCTRINE OF THE MEAN:")
    print("- Deficiency risk: procrastination/avoidant delay (primary).")
    print("- Center target: sustainable engaged progress.")
    print("- Excess risk: over-control/annoyance burden.")
    print("- Default asymmetry: penalty(procrastination) > penalty(annoyance),")
    print("  with feasibility guards dominant (slack violations are critical).")
    print("")
    print("CURRENT RUNTIME GUARDRAILS:")
    print(f"- pause_policy: {st.genome.get('g9_pause_policy')}")
    print(f"- replan_threshold: {st.genome.get('g8_replan_threshold')}")
    print(f"- study_mode: {st.study_mode} (burden/helped ratings when enabled)")
    print(f"- phenomenology.enabled: {bool(ph.get('enabled', True))}")
    print(f"- phenomenology.min_samples: {_to_int(ph.get('min_samples'), 3)}")
    print(f"- phenomenology.influence: {_to_float(ph.get('influence'), 0.35):.2f}")
    print(f"- use_cca: {st.use_cca} (placeholder only)")

# ----------------------------
# Parser
# ----------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="prok", description="PROK v5 — multi-task Ashby regulator with per-task TMT + domain recursion + algedonic alarm.")

    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("plan")
    sp.add_argument("--domain", required=True)
    sp.add_argument("--name", required=True)
    sp.add_argument("--total-hours", type=float, required=True)
    sp.add_argument("--deadline", required=True, help="YYYY-MM-DD or 3mo/2w/10d")
    sp.add_argument("--block-min", type=int, default=None)
    sp.add_argument("--tiny-min", type=int, default=None)
    sp.add_argument("--day-start", default=None)
    sp.add_argument("--day-end", default=None)
    sp.add_argument("--max-blocks-per-day", type=int, default=None)
    sp.set_defaults(func=cmd_plan)

    stask = sub.add_parser("task")
    stsub = stask.add_subparsers(dest="action", required=True)
    st_add = stsub.add_parser("add")
    st_add.add_argument("--domain", required=True)
    st_add.add_argument("--name", required=True)
    st_add.add_argument("--hours", type=float, dest="total_hours", required=True)
    st_add.add_argument("--deadline", required=True, help="YYYY-MM-DD or 3mo/2w/10d")
    st_add.add_argument("--block-min", type=int, default=None)
    st_add.add_argument("--tiny-min", type=int, default=None)
    st_add.add_argument("--day-start", default=None)
    st_add.add_argument("--day-end", default=None)
    st_add.add_argument("--max-blocks-per-day", type=int, default=None)
    st_add.set_defaults(func=cmd_task_add)
    st_list = stsub.add_parser("list"); st_list.set_defaults(func=cmd_task_list)
    st_sa = stsub.add_parser("set-active"); st_sa.add_argument("task_id"); st_sa.set_defaults(func=cmd_task_set_active)
    st_rm = stsub.add_parser("remove"); st_rm.add_argument("task_id"); st_rm.set_defaults(func=cmd_task_remove)
    st_edit = stsub.add_parser("edit")
    st_edit.add_argument("task_id")
    st_edit.add_argument("--domain", default=None)
    st_edit.add_argument("--name", default=None)
    st_edit.add_argument("--deadline", default=None)
    st_edit.add_argument("--hours", type=float, dest="total_hours", default=None)
    st_edit.add_argument("--remaining-minutes", type=int, default=None)
    st_edit.set_defaults(func=cmd_task_edit)

    sb = sub.add_parser("blackout")
    sbsub = sb.add_subparsers(dest="action", required=True)
    sb_add = sbsub.add_parser("add")
    sb_add.add_argument("--weekday", default=None, help="Mon-Fri or Tue")
    sb_add.add_argument("--start", default="08:00")
    sb_add.add_argument("--end", default="15:00")
    sb_add.add_argument("--date", default=None, help="YYYY-MM-DD")
    sb_add.add_argument("--daterange", default=None, help="YYYY-MM-DD..YYYY-MM-DD")
    sb_add.add_argument("--label", default="")
    sb_add.set_defaults(func=cmd_blackout)
    sb_list = sbsub.add_parser("list"); sb_list.set_defaults(func=cmd_blackout)
    sb_clear = sbsub.add_parser("clear"); sb_clear.set_defaults(func=cmd_blackout)

    stoday = sub.add_parser("today"); stoday.set_defaults(func=cmd_today)

    sr = sub.add_parser("run")
    sr.add_argument("--task", default=None)
    sr.add_argument("--domain", default=None)
    sr.add_argument("--name", default=None)
    sr.add_argument("--minutes", type=int, default=None)
    sr.add_argument("--checks", choices=["off","random","auto"], default="auto")
    sr.add_argument("--min-gap", type=int, default=None)
    sr.add_argument("--max-gap", type=int, default=None)
    sr.add_argument("--ask-trigger", choices=["never","p","dp","always"], default="dp")
    sr.add_argument("--no-haghbin", action="store_true")
    sr.add_argument("--fitness", default=None)
    sr.set_defaults(func=cmd_run)

    slog = sub.add_parser("log")
    slog.add_argument("--task", default=None)
    slog.add_argument("--domain", default=None)
    slog.add_argument("--name", default=None)
    slog.add_argument("--minutes", type=int, required=True)
    slog.set_defaults(func=cmd_log)

    ss = sub.add_parser("status"); ss.set_defaults(func=cmd_status)
    srs = sub.add_parser("reset"); srs.set_defaults(func=cmd_reset)

    scfg = sub.add_parser("config")
    scfgsub = scfg.add_subparsers(dest="action", required=True)
    scfgsub.add_parser("show").set_defaults(func=cmd_config)
    scfg_set = scfgsub.add_parser("set")
    scfg_set.add_argument("--block-min", type=int, default=None)
    scfg_set.add_argument("--tiny-min", type=int, default=None)
    scfg_set.add_argument("--day-start", default=None)
    scfg_set.add_argument("--day-end", default=None)
    scfg_set.add_argument("--max-blocks-per-day", type=int, default=None)
    scfg_set.set_defaults(func=cmd_config)

    sg = sub.add_parser("genome")
    sgsub = sg.add_subparsers(dest="action", required=True)
    sgsub.add_parser("show").set_defaults(func=cmd_genome)
    sg_wd = sgsub.add_parser("write-default"); sg_wd.add_argument("--out", default=None); sg_wd.set_defaults(func=cmd_genome)
    sg_load = sgsub.add_parser("load"); sg_load.add_argument("path"); sg_load.set_defaults(func=cmd_genome)

    sf = sub.add_parser("fitness")
    sfsub = sf.add_subparsers(dest="action", required=True)
    sfsub.add_parser("show").set_defaults(func=cmd_fitness)
    sf_wd = sfsub.add_parser("write-default"); sf_wd.add_argument("--out", default=None); sf_wd.set_defaults(func=cmd_fitness)
    sf_set = sfsub.add_parser("set"); sf_set.add_argument("path"); sf_set.set_defaults(func=cmd_fitness)

    sstudy = sub.add_parser("study")
    sst = sstudy.add_subparsers(dest="action", required=True)
    sst.add_parser("on").set_defaults(func=cmd_study)
    sst.add_parser("off").set_defaults(func=cmd_study)
    sid = sst.add_parser("id"); sid.add_argument("value"); sid.set_defaults(func=cmd_study)
    sst.add_parser("show").set_defaults(func=cmd_study)

    sgoal = sub.add_parser("goal")
    sg2 = sgoal.add_subparsers(dest="action", required=True)
    sg2.add_parser("show").set_defaults(func=cmd_goal)
    sg2.add_parser("clear").set_defaults(func=cmd_goal)
    sg2.add_parser("wizard").set_defaults(func=cmd_goal)
    sg2_set = sg2.add_parser("set"); sg2_set.add_argument("path"); sg2_set.set_defaults(func=cmd_goal)

    slang = sub.add_parser("lang")
    sl2 = slang.add_subparsers(dest="action", required=True)
    sl2.add_parser("show").set_defaults(func=cmd_lang)
    sl2.add_parser("clear").set_defaults(func=cmd_lang)
    sl2.add_parser("wizard").set_defaults(func=cmd_lang)
    sl2_set = sl2.add_parser("set"); sl2_set.add_argument("path"); sl2_set.set_defaults(func=cmd_lang)

    sph = sub.add_parser("phenom")
    sph2 = sph.add_subparsers(dest="action", required=True)
    sph2.add_parser("show").set_defaults(func=cmd_phenom)
    sph_reset = sph2.add_parser("reset"); sph_reset.set_defaults(func=cmd_phenom)
    sph_audit = sph2.add_parser("audit")
    sph_audit.add_argument("--out", default="PHENOM_AUDIT_EXAMPLE.md")
    sph_audit.add_argument("--log", default=str(LOG_FILE))
    sph_audit.set_defaults(func=cmd_phenom)
    sph_set = sph2.add_parser("set")
    sph_set.add_argument("--enabled", choices=["on", "off"], default=None)
    sph_set.add_argument("--min-samples", type=int, default=None)
    sph_set.add_argument("--influence", type=float, default=None)
    sph_set.set_defaults(func=cmd_phenom)

    sexp = sub.add_parser("export")
    sexp.add_argument("--out", default=None)
    sexp.set_defaults(func=cmd_export)

    seth = sub.add_parser("ethics")
    sethsub = seth.add_subparsers(dest="action", required=True)
    sethsub.add_parser("show").set_defaults(func=cmd_ethics)

    scca = sub.add_parser("cca"); scca.set_defaults(func=cmd_cca)

    return p

def main(argv: List[str]) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
