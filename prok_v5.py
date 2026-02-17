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
import copy
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
    source: str = "manual"
    uid: Optional[str] = None

@dataclass
class DateBlackout:
    date: str  # YYYY-MM-DD
    label: str = ""
    source: str = "manual"
    uid: Optional[str] = None

@dataclass
class DateRangeBlackout:
    start: str
    end: str
    label: str = ""
    source: str = "manual"
    uid: Optional[str] = None

@dataclass
class DateTimeBlackout:
    date: str  # YYYY-MM-DD
    start_hhmm: str
    end_hhmm: str
    label: str = ""
    source: str = "manual"
    uid: Optional[str] = None

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
        "mode": "fep",  # phi|fep|dag
        "lambda_procrast": 0.60,
        "mu_overcontrol": 0.25,
        "beta_ambiguity": 0.20,
        "eta_epistemic": 0.10,
        "dag_kappa": 0.30,
        "dag_adjust": ["domain", "label", "trigger", "slack", "drift"],
        "dag_min_samples": 5,
        "dag_laplace": 1.0,
        "signatures": {},
    }

def default_homeostat_text() -> str:
    return (
        "require slack != neg\n"
        "require recovery != slow\n"
        "\n"
        "penalty 2: annoyance = high\n"
        "penalty 1: drift = high\n"
        "penalty 1: progress = down\n"
        "\n"
        "prefer 1: slack in {ok, high}\n"
        "prefer 1: recovery in {fast, instant}\n"
        "prefer 1: stability in {steady, stable}\n"
    )

def default_nk_state() -> Dict[str, Any]:
    neighbors = {
        "g1": ["g2", "g3"],
        "g2": ["g1", "g4"],
        "g3": ["g1", "g5"],
        "g4": ["g2", "g6"],
        "g5": ["g3", "g7"],
        "g6": ["g4", "g8"],
        "g7": ["g5", "g9"],
        "g8": ["g6", "g10"],
        "g9": ["g7", "g10"],
        "g10": ["g8", "g9"],
    }
    return {
        "enabled": True,
        "k": 2,
        "neighbors": neighbors,
        "main": {},
        "pair": {},
        "last_genotype": {},
        "last_source": "none",
        "last_pred_score": None,
        "history": [],
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
# Homeostat DSL (require/penalty/prefer)
# ----------------------------

class HomeostatParseError(Exception):
    pass

@dataclass
class WeightedExpr:
    weight: int
    expr: Expr

@dataclass
class HomeostatProgram:
    requires: List[Expr]
    penalties: List[WeightedExpr]
    prefers: List[WeightedExpr]

def parse_pred_expr(src: str) -> Expr:
    toks = _tokenize(src.strip())
    p = _Parser(toks)
    expr = p.parse_expr()
    if p.peek()[0] != "EOF":
        raise HomeostatParseError(f"Unexpected token after expression: {p.peek()}")
    return expr

def parse_homeostat_text(src: str) -> HomeostatProgram:
    requires: List[Expr] = []
    penalties: List[WeightedExpr] = []
    prefers: List[WeightedExpr] = []
    for ln, raw in enumerate(src.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.fullmatch(r"require\s+(.+)", line, flags=re.IGNORECASE)
        if m:
            requires.append(parse_pred_expr(m.group(1)))
            continue
        m = re.fullmatch(r"penalty\s+(\d+)\s*:\s*(.+)", line, flags=re.IGNORECASE)
        if m:
            penalties.append(WeightedExpr(weight=max(0, int(m.group(1))), expr=parse_pred_expr(m.group(2))))
            continue
        m = re.fullmatch(r"prefer\s+(\d+)\s*:\s*(.+)", line, flags=re.IGNORECASE)
        if m:
            prefers.append(WeightedExpr(weight=max(0, int(m.group(1))), expr=parse_pred_expr(m.group(2))))
            continue
        raise HomeostatParseError(f"Invalid homeostat line {ln}: {raw}")
    return HomeostatProgram(requires=requires, penalties=penalties, prefers=prefers)

def parse_homeostat(path: Path) -> HomeostatProgram:
    return parse_homeostat_text(path.read_text(encoding="utf-8"))

def built_in_homeostat() -> HomeostatProgram:
    return parse_homeostat_text(default_homeostat_text())

def apply_homeostat(program: HomeostatProgram, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    passed: List[Dict[str, Any]] = []
    for c in candidates:
        ctx = c.get("ctx") or {}
        if all(eval_expr(r, ctx) for r in program.requires):
            passed.append(c)

    active = passed if passed else candidates
    out: List[Dict[str, Any]] = []
    for c in active:
        ctx = c.get("ctx") or {}
        penalty_score = sum(w.weight for w in program.penalties if eval_expr(w.expr, ctx))
        prefer_bonus = sum(w.weight for w in program.prefers if eval_expr(w.expr, ctx))
        c2 = dict(c)
        c2["homeostat_penalty"] = int(penalty_score)
        c2["homeostat_bonus"] = int(prefer_bonus)
        c2["homeostat_score"] = int(penalty_score - prefer_bonus)
        out.append(c2)
    return out

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
    datetime_blackouts: List[DateTimeBlackout] = None

    # Habit/momentum
    today_min_done: bool = False
    last_min_date: Optional[str] = None
    streak: int = 0
    last_streak_date: Optional[str] = None

    # Control layers
    genome: Dict[str, Any] = None
    fitness_path: Optional[str] = None
    homeostat_path: Optional[str] = None
    nk_state: Dict[str, Any] = None

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
    policy_constraints: Dict[str, Any] = None

def _init_defaults(st: ProkState) -> None:
    if st.weekly_blackouts is None: st.weekly_blackouts = []
    if st.date_blackouts is None: st.date_blackouts = []
    if st.daterange_blackouts is None: st.daterange_blackouts = []
    if st.datetime_blackouts is None: st.datetime_blackouts = []
    if st.tasks is None: st.tasks = []
    if st.genome is None: st.genome = default_genome()
    if "version" not in st.genome: st.genome = default_genome()
    if st.nk_state is None: st.nk_state = default_nk_state()
    if not isinstance(st.nk_state, dict): st.nk_state = default_nk_state()
    if "neighbors" not in st.nk_state: st.nk_state["neighbors"] = default_nk_state()["neighbors"]
    if "main" not in st.nk_state: st.nk_state["main"] = {}
    if "pair" not in st.nk_state: st.nk_state["pair"] = {}
    if "history" not in st.nk_state: st.nk_state["history"] = []
    if "enabled" not in st.nk_state: st.nk_state["enabled"] = True
    if "k" not in st.nk_state: st.nk_state["k"] = 2
    if "last_genotype" not in st.nk_state: st.nk_state["last_genotype"] = {}
    if "last_source" not in st.nk_state: st.nk_state["last_source"] = "none"
    if "last_pred_score" not in st.nk_state: st.nk_state["last_pred_score"] = None
    if st.goal_profile is None: st.goal_profile = default_goal_profile()
    if st.language_pack is None: st.language_pack = dict(DEFAULT_LANG)
    if st.phenomenology is None: st.phenomenology = default_phenomenology()
    if not isinstance(st.phenomenology, dict): st.phenomenology = default_phenomenology()
    if "enabled" not in st.phenomenology: st.phenomenology["enabled"] = True
    if "min_samples" not in st.phenomenology: st.phenomenology["min_samples"] = 3
    if "influence" not in st.phenomenology: st.phenomenology["influence"] = 0.35
    if "mode" not in st.phenomenology: st.phenomenology["mode"] = "fep"
    st.phenomenology["mode"] = str(st.phenomenology.get("mode") or "fep").lower()
    if st.phenomenology["mode"] not in ("phi", "fep", "dag"):
        st.phenomenology["mode"] = "fep"
    if "lambda_procrast" not in st.phenomenology: st.phenomenology["lambda_procrast"] = 0.60
    if "mu_overcontrol" not in st.phenomenology: st.phenomenology["mu_overcontrol"] = 0.25
    if "beta_ambiguity" not in st.phenomenology: st.phenomenology["beta_ambiguity"] = 0.20
    if "eta_epistemic" not in st.phenomenology: st.phenomenology["eta_epistemic"] = 0.10
    if "dag_kappa" not in st.phenomenology: st.phenomenology["dag_kappa"] = 0.30
    if "dag_adjust" not in st.phenomenology: st.phenomenology["dag_adjust"] = ["domain", "label", "trigger", "slack", "drift"]
    if "dag_min_samples" not in st.phenomenology: st.phenomenology["dag_min_samples"] = 5
    if "dag_laplace" not in st.phenomenology: st.phenomenology["dag_laplace"] = 1.0
    if "signatures" not in st.phenomenology or not isinstance(st.phenomenology.get("signatures"), dict):
        st.phenomenology["signatures"] = {}
    if st.domains is None: st.domains = {}
    if st.policy_constraints is None:
        st.policy_constraints = {
            "max_checks_per_hour": 10,
            "quiet_windows": [],  # ["13:00-15:00", "21:00-22:00"]
        }
    if "max_checks_per_hour" not in st.policy_constraints:
        st.policy_constraints["max_checks_per_hour"] = 10
    if "quiet_windows" not in st.policy_constraints or not isinstance(st.policy_constraints.get("quiet_windows"), list):
        st.policy_constraints["quiet_windows"] = []

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
        datetime_blackouts=[DateTimeBlackout(**x) for x in data.get("datetime_blackouts", [])],
        today_min_done=bool(data.get("today_min_done", False)),
        last_min_date=data.get("last_min_date"),
        streak=_to_int(data.get("streak"), 0),
        last_streak_date=data.get("last_streak_date"),
        genome=(data.get("genome") or None),
        fitness_path=data.get("fitness_path"),
        homeostat_path=data.get("homeostat_path"),
        nk_state=(data.get("nk_state") or None),
        domains=_load_domains(data),
        participant_id=_to_str(data.get("participant_id"), ""),
        study_mode=bool(data.get("study_mode", False)),
        goal_profile=(data.get("goal_profile") or None),
        language_pack=(data.get("language_pack") or None),
        phenomenology=(data.get("phenomenology") or None),
        use_cca=bool(data.get("use_cca", False)),
        cca_model_path=data.get("cca_model_path"),
        policy_constraints=(data.get("policy_constraints") or None),
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

def _phenomenology_action_score(
    stats: Dict[str, Any],
    mode: str,
    lambda_procrast: float,
    mu_overcontrol: float,
    beta_ambiguity: float,
    eta_epistemic: float,
) -> Tuple[float, int, Dict[str, float]]:
    rated = _to_int(stats.get("rated_helped"), 0)
    yes = _to_int(stats.get("helped_yes"), 0)
    burden_n = _to_int(stats.get("burden_n"), 0)
    burden_sum = _to_float(stats.get("burden_sum"), 0.0)
    tries = _to_int(stats.get("tries"), 0)
    procrast_n = _to_int(stats.get("procrastination_events"), 0)
    high_burden_n = _to_int(stats.get("high_burden_n"), 0)

    # Single-value transformation (phi):
    # viable - lambda*procrast_risk - mu*overcontrol_risk
    p_viable = (yes + 1.0) / (rated + 2.0)  # helped as viability proxy
    p_procrast = (procrast_n + 1.0) / (tries + 2.0)
    p_overcontrol = (high_burden_n + 1.0) / (burden_n + 2.0)
    phi = p_viable - float(lambda_procrast) * p_procrast - float(mu_overcontrol) * p_overcontrol
    # Uncertainty/exploration terms for lightweight active-inference-style selection.
    denom = max(1e-9, 1.0 + float(lambda_procrast) + float(mu_overcontrol))
    ambiguity = (
        (p_viable * (1.0 - p_viable))
        + float(lambda_procrast) * (p_procrast * (1.0 - p_procrast))
        + float(mu_overcontrol) * (p_overcontrol * (1.0 - p_overcontrol))
    ) / denom
    epistemic = 1.0 / math.sqrt(max(1.0, float(tries) + 1.0))

    score = phi
    if str(mode).lower() == "fep":
        score = phi - float(beta_ambiguity) * ambiguity + float(eta_epistemic) * epistemic
    return score, rated, {
        "mode": str(mode).lower(),
        "phi": phi,
        "ambiguity": ambiguity,
        "epistemic": epistemic,
        "p_viable": p_viable,
        "p_procrast": p_procrast,
        "p_overcontrol": p_overcontrol,
    }

def _event_covariates_for_dag(ev: Dict[str, Any]) -> Dict[str, Any]:
    ctx = dict(ev.get("homeostat_ctx") or {})
    if "domain" not in ctx:
        ctx["domain"] = str(ev.get("domain") or "unknown").lower()
    if "label" not in ctx:
        ctx["label"] = str(ev.get("label") or "UNKNOWN")
    if "trigger" not in ctx:
        ctx["trigger"] = str(ev.get("trigger") or "other")
    return ctx

def _dag_key(ctx: Dict[str, Any], adjust: List[str]) -> Tuple[str, ...]:
    return tuple(str(ctx.get(k, "NA")) for k in adjust)

def _safe_div(num: float, den: float) -> Optional[float]:
    return (num / den) if den > 0 else None

def _dag_adjusted_scores(
    st: ProkState,
    candidate_names: List[str],
    current_ctx: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    ph = st.phenomenology or {}
    adjust = [str(x) for x in (ph.get("dag_adjust") or ["domain", "label", "trigger", "slack", "drift"])]
    dag_min = max(1, _to_int(ph.get("dag_min_samples"), 5))
    laplace = max(0.0, _to_float(ph.get("dag_laplace"), 1.0))
    acts = set(candidate_names)
    notes: List[str] = []

    records = _load_log_records(LOG_FILE)
    events = [r for r in records if str(r.get("kind")) == "course_correct" and str(r.get("chosen") or "") in acts]
    if not events:
        notes.append("DAG fallback: no historical course_correct data for candidate actions.")
        return {}, notes

    # Localize to nearest context first (domain,label,trigger); fallback to all events.
    d0 = str(current_ctx.get("domain") or "unknown").lower()
    l0 = str(current_ctx.get("label") or "UNKNOWN")
    t0 = str(current_ctx.get("trigger") or "other")
    local = [
        e for e in events
        if str((e.get("domain") or "unknown")).lower() == d0
        and str(e.get("label") or "UNKNOWN") == l0
        and str(e.get("trigger") or "other") == t0
    ]
    pool = local if local else events
    if not local:
        notes.append("DAG fallback: no local (domain/label/trigger) data; using pooled historical data.")
    if len(pool) < dag_min:
        notes.append(f"DAG sparse-data: pool={len(pool)} < dag_min_samples={dag_min}; correction likely inactive.")

    cell_n: Dict[Tuple[str, ...], int] = defaultdict(int)
    cell_action_n: Dict[Tuple[str, ...], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    universe = sorted(set(str(e.get("chosen") or "") for e in pool if e.get("chosen")))
    if not universe:
        notes.append("DAG fallback: no action labels in historical pool.")
        return {}, notes
    A = float(len(universe))

    parsed: List[Tuple[Dict[str, Any], Tuple[str, ...], str]] = []
    for e in pool:
        a = str(e.get("chosen") or "")
        if not a:
            continue
        cov = _event_covariates_for_dag(e)
        key = _dag_key(cov, adjust)
        parsed.append((e, key, a))
        cell_n[key] += 1
        cell_action_n[key][a] += 1

    out: Dict[str, Dict[str, Any]] = {}
    for a in candidate_names:
        w_help_num = 0.0
        w_help_den = 0.0
        w_bur_num = 0.0
        w_bur_den = 0.0
        n_help = 0
        n_bur = 0
        for e, key, ae in parsed:
            if ae != a:
                continue
            nc = float(cell_n.get(key, 0))
            nac = float((cell_action_n.get(key) or {}).get(a, 0))
            p = (nac + laplace) / max(1e-9, (nc + laplace * A))
            w = min(25.0, 1.0 / max(1e-6, p))
            helped = e.get("helped")
            if helped is not None:
                y = 1.0 if bool(helped) else 0.0
                w_help_num += w * y
                w_help_den += w
                n_help += 1
            if isinstance(e.get("burden"), (int, float)):
                b = float(e.get("burden"))
                yb = 1.0 if b >= 4.0 else 0.0
                w_bur_num += w * yb
                w_bur_den += w
                n_bur += 1

        do_help = _safe_div(w_help_num, w_help_den)
        do_over = _safe_div(w_bur_num, w_bur_den)
        n_eff = min(n_help, n_bur) if (n_help and n_bur) else max(n_help, n_bur)
        if n_eff < dag_min:
            continue
        out[a] = {
            "do_helped": do_help,
            "do_overcontrol": do_over,
            "n": n_eff,
            "pool_n": len(pool),
            "local_pool": bool(local),
        }
    if not out:
        notes.append("DAG fallback: no action reached minimum adjusted sample support.")
    return out, notes

def phenomenology_recommendation(
    st: ProkState,
    signature: str,
    candidate_names: List[str],
    current_ctx: Optional[Dict[str, Any]] = None,
) -> Optional[Tuple[str, float, int, Dict[str, float]]]:
    ph = st.phenomenology or {}
    if not bool(ph.get("enabled", True)):
        return None
    min_samples = max(1, _to_int(ph.get("min_samples"), 3))
    mode = str(ph.get("mode") or "fep").lower()
    lambda_procrast = _to_float(ph.get("lambda_procrast"), 0.60)
    mu_overcontrol = _to_float(ph.get("mu_overcontrol"), 0.25)
    beta_ambiguity = _to_float(ph.get("beta_ambiguity"), 0.20)
    eta_epistemic = _to_float(ph.get("eta_epistemic"), 0.10)
    entry = (ph.get("signatures") or {}).get(signature) or {}
    actions = entry.get("actions") or {}
    allowed = set(candidate_names)
    dag_scores: Dict[str, Dict[str, Any]] = {}
    dag_notes: List[str] = []
    if mode == "dag":
        dag_scores, dag_notes = _dag_adjusted_scores(st, candidate_names, current_ctx or {})
    dag_kappa = _to_float(ph.get("dag_kappa"), 0.30)

    best_name = None
    best_score = -1e9
    best_n = 0
    best_components: Dict[str, float] = {}
    for name, stats in actions.items():
        if name not in allowed:
            continue
        score, rated, components = _phenomenology_action_score(
            stats,
            mode=("fep" if mode == "fep" else "phi"),
            lambda_procrast=lambda_procrast,
            mu_overcontrol=mu_overcontrol,
            beta_ambiguity=beta_ambiguity,
            eta_epistemic=eta_epistemic,
        )
        if mode == "dag":
            dag = dag_scores.get(name) or {}
            do_help = dag.get("do_helped")
            do_over = dag.get("do_overcontrol")
            if do_help is not None or do_over is not None:
                # Causal correction from backdoor-adjusted do-estimates.
                causal = (0.0 if do_help is None else (float(do_help) - 0.5)) - float(mu_overcontrol) * (0.0 if do_over is None else float(do_over))
                score = score + float(dag_kappa) * causal
                components["dag_causal"] = causal
                components["dag_do_helped"] = float(do_help) if do_help is not None else None
                components["dag_do_overcontrol"] = float(do_over) if do_over is not None else None
                components["dag_n"] = int(dag.get("n") or 0)
                components["dag_pool_n"] = int(dag.get("pool_n") or 0)
                components["dag_local_pool"] = 1.0 if dag.get("local_pool") else 0.0
            components["mode"] = "dag"
            if dag_notes:
                components["dag_notes"] = list(dag_notes)
        if rated < min_samples:
            continue
        if score > best_score:
            best_name = name
            best_score = score
            best_n = rated
            best_components = components
    if not best_name:
        return None
    return best_name, best_score, best_n, best_components

def phenomenology_record_outcome(
    st: ProkState,
    signature: str,
    chosen: str,
    ratings: Dict[str, Any],
    label: str,
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
        "procrastination_events": 0,
        "high_burden_n": 0,
    })
    a["tries"] = _to_int(a.get("tries"), 0) + 1
    if str(label).upper() == "PROCRASTINATION":
        a["procrastination_events"] = _to_int(a.get("procrastination_events"), 0) + 1
    if ratings.get("helped") is not None:
        a["rated_helped"] = _to_int(a.get("rated_helped"), 0) + 1
        if bool(ratings.get("helped")):
            a["helped_yes"] = _to_int(a.get("helped_yes"), 0) + 1
    if ratings.get("burden") is not None:
        a["burden_n"] = _to_int(a.get("burden_n"), 0) + 1
        b = float(ratings.get("burden"))
        a["burden_sum"] = _to_float(a.get("burden_sum"), 0.0) + b
        if b >= 4.0:
            a["high_burden_n"] = _to_int(a.get("high_burden_n"), 0) + 1

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
            score, rated, _ = _phenomenology_action_score({
                "rated_helped": stx.get("rated_helped", 0),
                "helped_yes": stx.get("helped_yes", 0),
                "burden_n": len(stx.get("burden", [])),
                "burden_sum": sum(stx.get("burden", [])),
                "tries": stx.get("tries", 0),
                "procrastination_events": stx.get("procrastination_events", 0),
                "high_burden_n": stx.get("high_burden_n", 0),
            },
                mode=str(ph.get("mode") or "fep").lower(),
                lambda_procrast=_to_float(ph.get("lambda_procrast"), 0.60),
                mu_overcontrol=_to_float(ph.get("mu_overcontrol"), 0.25),
                beta_ambiguity=_to_float(ph.get("beta_ambiguity"), 0.20),
                eta_epistemic=_to_float(ph.get("eta_epistemic"), 0.10),
            )
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

    phenom_n = (
        t2.get("fitness+phenomenology", {}).get("n", 0)
        + t2.get("homeostat+fitness+phenomenology", {}).get("n", 0)
    )
    baseline_n = (
        t2.get("fitness", {}).get("n", 0)
        + t2.get("homeostat+fitness", {}).get("n", 0)
    )
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
    lines.append(f"  - `mode`: `{str(ph.get('mode') or 'fep').lower()}`")
    lines.append(f"  - `lambda_procrast`: `{_to_float(ph.get('lambda_procrast'), 0.60):.2f}`")
    lines.append(f"  - `mu_overcontrol`: `{_to_float(ph.get('mu_overcontrol'), 0.25):.2f}`")
    lines.append(f"  - `beta_ambiguity`: `{_to_float(ph.get('beta_ambiguity'), 0.20):.2f}`")
    lines.append(f"  - `eta_epistemic`: `{_to_float(ph.get('eta_epistemic'), 0.10):.2f}`")
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
    lines.append(f"- Baseline-only selections (`selection_source in {{fitness,homeostat+fitness}}`): `{baseline_n if sel_field_present else 'N/A (legacy schema)'}`")
    lines.append(f"- Phenomenology selections (`selection_source includes phenomenology`): `{phenom_n}`")
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

def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    xs = sorted((a, b) for a, b in intervals if b > a)
    out: List[Tuple[int, int]] = [xs[0]]
    for a, b in xs[1:]:
        la, lb = out[-1]
        if a <= lb:
            out[-1] = (la, max(lb, b))
        else:
            out.append((a, b))
    return out

def blocked_intervals_for_day(st: ProkState, d: dt.date) -> List[Tuple[int, int]]:
    w0 = _parse_hhmm(st.day_start)
    w1 = _parse_hhmm(st.day_end)
    if w1 <= w0:
        return []
    intervals: List[Tuple[int, int]] = []

    wd = d.weekday()
    for b in st.weekly_blackouts:
        if not _weekly_active(b, wd):
            continue
        s0 = _parse_hhmm(b.start_hhmm)
        s1 = _parse_hhmm(b.end_hhmm)
        if s1 <= s0:
            intervals.append((w0, w1))
        else:
            intervals.append((max(w0, s0), min(w1, s1)))

    ds = d.isoformat()
    for b in st.datetime_blackouts:
        if b.date != ds:
            continue
        s0 = _parse_hhmm(b.start_hhmm)
        s1 = _parse_hhmm(b.end_hhmm)
        if s1 <= s0:
            continue
        intervals.append((max(w0, s0), min(w1, s1)))

    return _merge_intervals(intervals)

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

def minutes_blocked_for_day(st: ProkState, d: dt.date) -> int:
    return sum((b - a) for a, b in blocked_intervals_for_day(st, d))

def day_capacity_blocks(st: ProkState, d: dt.date) -> int:
    if is_date_blackout(st, d):
        return 0
    w0 = _parse_hhmm(st.day_start)
    w1 = _parse_hhmm(st.day_end)
    if w1 <= w0:
        return 0
    workable = w1 - w0
    blocked = minutes_blocked_for_day(st, d)
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
        slack_blocks = task_slack_blocks(st, task)
        ds = get_domain_state(st, task.domain)
        recovery = _recovery_bin(label1, trigger, theta1)
        drift_bin = _drift_bin(task.recent_drift_count)
        annoy_bin = _annoyance_bin(gaps[0])
        return {
            "slack": _slack_bin_homeostat(slack_blocks),
            "E": theta1.E, "V": theta1.V, "D": theta1.D, "G": theta1.G,
            "trigger": trigger,
            "procrastinating": (label1 == "PROCRASTINATION"),
            "intent_gap": (label1 == "PROCRASTINATION"),
            "expected_cost": True,
            "reward_near": (theta1.D == "N"),
            "recover_fast": recovery in ("fast", "instant"),
            "checks": "many" if gaps[0] <= 3 else ("medium" if gaps[0] <= 6 else "few"),
            "annoying": (gaps[0] <= 2) and _goal_early_has(goal, "low_annoyance", tiers=3),
            "annoyance": annoy_bin,
            "drift": drift_bin,
            "progress": _progress_bin(task),
            "recovery": recovery,
            "stability": _stability_bin(float(ds.drift_ema)),
            "stability_num": float(ds.drift_ema),
            "slack_blocks": int(slack_blocks),
            "circadian": circadian_bin(),
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

def select_recovery_with_trace(
    st: ProkState,
    task: Task,
    label: str,
    trigger: str,
    check_gaps: Tuple[int, int],
    fitness_program: Optional[ProkFitProgram],
    homeostat_program: Optional[HomeostatProgram],
) -> Dict[str, Any]:
    base_prog = fitness_program or load_fitness_program(st) or built_in_fitness()
    prog = fitness_from_goal_profile(base_prog, st.goal_profile or default_goal_profile())
    hprog = homeostat_program or load_homeostat_program(st) or built_in_homeostat()

    candidates = build_recovery_candidates(st, task=task, label=label, trigger=trigger, check_gaps=check_gaps)
    h_candidates = apply_homeostat(hprog, candidates)
    fit_scored = rank_candidates_scored(prog, h_candidates)
    fit_lex = {c["name"]: lex for lex, c in fit_scored}
    ranked = sorted(
        h_candidates,
        key=lambda c: (int(c.get("homeostat_score", 0)), fit_lex.get(c.get("name"), tuple())),
    )
    if not ranked:
        raise SystemExit("No admissible recovery candidates after homeostat/fitness filtering.")

    best = ranked[0]
    selection_source = "homeostat+fitness"
    phenom_mode = str((st.phenomenology or {}).get("mode") or "fep").lower()
    phenom_score: Optional[float] = None
    phenom_components: Optional[Dict[str, float]] = None
    shortlist = ranked[:3]

    ctx0 = dict(shortlist[0].get("ctx") or {}) if shortlist else {}
    ctx0["domain"] = task.domain
    ctx0["label"] = label
    ctx0["trigger"] = trigger
    rec = phenomenology_recommendation(st, phenomenology_signature(task, label, trigger), [c["name"] for c in shortlist], current_ctx=ctx0)
    if rec:
        rec_name, rec_score, rec_n, rec_components = rec
        phenom_score = rec_score
        phenom_components = rec_components
        influence = _to_float((st.phenomenology or {}).get("influence"), 0.35)
        if rec_name != best["name"] and rec_score >= influence:
            alt = next((c for c in shortlist if c["name"] == rec_name), None)
            if alt is not None:
                best = alt
                selection_source = "homeostat+fitness+phenomenology"
    elif phenom_mode == "dag":
        _, dag_notes = _dag_adjusted_scores(st, [c["name"] for c in shortlist], ctx0)
        if dag_notes:
            phenom_components = {"mode": "dag", "dag_notes": dag_notes}

    fit_rank_names = [c["name"] for _, c in fit_scored]
    trace_rows = []
    for c in ranked:
        trace_rows.append({
            "name": c.get("name"),
            "homeostat_score": c.get("homeostat_score"),
            "homeostat_penalty": c.get("homeostat_penalty"),
            "homeostat_bonus": c.get("homeostat_bonus"),
            "fitness_rank": (fit_rank_names.index(c.get("name")) + 1) if c.get("name") in fit_rank_names else None,
            "ctx": c.get("ctx"),
        })

    return {
        "best": best,
        "selection_source": selection_source,
        "phenom_mode": phenom_mode,
        "phenom_score": phenom_score,
        "phenom_components": phenom_components,
        "shortlist": shortlist,
        "trace_rows": trace_rows,
    }

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

def load_homeostat_program(st: ProkState) -> Optional[HomeostatProgram]:
    if not st.homeostat_path:
        return None
    p = Path(st.homeostat_path)
    if not p.exists():
        return None
    try:
        return parse_homeostat(p)
    except Exception as e:
        print(f"Warning: failed to parse homeostat DSL ({p}): {e}")
        return None

def _slack_bin_homeostat(blocks: int) -> str:
    if blocks < 0:
        return "neg"
    if blocks == 0:
        return "low"
    if blocks <= 2:
        return "ok"
    return "high"

def _drift_bin(n: int) -> str:
    if n >= 4:
        return "high"
    if n >= 2:
        return "med"
    if n >= 1:
        return "low"
    return "none"

def _annoyance_bin(min_gap: int) -> str:
    if min_gap <= 2:
        return "high"
    if min_gap <= 4:
        return "med"
    if min_gap <= 7:
        return "low"
    return "none"

def _stability_bin(drift_ema: float) -> str:
    if drift_ema >= 0.7:
        return "fragile"
    if drift_ema >= 0.4:
        return "wobbly"
    if drift_ema >= 0.2:
        return "steady"
    return "stable"

def _progress_bin(task: Task) -> str:
    if task.total_minutes <= 0:
        return "flat"
    done = max(0, task.total_minutes - task.remaining_minutes)
    frac = float(done) / float(max(1, task.total_minutes))
    if frac >= 0.8:
        return "surge"
    if frac >= 0.3:
        return "up"
    return "flat"

def _recovery_bin(label: str, trigger: str, theta: TMTBins) -> str:
    if label == "PROCRASTINATION" and (theta.D == "F" or theta.G == "H"):
        return "slow"
    if trigger in ("confusion", "anxiety") and theta.E == "L":
        return "ok"
    if theta.G == "L" and theta.D == "N":
        return "fast"
    return "ok"

def _nk_gene_space() -> Dict[str, List[Any]]:
    return {
        "g1_profile": ["conservative", "balanced", "aggressive"],
        "g2_profile": ["dense", "balanced", "sparse"],
        "g3_tightening": ["none", "mild", "strong"],
        "g4_classifier": ["lenient", "balanced", "strict"],
        "g5_routing": ["E-first", "V-first", "trigger", "mixed"],
        "g6_near_reward": [False, True],
        "g7_artifact": ["none", "3problems", "5flashcards", "5bullets"],
        "g8_replan_threshold": [0, 1, 2],
        "g9_pause_policy": ["soft", "hard"],
        "g10_explore": ["off", "low", "med", "high"],
    }

def _preset_g1(name: str) -> Dict[str, int]:
    presets = {
        "conservative": {"L,L,F,H": 10, "L,M,F,H": 10, "M,M,F,H": 25, "H,M,F,H": 25, "M,H,N,L": 35},
        "balanced": {"L,L,F,H": 10, "L,M,F,H": 10, "M,M,F,H": 25, "H,M,F,H": 25, "M,H,N,L": 50},
        "aggressive": {"L,L,F,H": 25, "L,M,F,H": 25, "M,M,F,H": 35, "H,M,F,H": 50, "M,H,N,L": 50},
    }
    return presets.get(name, presets["balanced"])

def _preset_g2(name: str) -> Dict[str, List[int]]:
    presets = {
        "dense": {"L,L,F,H": [2, 5], "L,M,F,H": [2, 5], "M,M,F,H": [4, 8], "M,M,N,L": [4, 8], "H,H,N,L": [6, 14]},
        "balanced": {"L,L,F,H": [2, 5], "L,M,F,H": [2, 5], "M,M,F,H": [4, 8], "M,M,N,L": [6, 14], "H,H,N,L": [8, 16]},
        "sparse": {"L,L,F,H": [4, 8], "L,M,F,H": [4, 8], "M,M,F,H": [6, 14], "M,M,N,L": [8, 16], "H,H,N,L": [10, 20]},
    }
    return presets.get(name, presets["balanced"])

def _genotype_from_genome(genome: Dict[str, Any]) -> Dict[str, Any]:
    g = genome or default_genome()
    gp = {
        "g1_profile": "balanced",
        "g2_profile": "balanced",
        "g3_tightening": str(g.get("g3_tightening") or "mild"),
        "g4_classifier": str(g.get("g4_classifier") or "balanced"),
        "g5_routing": str(g.get("g5_recovery_routing") or "trigger"),
        "g6_near_reward": bool(g.get("g6_enforce_near_reward", True)),
        "g7_artifact": str(g.get("g7_artifact") or "5bullets"),
        "g8_replan_threshold": _to_int(g.get("g8_replan_threshold"), 1),
        "g9_pause_policy": str(g.get("g9_pause_policy") or "soft"),
        "g10_explore": str(g.get("g10_explore") or "off"),
    }
    for k in ("conservative", "balanced", "aggressive"):
        if (g.get("g1_block_map") or {}) == _preset_g1(k):
            gp["g1_profile"] = k
            break
    for k in ("dense", "balanced", "sparse"):
        if (g.get("g2_check_map") or {}) == _preset_g2(k):
            gp["g2_profile"] = k
            break
    return gp

def _genome_from_genotype(genotype: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base or default_genome())
    out["g1_block_map"] = _preset_g1(str(genotype.get("g1_profile") or "balanced"))
    out["g2_check_map"] = _preset_g2(str(genotype.get("g2_profile") or "balanced"))
    out["g3_tightening"] = str(genotype.get("g3_tightening") or out.get("g3_tightening") or "mild")
    out["g4_classifier"] = str(genotype.get("g4_classifier") or out.get("g4_classifier") or "balanced")
    out["g5_recovery_routing"] = str(genotype.get("g5_routing") or out.get("g5_recovery_routing") or "trigger")
    out["g6_enforce_near_reward"] = bool(genotype.get("g6_near_reward", out.get("g6_enforce_near_reward", True)))
    out["g7_artifact"] = str(genotype.get("g7_artifact") or out.get("g7_artifact") or "5bullets")
    out["g8_replan_threshold"] = _to_int(genotype.get("g8_replan_threshold"), _to_int(out.get("g8_replan_threshold"), 1))
    out["g9_pause_policy"] = str(genotype.get("g9_pause_policy") or out.get("g9_pause_policy") or "soft")
    out["g10_explore"] = str(genotype.get("g10_explore") or out.get("g10_explore") or "off")
    return out

def _nk_mutation_neighbors(genotype: Dict[str, Any]) -> List[Dict[str, Any]]:
    space = _nk_gene_space()
    out: List[Dict[str, Any]] = []
    for gk, alleles in space.items():
        cur = genotype.get(gk)
        vals = list(alleles)
        if cur not in vals:
            vals.append(cur)
        for a in vals:
            if a == cur:
                continue
            ng = dict(genotype)
            ng[gk] = a
            out.append(ng)
    return out

def _nk_pair_key(a: str, b: str) -> str:
    return "|".join(sorted([a, b]))

def _nk_pair_val(ga: str, va: Any, gb: str, vb: Any) -> str:
    if ga <= gb:
        return f"{va}|{vb}"
    return f"{vb}|{va}"

def _nk_estimate(st: ProkState, genotype: Dict[str, Any]) -> float:
    nk = st.nk_state or default_nk_state()
    main = nk.get("main") or {}
    pair = nk.get("pair") or {}
    neighbors = nk.get("neighbors") or default_nk_state()["neighbors"]
    genes = list(_nk_gene_space().keys())

    terms: List[float] = []
    for gi in genes:
        vi = genotype.get(gi)
        mstat = ((main.get(gi) or {}).get(str(vi)) or {})
        mmean = _safe_rate(_to_int(mstat.get("sum"), 0), _to_int(mstat.get("n"), 0))
        if mmean is None:
            mmean = 0.0
        pvals: List[float] = []
        for gj in neighbors.get(gi, [])[:2]:
            vj = genotype.get(gj)
            pk = _nk_pair_key(gi, gj)
            pv = _nk_pair_val(gi, vi, gj, vj)
            pstat = ((pair.get(pk) or {}).get(pv) or {})
            pmean = _safe_rate(_to_int(pstat.get("sum"), 0), _to_int(pstat.get("n"), 0))
            if pmean is not None:
                pvals.append(pmean)
        pair_mean = _safe_avg(pvals) if pvals else 0.0
        terms.append(0.7 * mmean + 0.3 * pair_mean)
    return _safe_avg(terms) or 0.0

def nk_select_genome(st: ProkState) -> None:
    nk = st.nk_state or default_nk_state()
    if not bool(nk.get("enabled", True)):
        nk["last_source"] = "disabled"
        st.nk_state = nk
        return
    base_geno = _genotype_from_genome(st.genome)
    candidates = [base_geno] + _nk_mutation_neighbors(base_geno)
    scored = [(g, _nk_estimate(st, g)) for g in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)

    explore = str(st.genome.get("g10_explore") or "off").lower()
    eps = {"off": 0.0, "low": 0.10, "med": 0.20, "high": 0.35}.get(explore, 0.0)
    if len(scored) > 1 and random.random() < eps:
        chosen = random.choice(scored[1:])[0]
        source = "explore"
    else:
        chosen = scored[0][0]
        source = "exploit"
    st.genome = _genome_from_genotype(chosen, st.genome)
    nk["last_genotype"] = chosen
    nk["last_source"] = source
    nk["last_pred_score"] = round(_nk_estimate(st, chosen), 4)
    st.nk_state = nk

def nk_record_outcome(
    st: ProkState,
    planned_minutes: int,
    minutes_worked: int,
    drift_events: int,
    check_count: int,
    quit_early: bool,
    task: Task,
) -> None:
    nk = st.nk_state or default_nk_state()
    geno = nk.get("last_genotype") or _genotype_from_genome(st.genome)
    main = nk.setdefault("main", {})
    pair = nk.setdefault("pair", {})
    neighbors = nk.get("neighbors") or default_nk_state()["neighbors"]
    genes = list(_nk_gene_space().keys())

    feasible = 1.0 if (task.deadline is None or task_slack_blocks(st, task) >= 0) else 0.0
    progress = float(minutes_worked) / float(max(1, planned_minutes))
    disturbance = min(1.0, float(drift_events) / 4.0)
    checks = min(1.0, float(check_count) / 10.0)
    quit_pen = 1.0 if quit_early else 0.0
    utility = 1.0 * feasible + 0.8 * progress - 0.5 * disturbance - 0.2 * checks - 0.8 * quit_pen
    u_int = int(round(utility * 100))

    for gk in genes:
        gv = str(geno.get(gk))
        gstat = (main.setdefault(gk, {})).setdefault(gv, {"sum": 0, "n": 0})
        gstat["sum"] = _to_int(gstat.get("sum"), 0) + u_int
        gstat["n"] = _to_int(gstat.get("n"), 0) + 1

    for gk in genes:
        for gj in neighbors.get(gk, [])[:2]:
            if gk >= gj:
                continue
            pk = _nk_pair_key(gk, gj)
            pv = _nk_pair_val(gk, geno.get(gk), gj, geno.get(gj))
            pstat = (pair.setdefault(pk, {})).setdefault(pv, {"sum": 0, "n": 0})
            pstat["sum"] = _to_int(pstat.get("sum"), 0) + u_int
            pstat["n"] = _to_int(pstat.get("n"), 0) + 1

    hist = nk.setdefault("history", [])
    hist.append({
        "ts": _now_iso(),
        "utility": round(utility, 4),
        "u_int": u_int,
        "genotype": geno,
        "source": nk.get("last_source") or "unknown",
    })
    if len(hist) > 200:
        nk["history"] = hist[-200:]
    st.nk_state = nk

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
    homeostat_program: Optional[HomeostatProgram],
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
    check_timestamps: List[float] = []

    def can_fire_check() -> Tuple[bool, str]:
        if in_quiet_window(st):
            return False, "quiet_window"
        max_per_hour = _to_int((st.policy_constraints or {}).get("max_checks_per_hour"), 10)
        if max_per_hour <= 0:
            return False, "max_checks_per_hour=0"
        now = time.time()
        recent = [t for t in check_timestamps if now - t <= 3600.0]
        check_timestamps[:] = recent
        if len(recent) >= max_per_hour:
            return False, "max_checks_per_hour"
        return True, "ok"

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
        "nk_source": (st.nk_state or {}).get("last_source"),
        "nk_pred_score": (st.nk_state or {}).get("last_pred_score"),
        "nk_genotype": (st.nk_state or {}).get("last_genotype"),
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
        ok, why = can_fire_check()
        if not ok:
            print(f"\nCHECK skipped by policy constraint ({why}).")
            return ""
        check_timestamps.append(time.time())
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
        sel = select_recovery_with_trace(
            st=st,
            task=task,
            label=label,
            trigger=trigger,
            check_gaps=(min_gap, max_gap),
            fitness_program=fitness_program,
            homeostat_program=homeostat_program,
        )
        best = sel["best"]
        selection_source = sel["selection_source"]
        phenom_mode = sel["phenom_mode"]
        phenom_score = sel["phenom_score"]
        phenom_components = sel["phenom_components"]
        baseline_name = ((sel.get("trace_rows") or [{}])[0]).get("name")
        reason_codes: List[str] = ["homeostat_rank", "fitness_rank"]
        if selection_source.endswith("+phenomenology"):
            reason_codes.append("phenomenology_override")
        if phenom_mode == "dag":
            reason_codes.append("dag_mode")

        if selection_source == "homeostat+fitness+phenomenology":
            print(
                f"Phenomenology match: using {best['name']} "
                f"(mode={phenom_mode}, score={_to_float(phenom_score, 0.0):.2f}, "
                f"phi={_to_float((phenom_components or {}).get('phi'), 0.0):.2f}, "
                f"amb={_to_float((phenom_components or {}).get('ambiguity'), 0.0):.2f}, "
                f"epi={_to_float((phenom_components or {}).get('epistemic'), 0.0):.2f}, "
                f"v={_to_float((phenom_components or {}).get('p_viable'), 0.0):.2f}, "
                f"p={_to_float((phenom_components or {}).get('p_procrast'), 0.0):.2f}, "
                f"o={_to_float((phenom_components or {}).get('p_overcontrol'), 0.0):.2f}, "
                f"dag={_to_float((phenom_components or {}).get('dag_causal'), 0.0):.2f})."
            )
        dag_notes = (phenom_components or {}).get("dag_notes") if isinstance(phenom_components, dict) else None
        if dag_notes:
            for note in dag_notes[:2]:
                print(f"DAG note: {note}")

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
            "decision_reason_codes": reason_codes,
            "decision_trace": sel.get("trace_rows"),
            "decision_baseline": baseline_name,
            "phenomenology_signature": signature,
            "homeostat_score": best.get("homeostat_score"),
            "homeostat_penalty": best.get("homeostat_penalty"),
            "homeostat_bonus": best.get("homeostat_bonus"),
            "homeostat_ctx": best.get("ctx"),
            "phenom_mode": phenom_mode,
            "phenom_score": phenom_score,
            "phenom_components": phenom_components,
            "gaps": [min_gap, max_gap] if random_checks else None,
            **ratings,
        })
        phenomenology_record_outcome(st, signature, best["name"], ratings, label=label)

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

    nk_record_outcome(
        st=st,
        planned_minutes=block_minutes,
        minutes_worked=minutes_worked,
        drift_events=drift_events,
        check_count=check_count,
        quit_early=quit_early,
        task=task,
    )
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

def circadian_bin(now: Optional[dt.datetime] = None) -> str:
    x = now or dt.datetime.now()
    h = x.hour
    if 5 <= h < 11:
        return "morning"
    if 11 <= h < 17:
        return "afternoon"
    if 17 <= h < 22:
        return "evening"
    return "night"

def _parse_window(s: str) -> Tuple[int, int]:
    m = re.fullmatch(r"\s*(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})\s*", s or "")
    if not m:
        raise ValueError(f"Invalid window '{s}'. Use HH:MM-HH:MM")
    a = _parse_hhmm(m.group(1))
    b = _parse_hhmm(m.group(2))
    return a, b

def in_quiet_window(st: ProkState, now: Optional[dt.datetime] = None) -> bool:
    x = now or dt.datetime.now()
    mnow = x.hour * 60 + x.minute
    for raw in (st.policy_constraints or {}).get("quiet_windows") or []:
        try:
            a, b = _parse_window(str(raw))
        except Exception:
            continue
        if a <= b:
            if a <= mnow <= b:
                return True
        else:
            if mnow >= a or mnow <= b:
                return True
    return False

def deadline_risk(st: ProkState, task: Task) -> Tuple[float, str, List[str]]:
    if not task.deadline:
        return 0.0, "none", []
    slack = task_slack_blocks(st, task)
    reasons: List[str] = []
    score = 0.0
    if slack < 0:
        score += 2.0; reasons.append("negative_slack")
    elif slack == 0:
        score += 1.2; reasons.append("zero_slack")
    elif slack <= 2:
        score += 0.8; reasons.append("low_slack")
    if task.recent_drift_count >= 3:
        score += 0.8; reasons.append("drift_cluster")
    elif task.recent_drift_count >= 1:
        score += 0.3; reasons.append("recent_drift")
    try:
        days_left = max(0, (dt.date.fromisoformat(task.deadline) - _today()).days)
    except Exception:
        days_left = 0
    if days_left <= 3 and task.remaining_minutes > 0:
        score += 0.7; reasons.append("deadline_near")
    p = 1.0 / (1.0 + math.exp(-(score - 1.0)))
    if p >= 0.67:
        band = "high"
    elif p >= 0.34:
        band = "med"
    else:
        band = "low"
    return p, band, reasons

def _ics_unfold_lines(text: str) -> List[str]:
    out: List[str] = []
    for raw in text.splitlines():
        if not out:
            out.append(raw.rstrip("\r"))
            continue
        if raw.startswith(" ") or raw.startswith("\t"):
            out[-1] += raw[1:].rstrip("\r")
        else:
            out.append(raw.rstrip("\r"))
    return out

def _ics_parse_value(raw: str) -> Tuple[Optional[dt.datetime], Optional[dt.date], bool]:
    s = raw.strip()
    if re.fullmatch(r"\d{8}", s):
        d = dt.datetime.strptime(s, "%Y%m%d").date()
        return None, d, True
    if re.fullmatch(r"\d{8}T\d{6}Z?", s):
        z = s.endswith("Z")
        core = s[:-1] if z else s
        x = dt.datetime.strptime(core, "%Y%m%dT%H%M%S")
        return x, None, False
    if re.fullmatch(r"\d{8}T\d{4}Z?", s):
        z = s.endswith("Z")
        core = s[:-1] if z else s
        x = dt.datetime.strptime(core, "%Y%m%dT%H%M")
        return x, None, False
    return None, None, False

def _ics_event_map(lines: List[str]) -> Dict[str, Tuple[Dict[str, str], str]]:
    out: Dict[str, Tuple[Dict[str, str], str]] = {}
    for ln in lines:
        if ":" not in ln:
            continue
        k, v = ln.split(":", 1)
        parts = k.split(";")
        name = parts[0].strip().upper()
        params: Dict[str, str] = {}
        for p0 in parts[1:]:
            if "=" in p0:
                a, b = p0.split("=", 1)
                params[a.strip().upper()] = b.strip()
        out[name] = (params, v.strip())
    return out

def _rrule_days(rrule: str) -> List[int]:
    parts = {}
    for x in str(rrule).split(";"):
        if "=" in x:
            a, b = x.split("=", 1)
            parts[a.strip().upper()] = b.strip().upper()
    byday = parts.get("BYDAY")
    if not byday:
        return []
    m = {"MO": 0, "TU": 1, "WE": 2, "TH": 3, "FR": 4, "SA": 5, "SU": 6}
    out = []
    for d in byday.split(","):
        d = d.strip()
        if d in m:
            out.append(m[d])
    return out

def parse_ics_to_blackouts(path: Path) -> Dict[str, List[Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = _ics_unfold_lines(text)
    events: List[List[str]] = []
    cur: Optional[List[str]] = None
    for ln in lines:
        u = ln.strip().upper()
        if u == "BEGIN:VEVENT":
            cur = []
            continue
        if u == "END:VEVENT":
            if cur is not None:
                events.append(cur)
            cur = None
            continue
        if cur is not None:
            cur.append(ln)

    out_weekly: List[WeeklyBlackout] = []
    out_date: List[DateBlackout] = []
    out_range: List[DateRangeBlackout] = []
    out_dt: List[DateTimeBlackout] = []

    for ev in events:
        mp = _ics_event_map(ev)
        uid = (mp.get("UID", ({}, ""))[1] or "").strip() or f"ics-{random.randint(10000,99999)}"
        summary = (mp.get("SUMMARY", ({}, ""))[1] or "").strip() or "calendar"
        label = f"calendar:{summary}"

        ds = mp.get("DTSTART")
        de = mp.get("DTEND")
        if not ds:
            continue
        ds_params, ds_val = ds
        de_params, de_val = de if de else ({}, "")
        ds_dt, ds_date, ds_all = _ics_parse_value(ds_val)
        de_dt, de_date, de_all = _ics_parse_value(de_val) if de_val else (None, None, False)
        if ds_dt is None and ds_date is None:
            continue

        rrule = (mp.get("RRULE", ({}, ""))[1] or "").strip()
        is_weekly = ("FREQ=WEEKLY" in rrule.upper()) if rrule else False

        if is_weekly and ds_dt is not None:
            days = _rrule_days(rrule)
            if not days:
                days = [ds_dt.weekday()]
            end_dt = de_dt or (ds_dt + dt.timedelta(hours=1))
            s = ds_dt.strftime("%H:%M")
            e = end_dt.strftime("%H:%M")
            for wd in days:
                out_weekly.append(WeeklyBlackout(
                    weekday_start=wd,
                    weekday_end=wd,
                    start_hhmm=s,
                    end_hhmm=e,
                    label=label,
                    source="calendar",
                    uid=uid,
                ))
            continue

        if ds_all or (ds_params.get("VALUE", "").upper() == "DATE"):
            sdate = ds_date or (ds_dt.date() if ds_dt else None)
            if sdate is None:
                continue
            # iCal all-day DTEND is usually exclusive.
            if de_date:
                edate = de_date - dt.timedelta(days=1)
            else:
                edate = sdate
            if edate <= sdate:
                out_date.append(DateBlackout(date=sdate.isoformat(), label=label, source="calendar", uid=uid))
            else:
                out_range.append(DateRangeBlackout(start=sdate.isoformat(), end=edate.isoformat(), label=label, source="calendar", uid=uid))
            continue

        sdt = ds_dt or dt.datetime.combine(ds_date, dt.time(0, 0))
        edt = de_dt or (sdt + dt.timedelta(hours=1))
        if edt.date() != sdt.date():
            out_date.append(DateBlackout(date=sdt.date().isoformat(), label=label, source="calendar", uid=uid))
            continue
        out_dt.append(DateTimeBlackout(
            date=sdt.date().isoformat(),
            start_hhmm=sdt.strftime("%H:%M"),
            end_hhmm=edt.strftime("%H:%M"),
            label=label,
            source="calendar",
            uid=uid,
        ))

    return {
        "weekly": out_weekly,
        "date": out_date,
        "range": out_range,
        "datetime": out_dt,
    }

def _drop_calendar_blackouts(st: ProkState) -> None:
    st.weekly_blackouts = [b for b in st.weekly_blackouts if str(getattr(b, "source", "manual")) != "calendar"]
    st.date_blackouts = [b for b in st.date_blackouts if str(getattr(b, "source", "manual")) != "calendar"]
    st.daterange_blackouts = [b for b in st.daterange_blackouts if str(getattr(b, "source", "manual")) != "calendar"]
    st.datetime_blackouts = [b for b in st.datetime_blackouts if str(getattr(b, "source", "manual")) != "calendar"]

def cmd_blackout(args) -> None:
    st = load_state(); ensure_today(st); _init_defaults(st)
    if args.action == "list":
        print("Weekly blackouts:")
        for b in st.weekly_blackouts:
            print(f"  {b.weekday_start}-{b.weekday_end} {b.start_hhmm}-{b.end_hhmm} {b.label} [{b.source}]".strip())
        print("Date blackouts:")
        for b in st.date_blackouts:
            print(f"  {b.date} {b.label} [{b.source}]".strip())
        print("Date-range blackouts:")
        for b in st.daterange_blackouts:
            print(f"  {b.start}..{b.end} {b.label} [{b.source}]".strip())
        print("Date-time blackouts:")
        for b in st.datetime_blackouts:
            print(f"  {b.date} {b.start_hhmm}-{b.end_hhmm} {b.label} [{b.source}]".strip())
        return
    if args.action == "clear":
        st.weekly_blackouts = []; st.date_blackouts = []; st.daterange_blackouts = []; st.datetime_blackouts = []
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

def cmd_calendar(args) -> None:
    st = load_state(); ensure_today(st); _init_defaults(st)
    if args.action == "import":
        p = Path(args.ics)
        if not p.exists():
            raise SystemExit(f"ICS file not found: {p}")
        parsed = parse_ics_to_blackouts(p)
        if args.replace_all:
            st.weekly_blackouts = []
            st.date_blackouts = []
            st.daterange_blackouts = []
            st.datetime_blackouts = []
        else:
            _drop_calendar_blackouts(st)
        st.weekly_blackouts.extend(parsed["weekly"])
        st.date_blackouts.extend(parsed["date"])
        st.daterange_blackouts.extend(parsed["range"])
        st.datetime_blackouts.extend(parsed["datetime"])
        save_state(st)
        _log_event("calendar_import", {
            "participant_id": st.participant_id,
            "path": str(p),
            "weekly_n": len(parsed["weekly"]),
            "date_n": len(parsed["date"]),
            "range_n": len(parsed["range"]),
            "datetime_n": len(parsed["datetime"]),
            "replace_all": bool(args.replace_all),
        })
        print(
            "Imported calendar blackouts from "
            f"{p}: weekly={len(parsed['weekly'])}, date={len(parsed['date'])}, "
            f"range={len(parsed['range'])}, datetime={len(parsed['datetime'])}"
        )
        return
    if args.action == "show":
        wk = [b for b in st.weekly_blackouts if b.source == "calendar"]
        dd = [b for b in st.date_blackouts if b.source == "calendar"]
        rr = [b for b in st.daterange_blackouts if b.source == "calendar"]
        dtb = [b for b in st.datetime_blackouts if b.source == "calendar"]
        print(f"Calendar blackouts: weekly={len(wk)} date={len(dd)} range={len(rr)} datetime={len(dtb)}")
        for b in dtb[:10]:
            print(f"  {b.date} {b.start_hhmm}-{b.end_hhmm} {b.label}")
        return
    if args.action == "clear":
        _drop_calendar_blackouts(st)
        save_state(st)
        _log_event("calendar_clear", {"participant_id": st.participant_id})
        print("Cleared imported calendar blackouts.")
        return

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
            risk_txt = "-"
            if t and t.deadline:
                rp, rb, _ = deadline_risk(st, t)
                risk_txt = f"{rb}({_fmt_num(rp)})"
            print(f"  {tid} | {name} | {blocks}")
            print(f"    risk: {risk_txt}")

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
    if args.homeostat:
        st.homeostat_path = args.homeostat
    nk_select_genome(st)
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
    hprogram = load_homeostat_program(st) if st.homeostat_path else None

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
        homeostat_program=hprogram,
    )

def cmd_debug(args) -> None:
    st = load_state(); ensure_today(st); _init_defaults(st)
    if args.action != "decision":
        raise SystemExit("Unknown debug action.")
    task = select_active_task(st, args.task, args.domain, args.name)
    label = str(args.label or ("PROCRASTINATION" if args.kind == "p" else "DRIFT")).upper()
    if label not in ("DRIFT", "PROCRASTINATION", "BLOCKED"):
        raise SystemExit("label must be one of DRIFT|PROCRASTINATION|BLOCKED")
    trigger = str(args.trigger or "other")
    fit = load_fitness_program(st) if st.fitness_path else None
    homeo = load_homeostat_program(st) if st.homeostat_path else None
    mn = _to_int(args.min_gap, 6)
    mx = _to_int(args.max_gap, 14)
    if mx < mn:
        mx = mn
    sel = select_recovery_with_trace(
        st=st,
        task=task,
        label=label,
        trigger=trigger,
        check_gaps=(mn, mx),
        fitness_program=fit,
        homeostat_program=homeo,
    )
    print(f"DEBUG decision | task={task.id} | label={label} | trigger={trigger}")
    print(f"selection_source={sel['selection_source']} | chosen={sel['best']['name']} | phenom_mode={sel['phenom_mode']}")
    pcs = sel.get("phenom_components") or {}
    if pcs:
        print(
            f"phenom: phi={_to_float(pcs.get('phi'), 0.0):.2f} "
            f"amb={_to_float(pcs.get('ambiguity'), 0.0):.2f} "
            f"epi={_to_float(pcs.get('epistemic'), 0.0):.2f} "
            f"dag={_to_float(pcs.get('dag_causal'), 0.0):.2f}"
        )
        notes = pcs.get("dag_notes") or []
        for n in notes[:3]:
            print(f"dag_note: {n}")
    print("candidates:")
    for i, row in enumerate(sel.get("trace_rows") or [], 1):
        print(
            f"  {i}. {row.get('name')} | homeostat={row.get('homeostat_score')} "
            f"(p={row.get('homeostat_penalty')},b={row.get('homeostat_bonus')}) "
            f"| fitness_rank={row.get('fitness_rank')}"
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
        risk_txt = "-"
        if t.deadline:
            rp, rb, _ = deadline_risk(st, t)
            risk_txt = f"{rb}({_fmt_num(rp)})"
        active = "*" if st.active_task_id == t.id else " "
        print(f"   {active} {t.id} | {t.domain} | {t.name} | deadline={t.deadline or '-'} | remaining={t.remaining_minutes}/{t.total_minutes} | slack={slack_txt} | risk={risk_txt} | drift={t.recent_drift_count} | θ={asdict(t.tmt_bins)}")
    print(f"  genome.explore: {st.genome.get('g10_explore')}")
    print(f"  fitness_path: {st.fitness_path or '(none)'}")
    print(f"  homeostat_path: {st.homeostat_path or '(none)'}")
    print(f"  goal_profile: {st.goal_profile}")
    ph = st.phenomenology or {}
    print(f"  phenomenology: enabled={bool(ph.get('enabled', True))} | signatures={len((ph.get('signatures') or {}))} | min_samples={_to_int(ph.get('min_samples'), 3)} | influence={_to_float(ph.get('influence'), 0.35):.2f}")
    print(f"  use_cca: {st.use_cca} (v4 placeholder, not used)")
    nk = st.nk_state or {}
    hist = nk.get("history") or []
    print(
        "  nk: "
        f"enabled={bool(nk.get('enabled', True))} "
        f"k={_to_int(nk.get('k'), 2)} "
        f"source={nk.get('last_source') or 'none'} "
        f"pred={nk.get('last_pred_score')} "
        f"history={len(hist)}"
    )
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

def cmd_policy(args) -> None:
    st = load_state(); ensure_today(st); _init_defaults(st)
    if args.action == "show":
        print(_json_dump(st.policy_constraints or {}))
        return
    if args.action == "set":
        pc = dict(st.policy_constraints or {})
        if args.max_checks_per_hour is not None:
            pc["max_checks_per_hour"] = max(0, _to_int(args.max_checks_per_hour, _to_int(pc.get("max_checks_per_hour"), 10)))
        if args.quiet_add:
            arr = list(pc.get("quiet_windows") or [])
            for w in args.quiet_add:
                _ = _parse_window(w)
                arr.append(w.strip())
            pc["quiet_windows"] = arr
        if args.quiet_clear:
            pc["quiet_windows"] = []
        st.policy_constraints = pc
        save_state(st)
        print("Policy constraints updated.")
        return

def cmd_simulate(args) -> None:
    st = load_state(); ensure_today(st); _init_defaults(st)
    mm = max(0, _to_int(args.minutes_today, 0))
    target = args.task
    if target and not get_task_by_id(st, target):
        raise SystemExit(f"Task not found: {target}")
    rows = []
    for t in st.tasks:
        if not t.deadline:
            continue
        cur_slack = task_slack_blocks(st, t)
        rem2 = max(0, t.remaining_minutes - (mm if (target is None or t.id == target) else 0))
        need2 = int(math.ceil(max(0, rem2) / max(1, st.plan_block_min)))
        cap2 = task_capacity_blocks(st, t)
        slack2 = cap2 - need2
        p0, b0, _ = deadline_risk(st, t)
        t2 = Task(
            id=t.id, domain=t.domain, name=t.name, deadline=t.deadline,
            total_minutes=t.total_minutes, remaining_minutes=rem2,
            tmt_bins=t.tmt_bins, recent_drift_count=t.recent_drift_count,
        )
        p1, b1, _ = deadline_risk(st, t2)
        rows.append((t.id, t.name, cur_slack, slack2, p0, b0, p1, b1))
    print(f"COUNTERFACTUAL: log {mm} minute(s) today " + (f"to task={target}" if target else "across all tasks with deadlines"))
    for r in rows:
        print(
            f"{r[0]} | {r[1]} | slack {r[2]} -> {r[3]} | "
            f"risk {r[5]}({_fmt_num(r[4])}) -> {r[7]}({_fmt_num(r[6])})"
        )
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

def cmd_homeostat(args) -> None:
    st = load_state(); ensure_today(st); _init_defaults(st)
    if args.action == "show":
        print(st.homeostat_path or "(none)")
        return
    if args.action == "write-default":
        out = Path(args.out or "homeostat.default.prokfit")
        out.write_text(default_homeostat_text(), encoding="utf-8")
        print(f"Wrote default homeostat file to {out}")
        return
    if args.action == "set":
        p = Path(args.path)
        _ = parse_homeostat(p)
        st.homeostat_path = str(p)
        save_state(st)
        print(f"Set homeostat DSL: {p}")
        return

def cmd_nk(args) -> None:
    st = load_state(); ensure_today(st); _init_defaults(st)
    nk = st.nk_state or default_nk_state()
    if args.action == "show":
        history = nk.get("history") or []
        recent = history[-20:]
        utils = [float(h.get("utility")) for h in recent if isinstance(h.get("utility"), (int, float))]
        print(_json_dump({
            "enabled": bool(nk.get("enabled", True)),
            "k": _to_int(nk.get("k"), 2),
            "last_source": nk.get("last_source"),
            "last_pred_score": nk.get("last_pred_score"),
            "history_n": len(history),
            "recent_utility_avg": _safe_avg(utils),
            "last_genotype": nk.get("last_genotype") or {},
        }))
        return
    if args.action == "set":
        if args.enabled is not None:
            nk["enabled"] = (str(args.enabled).lower() == "on")
        if args.k is not None:
            nk["k"] = max(0, min(4, _to_int(args.k, _to_int(nk.get("k"), 2))))
        st.nk_state = nk
        save_state(st)
        print("NK settings updated.")
        return
    if args.action == "reset":
        st.nk_state = default_nk_state()
        save_state(st)
        print("NK memory reset.")
        return

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
            "mode": str(ph.get("mode") or "fep").lower(),
            "lambda_procrast": _to_float(ph.get("lambda_procrast"), 0.60),
            "mu_overcontrol": _to_float(ph.get("mu_overcontrol"), 0.25),
            "beta_ambiguity": _to_float(ph.get("beta_ambiguity"), 0.20),
            "eta_epistemic": _to_float(ph.get("eta_epistemic"), 0.10),
            "dag_kappa": _to_float(ph.get("dag_kappa"), 0.30),
            "dag_adjust": [str(x) for x in (ph.get("dag_adjust") or [])],
            "dag_min_samples": _to_int(ph.get("dag_min_samples"), 5),
            "dag_laplace": _to_float(ph.get("dag_laplace"), 1.0),
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
            best_comp: Dict[str, float] = {}
            for name, stats in actions.items():
                score, rated, comp = _phenomenology_action_score(
                    stats or {},
                    mode=str(ph.get("mode") or "fep").lower(),
                    lambda_procrast=_to_float(ph.get("lambda_procrast"), 0.60),
                    mu_overcontrol=_to_float(ph.get("mu_overcontrol"), 0.25),
                    beta_ambiguity=_to_float(ph.get("beta_ambiguity"), 0.20),
                    eta_epistemic=_to_float(ph.get("eta_epistemic"), 0.10),
                )
                if rated > 0 and score > best_score:
                    best_name = name
                    best_score = score
                    best_n = rated
                    best_comp = comp
            if best_name == "-":
                print(f"  {sig} | n={_to_int(entry.get('n'), 0)} | best=-")
            else:
                print(
                    f"  {sig} | n={_to_int(entry.get('n'), 0)} | best={best_name} "
                    f"score={best_score:.2f} rated={best_n} mode={best_comp.get('mode', 'fep')} "
                    f"(phi={best_comp.get('phi', 0.0):.2f}, amb={best_comp.get('ambiguity', 0.0):.2f}, epi={best_comp.get('epistemic', 0.0):.2f}, "
                    f"v={best_comp.get('p_viable', 0.0):.2f}, p={best_comp.get('p_procrast', 0.0):.2f}, o={best_comp.get('p_overcontrol', 0.0):.2f})"
                )
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
        if args.mode is not None:
            md = str(args.mode).lower()
            if md in ("phi", "fep", "dag"):
                ph["mode"] = md
        if args.min_samples is not None:
            ph["min_samples"] = max(1, _to_int(args.min_samples, _to_int(ph.get("min_samples"), 3)))
        if args.influence is not None:
            ph["influence"] = max(0.0, min(1.0, _to_float(args.influence, _to_float(ph.get("influence"), 0.35))))
        if args.lambda_procrast is not None:
            ph["lambda_procrast"] = max(0.0, min(2.0, _to_float(args.lambda_procrast, _to_float(ph.get("lambda_procrast"), 0.60))))
        if args.mu_overcontrol is not None:
            ph["mu_overcontrol"] = max(0.0, min(2.0, _to_float(args.mu_overcontrol, _to_float(ph.get("mu_overcontrol"), 0.25))))
        if args.beta_ambiguity is not None:
            ph["beta_ambiguity"] = max(0.0, min(2.0, _to_float(args.beta_ambiguity, _to_float(ph.get("beta_ambiguity"), 0.20))))
        if args.eta_epistemic is not None:
            ph["eta_epistemic"] = max(0.0, min(2.0, _to_float(args.eta_epistemic, _to_float(ph.get("eta_epistemic"), 0.10))))
        if args.dag_kappa is not None:
            ph["dag_kappa"] = max(0.0, min(2.0, _to_float(args.dag_kappa, _to_float(ph.get("dag_kappa"), 0.30))))
        if args.dag_min_samples is not None:
            ph["dag_min_samples"] = max(1, _to_int(args.dag_min_samples, _to_int(ph.get("dag_min_samples"), 5)))
        if args.dag_laplace is not None:
            ph["dag_laplace"] = max(0.0, min(10.0, _to_float(args.dag_laplace, _to_float(ph.get("dag_laplace"), 1.0))))
        if args.dag_adjust is not None:
            raw = [x.strip() for x in str(args.dag_adjust).split(",") if x.strip()]
            if raw:
                ph["dag_adjust"] = raw
        st.phenomenology = ph
        save_state(st)
        print("Phenomenology settings updated.")
        return

    if args.action == "reset":
        keep = {
            "enabled": bool(ph.get("enabled", True)),
            "min_samples": max(1, _to_int(ph.get("min_samples"), 3)),
            "influence": max(0.0, min(1.0, _to_float(ph.get("influence"), 0.35))),
            "mode": str(ph.get("mode") or "fep").lower(),
            "lambda_procrast": max(0.0, min(2.0, _to_float(ph.get("lambda_procrast"), 0.60))),
            "mu_overcontrol": max(0.0, min(2.0, _to_float(ph.get("mu_overcontrol"), 0.25))),
            "beta_ambiguity": max(0.0, min(2.0, _to_float(ph.get("beta_ambiguity"), 0.20))),
            "eta_epistemic": max(0.0, min(2.0, _to_float(ph.get("eta_epistemic"), 0.10))),
            "dag_kappa": max(0.0, min(2.0, _to_float(ph.get("dag_kappa"), 0.30))),
            "dag_adjust": [str(x) for x in (ph.get("dag_adjust") or ["domain", "label", "trigger", "slack", "drift"])],
            "dag_min_samples": max(1, _to_int(ph.get("dag_min_samples"), 5)),
            "dag_laplace": max(0.0, min(10.0, _to_float(ph.get("dag_laplace"), 1.0))),
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
    lam = _to_float(ph.get("lambda_procrast"), 0.60)
    mu = _to_float(ph.get("mu_overcontrol"), 0.25)
    print(f"- Default asymmetry: penalty(procrastination) > penalty(annoyance) [lambda={lam:.2f}, mu={mu:.2f}],")
    print("  with feasibility guards dominant (slack violations are critical).")
    print("")
    print("CURRENT RUNTIME GUARDRAILS:")
    print(f"- pause_policy: {st.genome.get('g9_pause_policy')}")
    print(f"- replan_threshold: {st.genome.get('g8_replan_threshold')}")
    print(f"- study_mode: {st.study_mode} (burden/helped ratings when enabled)")
    print(f"- phenomenology.enabled: {bool(ph.get('enabled', True))}")
    print(f"- phenomenology.mode: {str(ph.get('mode') or 'fep').lower()}")
    print(f"- phenomenology.min_samples: {_to_int(ph.get('min_samples'), 3)}")
    print(f"- phenomenology.influence: {_to_float(ph.get('influence'), 0.35):.2f}")
    print(f"- phenomenology.lambda_procrast: {lam:.2f}")
    print(f"- phenomenology.mu_overcontrol: {mu:.2f}")
    print(f"- phenomenology.beta_ambiguity: {_to_float(ph.get('beta_ambiguity'), 0.20):.2f}")
    print(f"- phenomenology.eta_epistemic: {_to_float(ph.get('eta_epistemic'), 0.10):.2f}")
    print(f"- use_cca: {st.use_cca} (placeholder only)")

def cmd_phase(args) -> None:
    st = load_state(); ensure_today(st); _init_defaults(st)
    records = _load_log_records(LOG_FILE)
    level, reasons = compute_alarm(st)

    active = get_task_by_id(st, st.active_task_id) if st.active_task_id else None
    tasks_with_deadline = [t for t in st.tasks if t.deadline]
    slack_map = {t.id: task_slack_blocks(st, t) for t in tasks_with_deadline}
    min_slack = min(slack_map.values()) if slack_map else 0
    active_slack = task_slack_blocks(st, active) if (active and active.deadline) else None

    drift_events = [r for r in records if str(r.get("kind")) in ("course_correct", "d", "p")]
    recent = drift_events[-20:]
    procrast_recent = sum(1 for r in recent if str(r.get("label") or "").upper() == "PROCRASTINATION")
    drift_recent = sum(1 for r in recent if str(r.get("label") or "").upper() == "DRIFT")
    burden_vals = [float(r.get("burden")) for r in recent if isinstance(r.get("burden"), (int, float))]
    burden_avg = _safe_avg(burden_vals)

    # Direction proxy: compare last 10 vs previous 10 disturbance burden.
    tail = drift_events[-10:]
    prev = drift_events[-20:-10]
    tail_p = sum(1 for r in tail if str(r.get("label") or "").upper() == "PROCRASTINATION")
    prev_p = sum(1 for r in prev if str(r.get("label") or "").upper() == "PROCRASTINATION")
    if len(tail) >= 3 and len(prev) >= 3:
        if tail_p < prev_p:
            direction = "toward_viable_attractor"
        elif tail_p > prev_p:
            direction = "away_from_viable_attractor"
        else:
            direction = "neutral"
    else:
        direction = "unknown_insufficient_history"

    ph = st.phenomenology or {}
    if level == "critical":
        controller_mode = "structural"
    elif level == "warning":
        controller_mode = "corrective"
    else:
        controller_mode = "routine"

    phase_vec = {
        "time": _iso_today(),
        "active_task_id": active.id if active else None,
        "active_domain": active.domain if active else None,
        "slack_active_blocks": active_slack,
        "slack_min_blocks": int(min_slack),
        "task_drift_active": int(active.recent_drift_count) if active else None,
        "domain_drift_ema_active": (
            _to_float(get_domain_state(st, active.domain).drift_ema, 0.0) if active else None
        ),
        "procrastination_recent_20": int(procrast_recent),
        "drift_recent_20": int(drift_recent),
        "burden_recent_avg": None if burden_avg is None else round(burden_avg, 2),
        "alarm": level,
        "controller_mode": controller_mode,
        "phenom_mode": str(ph.get("mode") or "fep").lower(),
    }
    print(_json_dump(phase_vec))

    print("Phase interpretation:")
    print(f"- Distance to viability boundary (blocks): {min_slack}")
    print(f"- Attractor direction: {direction}")
    if reasons:
        print("- Active constraints:")
        for r in reasons[:3]:
            print(f"  - {r}")

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

    scal = sub.add_parser("calendar")
    scalsub = scal.add_subparsers(dest="action", required=True)
    scal_import = scalsub.add_parser("import")
    scal_import.add_argument("--ics", required=True, help="Path to .ics calendar export")
    scal_import.add_argument("--replace-all", action="store_true", help="Replace all existing blackouts (manual + calendar)")
    scal_import.set_defaults(func=cmd_calendar)
    scalsub.add_parser("show").set_defaults(func=cmd_calendar)
    scalsub.add_parser("clear").set_defaults(func=cmd_calendar)

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
    sr.add_argument("--homeostat", default=None)
    sr.set_defaults(func=cmd_run)

    sdbg = sub.add_parser("debug")
    sdbg_sub = sdbg.add_subparsers(dest="action", required=True)
    sdec = sdbg_sub.add_parser("decision")
    sdec.add_argument("--task", default=None)
    sdec.add_argument("--domain", default=None)
    sdec.add_argument("--name", default=None)
    sdec.add_argument("--kind", choices=["d", "p"], default="d")
    sdec.add_argument("--label", default=None, help="DRIFT|PROCRASTINATION|BLOCKED (optional)")
    sdec.add_argument("--trigger", default="other")
    sdec.add_argument("--min-gap", type=int, default=6)
    sdec.add_argument("--max-gap", type=int, default=14)
    sdec.set_defaults(func=cmd_debug)

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

    spol = sub.add_parser("policy")
    spolsub = spol.add_subparsers(dest="action", required=True)
    spolsub.add_parser("show").set_defaults(func=cmd_policy)
    spol_set = spolsub.add_parser("set")
    spol_set.add_argument("--max-checks-per-hour", type=int, default=None)
    spol_set.add_argument("--quiet-add", action="append", default=None, help="Add quiet window HH:MM-HH:MM (repeatable)")
    spol_set.add_argument("--quiet-clear", action="store_true")
    spol_set.set_defaults(func=cmd_policy)

    ssim = sub.add_parser("simulate")
    ssim.add_argument("--minutes-today", type=int, required=True)
    ssim.add_argument("--task", default=None, help="Optional task_id; if omitted applies minutes to all deadline tasks")
    ssim.set_defaults(func=cmd_simulate)

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

    sh = sub.add_parser("homeostat")
    shsub = sh.add_subparsers(dest="action", required=True)
    shsub.add_parser("show").set_defaults(func=cmd_homeostat)
    sh_wd = shsub.add_parser("write-default"); sh_wd.add_argument("--out", default=None); sh_wd.set_defaults(func=cmd_homeostat)
    sh_set = shsub.add_parser("set"); sh_set.add_argument("path"); sh_set.set_defaults(func=cmd_homeostat)

    snk = sub.add_parser("nk")
    snksub = snk.add_subparsers(dest="action", required=True)
    snksub.add_parser("show").set_defaults(func=cmd_nk)
    snk_set = snksub.add_parser("set")
    snk_set.add_argument("--enabled", choices=["on", "off"], default=None)
    snk_set.add_argument("--k", type=int, default=None)
    snk_set.set_defaults(func=cmd_nk)
    snksub.add_parser("reset").set_defaults(func=cmd_nk)

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
    sph_set.add_argument("--mode", choices=["phi", "fep", "dag"], default=None)
    sph_set.add_argument("--min-samples", type=int, default=None)
    sph_set.add_argument("--influence", type=float, default=None)
    sph_set.add_argument("--lambda-procrast", type=float, default=None)
    sph_set.add_argument("--mu-overcontrol", type=float, default=None)
    sph_set.add_argument("--beta-ambiguity", type=float, default=None)
    sph_set.add_argument("--eta-epistemic", type=float, default=None)
    sph_set.add_argument("--dag-kappa", type=float, default=None)
    sph_set.add_argument("--dag-adjust", default=None, help="comma-separated covariates for DAG adjustment")
    sph_set.add_argument("--dag-min-samples", type=int, default=None)
    sph_set.add_argument("--dag-laplace", type=float, default=None)
    sph_set.set_defaults(func=cmd_phenom)

    sexp = sub.add_parser("export")
    sexp.add_argument("--out", default=None)
    sexp.set_defaults(func=cmd_export)

    sphase = sub.add_parser("phase")
    sphase_sub = sphase.add_subparsers(dest="action", required=True)
    sphase_sub.add_parser("show").set_defaults(func=cmd_phase)

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
