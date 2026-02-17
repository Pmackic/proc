# PROK v5 (multi-task, study-ready CLI)

PROK v5 is a **local-first procrastination regulator**: you define tasks and deadlines, it builds a feasible block plan, runs focus sessions with **random check-ins + instant self-report**, and **course-corrects** when you drift or procrastinate.

What’s new in v5:
- **Multi-task planning** with per-task slack and active task targeting
- **Per-domain recursion signals** (drift EMA) and **task-level recovery stats**
- **Algedonic alarm** (`ok | warning | critical`) that flags structural infeasibility and recommends fixes
- **Homeostat engine** (`require/penalty/prefer`) that safety-filters and scores recovery candidates
- **NK slow loop** that adapts genome choices across sessions from outcome feedback

Python stdlib only. Target: Python 3.10+.

See the detailed spec: `prok_spec_v5.md` and `PROK_v5_methodology.md`.

---

## Quick start

Run from an empty folder (recommended: one folder per user/study):

```bash
python prok_v5.py task add --domain skola --name "bio ch3" --hours 20 --deadline 2mo
python prok_v5.py task add --domain work --name "report" --hours 6 --deadline 2026-03-01
python prok_v5.py calendar import --ics my_calendar.ics
python prok_v5.py task list
python prok_v5.py today
python prok_v5.py run --task <task_id> --checks auto
```

---

## Core commands

### `task`
Add, list, set active, edit, or remove tasks.

Add:
```bash
python prok_v5.py task add --domain skola --name "bio ch3" --hours 20 --deadline 2mo
```

List:
```bash
python prok_v5.py task list
```

Set active:
```bash
python prok_v5.py task set-active <task_id>
```

Edit/remove:
```bash
python prok_v5.py task edit <task_id> --hours 18 --deadline 2026-04-15
python prok_v5.py task remove <task_id>
```

`plan` is a compatibility alias for `task add`:
```bash
python prok_v5.py plan --domain skola --name "bio ch3" --total-hours 20 --deadline 2mo
```

### `today`
Shows what “good” looks like today: remaining work, capacity, slack, and a recommended run.

```bash
python prok_v5.py today
```

### `run`
Runs a focus session for the active task or a specific task.

Active task:
```bash
python prok_v5.py run --checks auto
```

Target a task:
```bash
python prok_v5.py run --task <task_id> --checks random --min-gap 6 --max-gap 14
```

Trigger + Haghbin prompts:
```bash
python prok_v5.py run --ask-trigger dp   # ask trigger on drift/procrastinate (default)
python prok_v5.py run --ask-trigger always
python prok_v5.py run --no-haghbin
```

#### During a run (how to interact)
Type a single letter **then press Enter**:

- `d` = drifting
- `p` = procrastinating
- `b` = blocked/pause
- `r` = resume
- `c` = force a check now
- `q` = quit the session

Notes:
- If the program is asking a question (e.g., a check prompt), it will wait for your reply.
- Random checks pop up when enabled; answer `e/d/p/b`.

### `log`
Log minutes directly to the active task or a specific task.

```bash
python prok_v5.py log --minutes 30
python prok_v5.py log --task <task_id> --minutes 45
```

### `status`
Shows global and per-task summaries, including alarm state.

```bash
python prok_v5.py status
```

### `policy`
Set policy constraints used during runs (check caps + quiet windows).

```bash
python prok_v5.py policy show
python prok_v5.py policy set --max-checks-per-hour 8 --quiet-add 13:00-14:00
python prok_v5.py policy set --quiet-clear
```

### `simulate`
Counterfactual quick check: estimate slack/risk change if you log minutes today.

```bash
python prok_v5.py simulate --minutes-today 60 --task <task_id>
python prok_v5.py simulate --minutes-today 30
```

### `debug decision`
Prints the recovery decision trace for a simulated disturbance without starting a session.

```bash
python prok_v5.py debug decision --task <task_id> --kind d --trigger distraction
```

---

## Constraints (blackouts)

Blackouts remove time from your available work window and affect feasibility.

```bash
python prok_v5.py blackout add --weekday Mon-Fri --start 08:00 --end 15:00 --label school
python prok_v5.py blackout add --date 2026-03-10 --label trip
python prok_v5.py blackout add --daterange 2026-04-01..2026-04-07 --label vacation

python prok_v5.py blackout list
python prok_v5.py blackout clear
```

## Calendar import (ICS)

Import calendar events as planning blackouts so `today`/replan/run capacity automatically respects them.

```bash
python prok_v5.py calendar import --ics my_calendar.ics
python prok_v5.py calendar show
python prok_v5.py calendar clear
```

Notes:
- one-off timed events become date-time blackouts
- all-day events become date/date-range blackouts
- weekly RRULE events (`FREQ=WEEKLY`) become weekly blackouts
- import replaces previous calendar-imported blackouts by default; use `--replace-all` to replace manual blackouts too

---

## Config (global planner/session knobs)

```bash
python prok_v5.py config show
python prok_v5.py config set --block-min 50 --tiny-min 10 --day-start 08:00 --day-end 22:00 --max-blocks-per-day 6
```

---

## Algedonic alarm

Alarm states:
- `ok`: feasibility and drift are within tolerances
- `warning`: elevated drift (task or domain) but not infeasible
- `critical`: infeasible slack or high drift cluster

When alarm is `critical`, the CLI prints structural fixes such as:
- increase `--max-blocks-per-day`
- widen the day window (`--day-start/--day-end`)
- split a task into smaller tasks
- edit task deadline/hours

---

## Study mode (for research)

Study mode adds lightweight ratings after each recovery action.

```bash
python prok_v5.py study id P001
python prok_v5.py study on
python prok_v5.py study show
python prok_v5.py study off
```

When study mode is ON, after each recovery action you’ll be asked:
- Burden: 1–5 (optional)
- Helped restart? y/n (optional)

These are logged to `.prok_log.jsonl`.

---

## Goals / fitness profile

The regulator uses a small goal model (ρ) that biases how it chooses recovery packages.

```bash
python prok_v5.py goal show
python prok_v5.py goal wizard
python prok_v5.py goal set goal_profile.json
python prok_v5.py goal clear
```

Built-in goal keywords:
- `feasible`
- `recover_fast`
- `low_annoyance`
- `progress`
- `streak_up`

---

## Language pack

Customize user-facing words without changing the internal model.

```bash
python prok_v5.py lang show
python prok_v5.py lang wizard
python prok_v5.py lang set language_pack.json
python prok_v5.py lang clear
```

---

## Genome + Fitness DSL (advanced)

```bash
python prok_v5.py genome show
python prok_v5.py genome write-default --out genome.default.json
python prok_v5.py genome load genome.default.json

python prok_v5.py fitness show
python prok_v5.py fitness write-default --out fitness.default.prokfit
python prok_v5.py fitness set fitness.default.prokfit
```

### Homeostat DSL (advanced)

Homeostat is applied before fitness ranking:
- `require` = hard veto (when alternatives exist)
- `penalty N:` = soft cost
- `prefer N:` = soft bonus

```bash
python prok_v5.py homeostat show
python prok_v5.py homeostat write-default --out homeostat.default.prokfit
python prok_v5.py homeostat set homeostat.default.prokfit
```

### NK loop (advanced)

NK adapts genome policy over time (exploit/explore, local memory tables):

```bash
python prok_v5.py nk show
python prok_v5.py nk set --enabled on --k 2
python prok_v5.py nk reset
```

---

## Mathematical phenomenology (new)

PROK can now learn **structural disturbance signatures** (domain + label + trigger + TMT bins) and reuse recoveries that worked on homologous cases.

Signature shape:
- `domain|label|trigger|E{E}V{V}D{D}G{G}` (example: `skola|DRIFT|distraction|EMVMDNGL`)

Inspect learned knowledge:
```bash
python prok_v5.py phenom show
```

Tune influence:
```bash
python prok_v5.py phenom set --enabled on --mode fep --min-samples 3 --influence 0.35
# optional asymmetric mean weights:
python prok_v5.py phenom set --lambda-procrast 0.60 --mu-overcontrol 0.25
# optional FEP uncertainty/exploration weights:
python prok_v5.py phenom set --beta-ambiguity 0.20 --eta-epistemic 0.10
# DAG causal mode (backdoor/IPW-style correction from logs):
python prok_v5.py phenom set --mode dag --dag-kappa 0.30 --dag-adjust domain,label,trigger,slack,drift --dag-min-samples 5 --dag-laplace 1.0
```

Clear learned signatures (keep settings):
```bash
python prok_v5.py phenom reset
```

Generate a Markdown audit from logs:
```bash
python prok_v5.py phenom audit --out PHENOM_AUDIT_EXAMPLE.md
# optional custom log file:
python prok_v5.py phenom audit --log some_run.jsonl --out audit.md
```

Selection behavior:
- baseline candidate ranking uses homeostat + fitness DSL + goals
- phenomenology can override within the top candidates only when enough evidence exists
- evidence gate is controlled by `--min-samples`
- override sensitivity is controlled by `--influence` (0..1)
- single-value transformation per signature/action:
  - `phi = p_viable - lambda_procrast*p_procrast - mu_overcontrol*p_overcontrol`
  - default asymmetry: `lambda_procrast > mu_overcontrol`
- in `--mode fep`, score adds uncertainty + learning terms:
  - `score = phi - beta_ambiguity*ambiguity + eta_epistemic*epistemic`
- in `--mode dag`, score adds causal correction:
  - estimate `E[helped | do(action)]` and `E[overcontrol | do(action)]` from logs with backdoor-adjusted IPW over configured covariates (`--dag-adjust`)
  - `score = phi + dag_kappa * ((do_helped - 0.5) - mu_overcontrol*do_overcontrol)`
  - still restricted to top-K baseline-safe candidates (homeostat envelope preserved)

Logs:
- `course_correct` events now include `phenomenology_signature`, `selection_source`, and `homeostat_ctx` (covariates for DAG mode)
- when DAG data are sparse, runtime prints `DAG note: ...` fallback warnings

---

## Tutorial (end-to-end)

Use this once in a fresh folder:

### 1) Create tasks

```bash
python prok_v5.py task add --domain skola --name "MAT" --hours 10 --deadline 2026-03-12
python prok_v5.py task add --domain work --name "report" --hours 6 --deadline 2026-03-01
python prok_v5.py task list
```

### 2) Configure planner window

```bash
python prok_v5.py config set --block-min 50 --tiny-min 10 --day-start 08:00 --day-end 22:00 --max-blocks-per-day 6
python prok_v5.py today
```

### 3) Enable homeostat + NK

```bash
python prok_v5.py homeostat write-default --out homeostat.default.prokfit
python prok_v5.py homeostat set homeostat.default.prokfit
python prok_v5.py nk set --enabled on --k 2
python prok_v5.py nk show
```

### 4) Run one focus block

```bash
python prok_v5.py run --task <task_id> --checks auto --ask-trigger dp
```

During run:
- type `d` or `p` + Enter to trigger course-correction
- type `c` + Enter to force a check
- type `q` + Enter to quit

### 5) Review adaptation

```bash
python prok_v5.py status
python prok_v5.py nk show
python prok_v5.py phenom show
```

### 6) Audit and export

```bash
python prok_v5.py phenom audit --out PHENOM_AUDIT_EXAMPLE.md
python prok_v5.py export --out prok_bundle.zip
```

---

## Data files and export

PROK writes files in the current directory:
- `.prok_state.json` — current plan + settings + study metadata
- `.prok_log.jsonl` — line-delimited JSON events

Export a shareable bundle:
```bash
python prok_v5.py export
# or choose a path:
python prok_v5.py export --out my_bundle.zip
```

---

## Migration from v4

If `.prok_state.json` contains v4 single-plan fields (`domain`, `deadline`, `total_minutes`, `remaining_minutes`), v5 auto-migrates to:
- `tasks = [one migrated task]`
- `active_task_id` set to that task

No migration script is required.

---

## CCA (placeholder only)

CCA integration exists as an interface only and is **NOT used** in v5.

```bash
python prok_v5.py cca
```

---

## Troubleshooting

**“Nothing happens when I type d/p.”**
- Make sure you press **Enter** after the letter.
- If you see a prompt waiting for input (e.g., CHECK), answer that prompt first.

**State/log location**
- If you want separate tasks or participants, use separate folders (each folder has its own `.prok_state.json` and `.prok_log.jsonl`).

---

## Ethics / scope note

This is a self-regulation prototype for research and personal use. It is not a diagnostic tool and does not guarantee control—its goal is **faster recovery and feasibility preservation**, not perfection.

Ethical stance (short):
- target is a sustainable engaged mean (flow-capable work), not maximal pressure
- control is asymmetric: procrastination risk is treated as more harmful than mild annoyance when feasibility is threatened
- user agency remains primary: pause/quit and language customization are always available

Show current doctrine + guardrails:
```bash
python prok_v5.py ethics show
```

Phase-space snapshot:
```bash
python prok_v5.py phase show
```
