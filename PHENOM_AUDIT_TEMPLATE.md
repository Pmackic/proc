# PROK v5 Phenomenology Audit Template

Use this template to report the phenomenology layer consistently across pilots, MRTs, and field evaluations.

Companion example:
- See `PHENOM_AUDIT_EXAMPLE.md` for a pre-filled legacy-baseline audit derived from local logs.
- Generate/update via CLI: `python prok_v5.py phenom audit --out PHENOM_AUDIT_EXAMPLE.md`

---

## 1) Study metadata

- Study ID:
- Dataset window (start/end):
- Participants (n):
- Sessions (n):
- Course-correct events (n):
- Phenomenology settings:
  - `enabled`:
  - `mode` (`phi` or `fep`):
  - `min_samples (n_min)`:
  - `influence (tau)`:
  - `lambda_procrast`:
  - `mu_overcontrol`:
  - `beta_ambiguity`:
  - `eta_epistemic`:

---

## 2) Core metrics (required)

### A. Coverage and sparsity

- Signature count `|Sigma|`:
- Events mapped to signatures:
- Coverage rate (`events with signature`) = ___ / ___ = ___
- Evidence-qualified coverage (`n(signature) >= n_min`) = ___ / ___ = ___
- Median events per signature:
- P90 events per signature:
- Gini (or concentration proxy) over signature counts:

Interpretation note:
- Low qualified coverage + high concentration implies signature sparsity/over-fragmentation.

### B. Override behavior

- Candidate decisions total:
- Baseline-only selections (`selection_source=fitness`):
- Phenomenology selections (`selection_source=fitness+phenomenology`):
- Override rate = ___ / ___ = ___
- Confidence-gated rejection count (recommended to log/derive):

### C. Outcome deltas

Primary:
- Recovery success rate (`helped==true`) by selection source
- Recovery burden mean by selection source
- Time-to-recover (if available) by selection source

Report deltas:
- `Delta_success = success(phenom) - success(fitness)`
- `Delta_burden = burden(phenom) - burden(fitness)` (negative preferred)
- `Delta_time = ttr(phenom) - ttr(fitness)` (negative preferred)

---

## 3) Required tables

### Table T1: Signature summary

Columns:
- `signature`
- `events_n`
- `qualified` (bool; `events_n >= n_min`)
- `best_action`
- `best_action_rated_n`
- `best_action_helped_rate`
- `best_action_burden_avg`

### Table T2: Action comparison by selection source

Columns:
- `selection_source` (`fitness` / `fitness+phenomenology`)
- `events_n`
- `helped_yes_n`
- `helped_rated_n`
- `helped_rate`
- `burden_n`
- `burden_avg`
- `time_to_recover_avg` (if available)

### Table T3: Trigger/domain stratification (recommended)

Columns:
- `domain`
- `trigger`
- `selection_source`
- `events_n`
- `helped_rate`
- `burden_avg`

---

## 4) Minimal statistical checks

Use mixed-effects models when sample size permits.

- Override acceptance model (logistic):
  - `override ~ signature_count + burden + trigger + (1|participant)`
- Recovery effectiveness model:
  - Logistic for helped: `helped ~ selection_source + trigger + domain + (1|participant)`
  - Linear/survival for recovery time: `ttr ~ selection_source + trigger + domain + (1|participant)`

At minimum, report:
- point estimate
- 95% CI
- p-value or equivalent interval-based decision

---

## 5) Ablation block (required for field reports)

Compare:
1) Policy layer only (fitness DSL + goals)
2) Policy + phenomenology layer

Report:
- Feasibility outcomes (slack violations)
- Recovery outcomes (success/time)
- Burden outcomes
- Adherence outcomes

---

## 6) Decision rubric

Mark each item as `PASS / WATCH / FAIL`.

- Coverage adequate (qualified coverage >= predeclared threshold):
- Sparsity acceptable (no extreme concentration):
- Override rate non-trivial but bounded:
- `Delta_success` non-negative:
- `Delta_burden` non-positive (or within tolerance):
- No feasibility regression:

Final recommendation:
- `KEEP AS-IS` / `TUNE SIGNATURE GRANULARITY` / `RAISE n_min` / `LOWER influence` / `DISABLE TEMPORARILY`
