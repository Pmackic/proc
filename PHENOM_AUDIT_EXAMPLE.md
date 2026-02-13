# PROK v5 Phenomenology Audit Example

This example is pre-filled from `.prok_log.jsonl` as of 2026-02-13.

---

## 1) Study metadata

- Study ID: `local-auto-audit`
- Dataset window: `2026-02-04 to 2026-02-12`
- Participants (n): `1`
- Sessions (n): `5`
- Course-correct events (n): `9`
- Phenomenology settings at audit time (`.prok_state.json`):
  - `enabled`: `true`
  - `min_samples (n_min)`: `3`
  - `influence (tau)`: `0.35`

Note:
- Log events are legacy/pre-phenomenology for intervention events; no `phenomenology_signature` fields are present.
- Log events are legacy/pre-phenomenology for intervention events; no `selection_source` fields are present.

---

## 2) Core metrics

### A) Coverage and sparsity

- Signature count `|Sigma|`: `9`
- Events mapped to signatures: `9 / 9 = 1.00`
- Evidence-qualified coverage (`n(signature) >= n_min`): `0 / 9 = 0.00`
- Median events per signature: `1`
- P90 events per signature: `1`
- Gini over signature counts: `0.00`

### B) Override behavior

- Candidate decisions total: `9`
- Baseline-only selections (`selection_source=fitness`): `N/A (legacy schema)`
- Phenomenology selections (`selection_source=fitness+phenomenology`): `0`
- Override rate: `0 / 9 = 0.00`
- Confidence-gated rejection count: `N/A`

### C) Outcome deltas

- Recovery success rated events: `4`
- Recovery success yes: `3`
- Recovery success rate (overall rated): `0.75`
- Burden ratings count: `5`
- Burden mean: `2.20`

Delta block:
- `Delta_success`: `N/A`
- `Delta_burden`: `N/A`
- `Delta_time`: `N/A`

---

## 3) Required tables

### T1) Signature summary

| signature | events_n | qualified (`>=n_min`) | best_action | best_action_rated_n | best_action_helped_rate | best_action_burden_avg |
|---|---:|---|---|---:|---:|---:|
| skola|DRIFT|anxiety|EMVMDNGL | 1 | no | E_RECOVERY | 1 | 1.00 | 3.00 |
| skola|DRIFT|confusion|EMVMDNGL | 1 | no | E_RECOVERY | 1 | 1.00 | 2.00 |
| unknown|DRIFT|anxiety|EMVLDNGL | 1 | no | E_RECOVERY | 0 | N/A | N/A |
| unknown|DRIFT|distraction|EHVMDNGL | 1 | no | E_RECOVERY | 0 | N/A | N/A |
| unknown|PROCRASTINATION|boredom|EHVLDNGL | 1 | no | E_RECOVERY | 0 | N/A | N/A |
| unknown|PROCRASTINATION|confusion|EMVMDNGL | 1 | no | E_RECOVERY | 0 | N/A | N/A |
| unknown|PROCRASTINATION|fatigue|EMVLDNGL | 1 | no | E_RECOVERY | 0 | N/A | 2.00 |
| work|DRIFT|distraction|EHVMDNGL | 1 | no | E_RECOVERY | 1 | 1.00 | 3.00 |
| work|PROCRASTINATION|fatigue|EMVMDNGL | 1 | no | E_RECOVERY | 1 | 0.00 | 1.00 |

### T2) Action comparison by selection source

| selection_source | events_n | helped_yes_n | helped_rated_n | helped_rate | burden_n | burden_avg | time_to_recover_avg |
|---|---:|---:|---:|---:|---:|---:|---|
| fitness (inferred, legacy) | 9 | 3 | 4 | 0.75 | 5 | 2.20 | N/A |

### T3) Trigger/domain stratification

| domain | trigger | selection_source | events_n | helped_rate | burden_avg |
|---|---|---|---:|---:|---:|
| skola | anxiety | fitness (inferred, legacy) | 1 | 1.00 | 3.00 |
| skola | confusion | fitness (inferred, legacy) | 1 | 1.00 | 2.00 |
| unknown | anxiety | fitness (inferred, legacy) | 1 | N/A | N/A |
| unknown | boredom | fitness (inferred, legacy) | 1 | N/A | N/A |
| unknown | confusion | fitness (inferred, legacy) | 1 | N/A | N/A |
| unknown | distraction | fitness (inferred, legacy) | 1 | N/A | N/A |
| unknown | fatigue | fitness (inferred, legacy) | 1 | N/A | 2.00 |
| work | distraction | fitness (inferred, legacy) | 1 | 1.00 | 3.00 |
| work | fatigue | fitness (inferred, legacy) | 1 | 0.00 | 1.00 |

---

## 4) Minimal statistical checks

Not auto-run by this command.

---

## 5) Ablation block

Current result:
- Baseline policy behavior is observed from available events.
- Post-upgrade phenomenology-vs-fitness ablation requires new intervention data.

---

## 6) Decision rubric

- Coverage adequate: `FAIL`
- Sparsity acceptable: `FAIL`
- Override rate non-trivial but bounded: `WATCH`
- `Delta_success` non-negative: `WATCH`
- `Delta_burden` non-positive: `WATCH`
- No feasibility regression: `WATCH`

Final recommendation:
- `TUNE SIGNATURE GRANULARITY` and collect more post-upgrade intervention data before increasing influence.
