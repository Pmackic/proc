# PROK v5 Methodology (Study-Ready, Multi-Task)

This document specifies a **study-ready methodology** for developing and evaluating **PROK v5**, a cybernetic procrastination regulator (Ashby-style feedback control) that integrates **discrete TMT variables**, **Haghbin procrastination criteria**, a **finite transform library (strategies)**, and an explicit **user goal/fitness model**.  
It also specifies how **user-facing language** and **strategies** are **mined from focus groups**, then refined empirically.  
**CCA is included only as a placeholder and is set to NOT USED in v5**.

**Upgrade notes (v4 → v5):**
- Multi-task workload with per-task outcomes and per-domain recursion
- Algedonic alarm adds structural-fix guidance when feasibility degrades

---

## 1) Research aims

### A1 — Construct the regulator’s “requisite variety”
Build a justified, finite set of:
- **States** (e.g., engaged, drift, procrastination, blocked)
- **Triggers** (e.g., confusion, boredom, distraction, anxiety, fatigue, other)
- **Transforms/strategies 𝒜**: micro-interventions that move users back toward engagement
- **Genes**: discrete controller parameters (policy knobs) that determine how regulation is applied

Operationalization in runtime:
- track disturbance-class variety from logs,
- track response-action variety from applied recoveries,
- enforce periodic audit that controller variety does not collapse below disturbance variety.

### A2 — Build and validate a Regulatory Calibration Questionnaire (RCQ)
Create a questionnaire that estimates:
- disturbance profile (trigger exposure),
- check tolerance/annoyance sensitivity,
- recovery preferences and expected efficacy,
- delay appraisal (rational vs needless delay tendencies),
- time-horizon feel (near vs far),
- self-monitoring reliability,
- **fitness/goal priorities** (what “success” means to the user).

### A3 — Parameterize and tune the regulator
Use focus-group products + RCQ priors + diary/log evidence to tune:
- action selection (which strategy to apply),
- monitoring intensity (check gaps),
- block length recommendations,
- tightening and replanning thresholds.

### A4 — Field-evaluate outcomes and acceptability
Test whether PROK v5 improves:
- **time-to-recover** after drift/procrastination,
- **feasibility preservation** (slack not negative),
- progress and completion,
- adherence and user burden.

---

## 2) System under study (brief)

PROK v5 implements an Ashby regulator loop:

- **Plant**: person + task + environment
- **Essential variables**: feasibility (slack), progress, recovery latency, adherence, burden
- **Observation**: self-reports (d/p/b/e), optional micro-ratings (burden/helpfulness), context tags
- **Regulator**: selects transforms 𝒜 and adapts policy “genes” over time
- **Goal/fitness model ρ**: user priorities affect action ranking (without changing the core measurable outcomes)

Design-for-a-Brain compliance commitments:
- **Essential variables explicitly monitored** (feasibility/drift boundaries),
- **Requisite variety diagnostics** in constitutional checks,
- **Ultrastable step mechanism** (bounded NK mutation) when essential variables are threatened.

### Three-layer architecture (big-picture framing)
PROK v5 should be analyzed as three coupled layers:
1) **Control core**: feasibility preservation via slack, replanning, and alarm channel  
2) **Policy layer**: fitness DSL + goal profile ρ (explicit ranking of acceptable actions)  
2b) **System 2 coordination layer**: anti-oscillation damping service across operational units/tasks  
3) **Phenomenology layer**: signature-based homology memory that transfers empirically successful recoveries across structurally similar episodes

This decomposition is methodologically important:
- adaptation can improve without introducing opaque end-to-end models
- safety is constrained by policy envelopes (phenomenology re-ranks within admissible candidates)
- all layers remain auditable in local state/log artifacts

v5 adds **task-level** and **domain-level** essential variables:
- per-task slack (capacity until deadline − remaining blocks)
- per-task completion and recovery outcomes
- per-domain drift EMA (stability by domain)

Integrated model components (evaluated within the study, not as separate tracks):
- calendar-aware routing features (time-to-next-event, event density) in policy selection,
- energy/circadian bins as context moderators for intervention efficacy,
- dependency-aware task graph constraints in replanning,
- deadline hazard/risk bins feeding alarm/homeostat predicates,
- causal phenomenology estimators (IPW baseline, DR upgrades where feasible),
- counterfactual planning probes for projected slack/alarm under alternative effort plans,
- explicit user policy constraints (check caps, no-prompt windows),
- decision reason codes for per-step auditability,
- staleness/confidence decay in signature and NK memory,
- incremental online updates for large-log operation.
- System 2 damping controls (minimum stick time, max switches/day, current-task preference) with explicit override tracking.

### Discrete motivational layer (TMT bins)
Discrete bins:  
- Expectancy **E ∈ {L,M,H}**  
- Value **V ∈ {L,M,H}**  
- Delay feel **D ∈ {N,F}** (near/far)  
- Impulsiveness **Γ ∈ {L,H}**

### Procrastination classifier (Haghbin operationalization)
Haghbin’s concept is operationalized with minimal checks:
- Intention (I)
- Expected cost/consequences (C)
- Needless/irrational delay (N)
- Negative affect (F)
Plus two elements inferred structurally:
- Voluntary delay (from report/behavior)
- Awareness (from loop participation and reflective prompts)

---

## 3) Study design overview (phases)

**Phase 1 — Focus groups**  
Elicit (a) behavior pool, (b) state/trigger taxonomy, (c) **strategies 𝒜** (transforms) from real episodes, (d) genes + alleles, (e) user-facing language pack, (f) user fitness priorities.

**Phase 2 — RCQ development & validation**  
Psychometric development + criterion validation against diary/log outcomes.

**Phase 2.5 — Diary/ESM bridge (7–14 days)**  
Estimate empirical transition dynamics and transform effectiveness; measure observed burden and perceived success.

**Phase 3 — Controlled tuning / exploration (NK-compatible)**  
Introduce deliberate policy variation (micro-randomization and/or crossover of neighbor genomes) to estimate gene effects and interactions.

**Phase 4 — Field evaluation (pilot then controlled)**  
Within-person baseline → regulator activation, optionally RCT; outcomes include feasibility, recovery, adherence, burden, and completion.

---

## 4) Participants and recruitment

### Population
Students and community adults engaged in goal-directed tasks with deadlines (e.g., exam prep, writing, admin, job tasks).

### Inclusion criteria
- Age ≥ 18 (or ethics-approved adolescent protocol if applicable)
- Current deadline-bound task
- Ability to run the CLI/app and complete brief self-reports

### Target sample sizes (pragmatic)
- Phase 1: 2–4 focus groups × 6–10 (≈12–40)
- Phase 2: pilot n≈150 (item reduction), validation n≈200–300 (CFA)
- Phase 2.5: n≈60–120 for 7–14 days
- Phase 4: pilot n≈40–80; later controlled study n≈120–250

(Adjust based on resources and ethics constraints.)

---

## 5) Phase 1 — Focus groups (behavior pool + strategies + genes + language + goals)

### 5.1 Structure (60–90 minutes)
1) **Episode elicitation**: “Tell a recent time you drifted/procrastinated. What happened right before? What did you do next?”  
2) **State & trigger mapping**: Participants map episodes to states/triggers; propose missing categories.  
3) **Strategy mining (transforms 𝒜)**: “What got you back to work fastest? What backfired? What would you want the system to suggest?”  
4) **Gene/allele elicitation** (controller knobs): thresholds and preferences that can be made discrete (block lengths, check gaps, tightening, routing, pause rules, replan thresholds).  
5) **Language elicitation (λ)**: preferred words for states/triggers, acceptable vs annoying phrasing, comprehension probes.  
6) **Fitness/goal elicitation (ρ)**: what outcomes matter most (feasibility, speed, stress, autonomy, annoyance, learning quality, grades).

### 5.2 Strategy mining deliverable (explicit)
Strategies are mined via open coding into **canonical transform families**, each with:
- Preconditions: (state, trigger, minimal context)
- Steps (≤ 5 minutes to execute)
- Expected transition target (recovering/engaged)
- Failure modes (“when this doesn’t work”)

### 5.3 Coding and reliability
- Transcripts open-coded into: state, trigger, strategy, gene statement, language preferences, goal priorities.
- **Two independent coders** code a subset (≥20–30%).
- Reliability:
  - **Cohen’s κ** (two coders) or **Krippendorff’s α** (≥2 coders) for state/trigger/strategy codes.
  - Target: κ/α ≥ .70 for core categories.

### 5.4 Preference estimation for goals
To quantify user-level fitness priorities:
- Use **Best–Worst Scaling (BWS)** or **paired comparisons** on outcome attributes.
- Models:
  - BWS: **conditional logit** (and optionally **mixed logit**)
  - Paired comparisons: **Bradley–Terry model**
- Output: per-user or per-class preference vectors ρ.

---

## 6) Phase 2 — RCQ development and validation (includes goal/fitness)

### 6.1 Item generation
Items derived from:
- Phase 1 codes and wording,
- theoretical constructs (TMT/Haghbin/self-regulation),
- diary feasibility (low burden).

### 6.2 Core RCQ domains
1) Trigger exposure profile  
2) Recovery preference/expected efficacy  
3) Check tolerance / annoyance sensitivity  
4) Delay appraisal (rational vs needless delay)  
5) Time-horizon feel (near reward need)  
6) Self-monitoring reliability  
7) **Fitness priority module** (short BWS-like items or stable preference scales)

### 6.3 Psychometric analysis
- Suitability: **KMO**, **Bartlett’s test**
- Factor count: **parallel analysis**
- Structure:
  - **EFA** (oblique rotation) → item reduction
  - **CFA** (second sample or split-sample)
- Fit indices: **CFI/TLI**, **RMSEA**, **SRMR**
- Reliability: **McDonald’s ω** (primary), α (secondary)
- Test–retest (subset): **ICC(2,1)** or Pearson r
- Validity:
  - Criterion: predict diary/log outcomes (recovery time, drift frequency, burden)
  - Convergent/discriminant: optional links to HEXACO/PIKSNAP (not required)
- Invariance (if sample allows): configural → metric → scalar (ΔCFI/ΔRMSEA thresholds)

---

## 7) Phase 2.5 — Diary/ESM bridge (7–14 days)

### 7.1 Measures per prompt
- State (engaged/drift/procrastinate/blocked)
- Trigger tag
- Brief Haghbin markers (minimal)
- Strategy used (if any)
- Outcome: recovered within X minutes (yes/no)
- Burden/annoyance rating (optional 1–5)
- Daily: minutes progressed and perceived success
- Calendar proximity features (time-to-next-event, event density in day window)
- Energy/circadian bin (time-of-day + optional fatigue self-report)
- Task dependency context (blocked by prerequisite yes/no)
- Decision trace fields (homeostat/fitness/phenom/NK reason codes)

### 7.2 Analyses
**State dynamics**
- Empirical transition matrices
- **Multilevel multinomial logistic regression**: next-state ~ state + trigger + time-of-day + (random intercepts per person)
- Optional: **Hidden Markov Model** if latent states suspected

**Task/domain heterogeneity (v5)**
- Mixed-effects models with **task nested in person** (random intercepts for person and task)
- Include domain as a fixed effect or random slope where data permit
- Report variance components to quantify cross-task instability

**Strategy effectiveness**
- Recovery success (binary): **mixed-effects logistic regression**
- Recovery time: **multilevel survival analysis** (Cox frailty or AFT mixed model)

**RCQ → behavior**
- Predict outcomes: mixed models
- Optional cross-validated predictive modeling (AUC/RMSE)
- Optional **CCA** reserved for later (see §11); not used in v5.

**Phenomenology diagnostics (new)**
- Signature coverage: proportion of events assigned to signatures with evidence `n >= n_min`
- Signature sparsity: distribution of sample counts per signature (heavy-tail concentration indicates over-fragmentation)
- Transfer gain: delta in recovery success/time when selection source is `fitness+phenomenology` vs `fitness` only
- Confidence gating effect: how often overrides are blocked by evidence threshold or influence threshold

**Causal-DAG phenomenology diagnostics (implemented option)**
- When `mode=dag`, estimate action effects with backdoor-adjusted IPW within logged course-correct events.
- Required covariates are configurable (`dag_adjust`; default `domain,label,trigger,slack,drift`).
- Core estimands:
  - `E[helped | do(action)]`
  - `E[overcontrol | do(action)]`, where overcontrol is `burden>=4`
- DAG-adjusted score term:
  - `delta_do(action) = (do_helped-0.5) - mu_overcontrol*do_overcontrol`
  - combined as `phi + dag_kappa*delta_do`.
- Policy integration constraint: DAG re-ranking remains inside homeostat-feasible top-K options only.

Recommended models:
- Mixed-effects logistic for override acceptance (accepted ~ signature_count + burden + trigger + (1|person))
- Mixed-effects survival/linear models for recovery-time deltas conditional on selection source
- If `mode=dag`, include adjusted-policy indicator:
  - `helped ~ dag_adjusted_selection + trigger + domain + (1|participant)`
  - `burden ~ dag_adjusted_selection + trigger + domain + (1|participant)`

---

## 8) Phase 3 — Controlled tuning / NK-compatible exploration

### 8.1 Genotype definition (genes)
A small discrete genotype (example set):
- Block mapping rule (alleles: 10/25/50 or state-dependent)
- Check-gap rule (tight/medium/loose; random vs scheduled)
- Tightening strength (none/mild/strong)
- Routing policy (E-first/V-first/Γ-first/trigger-based)
- Replan threshold (1/2/3 drift events)
- Pause policy (soft/hard)
- Near-reward enforcement (off/on)

### 8.2 Exploration designs
**Micro-randomized trial (MRT)**  
When drift/procrastination is detected, randomize among a small subset of candidate strategies.

**Crossover of neighbor genomes**  
Rotate “one-gene mutations” across days/weeks to estimate main effects and key interactions.

**Context-policy interventions (integrated)**
- Randomize calendar-aware short-form routing near events vs standard routing.
- Randomize energy-aware routing vs context-agnostic routing.
- Randomize memory-decay settings to test staleness handling.
- Evaluate user policy constraints (check caps/no-prompt windows) as enforced policy conditions.
- Evaluate System 2 damping settings:
  - min stick time,
  - max switches/day,
  - current-task preference under non-urgent conditions.

### 8.3 Analyses
- MRT proximal outcomes: **WCLS** or **GEE**, with moderators (trigger, RCQ scales)
- Gene effects: hierarchical mixed models with preregistered interactions
- Multiple comparisons: **Benjamini–Hochberg FDR**
- Ruggedness proxies (for NK claims): bootstrapped variance of local fitness differences across neighboring genomes

---

## 9) Phase 4 — Field evaluation (pilot and/or controlled)

### 9.1 Designs
**Pilot (within-person)**  
Week 1 baseline planning/logging → Weeks 2–4 regulator active.

**Controlled (optional RCT)**  
Planner-only vs planner+regulator.

### 9.2 Primary outcomes (system-level fitness)
- Time-to-recover after drift/procrastination
- Feasibility preservation (slack ≥ 0 over horizon)
- Progress rate (minutes/day; sessions/week)
- Adherence (usage retention)
- Burden/annoyance
- Deadline miss risk calibration (hazard/risk-bin reliability)
- Counterfactual planning accuracy (predicted vs realized slack/alarm under documented plan variants)

### 9.3 Secondary outcomes (user-level fitness alignment)
- Satisfaction with progress
- Perceived autonomy/control
- Stress (brief validated scale if possible)
- Self-efficacy

### 9.4 Statistical models
- Continuous: **linear mixed models**
- Binary: **mixed-effects logistic**
- Counts: **negative binomial mixed**
- Time-to-event: **Cox frailty** or mixed AFT

Baseline adjustment:
- ANCOVA-style: include baseline as covariate in mixed models

Utility / goal-weighted analysis:
- Estimate preference weights ρ from BWS/paired comparisons
- Compute participant-specific utility U_i from standardized outcomes
- Model U_i with mixed models; sensitivity analyses with alternative weightings

Missing data:
- Mixed models under MAR
- Sensitivity: **multiple imputation**; pattern-mixture checks if needed

Phenomenology-specific reporting in field evaluation:
- report override rate and confidence-gated override rate
- stratify outcomes by high-coverage vs sparse signatures
- include ablation: policy layer only vs policy+phenomenology vs policy+phenomenology(DAG)

---

## 10) Empirical substitution points (what starts hard-coded, what becomes learned)

### Hard-coded in v5
- State taxonomy (initial)
- Trigger list (initial)
- Initial transform families 𝒜
- Discrete TMT bins and update heuristics
- Default genome values
- Fitness DSL (base ranking rules)
- CCA interface exists but is OFF
- DAG structure defaults are explicit and user-configurable (`dag_adjust`, `dag_kappa`, smoothing/min-sample guards)

### Learnable via study
- Expand/refine states/triggers (Phase 1–2.5)
- Transform library expansion from focus groups (Phase 1) + pruning from logs (Phase 2.5–4)
- Transition probabilities and context moderators (Phase 2.5)
- Personalized priors over genes from RCQ (Phase 2)
- Gene effects and interactions from MRT/crossover (Phase 3)
- Fitness weights ρ from BWS/paired comparisons (Phase 1–4)
- Language pack λ refined for comprehension and burden reduction (Phase 1–4)
- Signature map design (granularity/coarsening rules) to control sparsity while preserving actionable structure
- Phenomenology thresholds (`n_min`, influence) tuned from empirical transfer-performance curves
- Calendar-aware routing parameters and near-event policy switches
- Energy/circadian-conditioned effectiveness maps
- Task graph dependency effects on replanning performance
- Deadline hazard model calibration and alarm-risk thresholds
- User policy constraint defaults and tolerance profiles
- Decision-trace schema sufficiency for audit/reproducibility
- Memory staleness decay schedules and incremental update windows

---

## 11) CCA placeholder (NOT USED in v5)

CCA is reserved for later versions as an optional mapping from (RCQ + trait batteries) → genome priors.  
In **v5**:
- `use_cca = false`
- no CCA outputs are applied to policy
- analysis may store CCA-ready datasets, but it is not part of the regulator or primary inference.

---

## 12) Ethics and risk management
- The regulator is not a diagnostic tool and does not guarantee perfect control.
- Burden is minimized: short prompts, optional ratings, easy pause/quit.
- Participants can discontinue at any time.
- Data are anonymized; logs are stored securely.
- In high-stress contexts (exams), emphasize “recovery” rather than “failure.”

### 12.1 Aristotelian clarification for study interpretation
Use the following interpretation frame:
- **Physics:** the regulator supports transition from delayed potential to enacted work (timely activity).
- **Categories:** operational variables are treated as structured predicates of task/domain/time/quality/relation (not a single scalar mood).

### 12.1.1 Additional interpretation anchors
- **Ethics:** target is an engaged practical mean (flow-capable), not maximal pressure.
- **Politics:** user agency is preserved by local-first control, transparency, and right to pause/quit/retune language.

### 12.2 Unequal doctrine of the mean (procrastination-centered)
The ethical center is asymmetric:
- deficiency side: procrastination/avoidant delay (primary failure mode),
- excess side: over-control burden/annoyance,
- mean: sustainable engaged progress.

Methodological rule:
- treat procrastination reduction as higher-priority than small burden increases when feasibility is under threat,
- but never relax hard safeguards against coercive overreach (pause rights, burden tracking, feasibility constraints).

### 12.3 What to report empirically
In addition to standard outcomes, report an asymmetry audit:
- `Pr(procrastination-like events)` vs `Pr(over-control burden events)`
- burden change conditional on procrastination reduction
- any evidence that burden minimization degraded feasibility or recovery

A design is ethically acceptable only if:
- feasibility is preserved,
- procrastination burden decreases over time,
- burden costs remain within predeclared tolerances.

### 12.4 Analogical reporting extension (physics -> categories -> politics)
For theory-consistent reporting, include one short mapping table in studies:
- disturbance/entropy indicators,
- classifier/action differentiation (demon role),
- algedonic state transitions (`ok/warning/critical`),
- memory/policy recursion markers (autopoietic maintenance),
- Aristotelian final-cluster outcomes (feasible, engaged, proportionate, agency-preserving).

This keeps philosophical interpretation tethered to measurable system artifacts rather than narrative-only claims.

---

## 13) Reporting checklist (what to include in papers)

**Phase 1 paper**
- Codebook, κ/α reliability, strategy library, gene list, language findings, preference model results

**Phase 2 paper**
- RCQ structure (EFA/CFA), reliability, validity against diary/log outcomes, invariance (if tested)

**Phase 4 paper**
- Design, primary outcomes, mixed model results, adherence/dropout, burden, goal-weighted utility analysis
- Phenomenology audit: signature coverage/sparsity, override behavior, and ablation results

Operational template:
- Use `PHENOM_AUDIT_TEMPLATE.md` as the minimum reporting structure for phenomenology analyses.

---

## 14) Minimal publishable path (fastest credible)
1) Phase 1: strategy mining + gene justification + language pack + goal preferences  
2) Phase 2: RCQ development/validation  
3) Phase 4 pilot: within-person baseline vs regulator with logged outcomes and burden
