# PROK v5: Ashby–TMT–Haghbin Regulator with Multi-Task Planning + Domain Recursion + Algedonic Alarm (Spec)

This is a **buildable specification** for PROK v5.

**Upgrade notes (v4 → v5):**
- Replace single-plan fields with `tasks[]` + `active_task_id`.
- Per-task TMT bins and drift counters; per-domain recursion state.
- Multi-task planner allocates daily blocks across tasks.
- Add algedonic alarm channel computed from slack + drift signals.

**What v5 adds:**
- Multiple concurrent tasks with per-task deadlines and workload
- Domain recursion: per-domain drift EMA and preferred block/check overrides
- Algedonic alarm channel (ok/warning/critical) with structural-fix guidance
- All v4 study mode, goals, language, and CCA-placeholder remain

---

## 0) Big picture

User provides:
- multiple tasks (domain, workload, deadline)
- constraints (when they cannot work)
- session interaction signals (check responses / self-report)
- (optional) fitness priorities (what matters: recover fast, low annoyance, etc.)
- (optional) vocabulary preferences (“drift” vs “slip”, “blocked” vs “stuck”)

System does:
- plan a feasible multi-task schedule (allocate blocks across tasks)
- run sessions with checkclock + immediate report
- detect drift/procrastination/blocked
- apply corrective actions + tighten policy
- replan after slippage to keep feasibility until deadline
- log outcomes in a study-ready way
- optional NK learns which settings work best by user type
- optional CDA/traits seed priors (future)
- optional CCA mapping **exists as an interface** but is OFF in v5
- compute algedonic alarm (ok/warning/critical) from slack + drift

### 0.1 Ethical clarification (Aristotelian framing)
This regulator is normatively oriented to practical agency, not mere output.

Aristotelian anchors used in v5 interpretation:
- **Physics (kinesis -> energeia):** regulation moves episodes from blocked/delayed potential into timely activity.
- **Categories:** observations are structured as substance/context (task/domain), quality (drift/procrastination/flow markers), relation (deadline/slack), quantity (minutes/blocks), time (session/horizon), and action/passion (intervention/response).
- **Ethics (doctrine of the mean):** target state is not maximal pressure, but stable engaged work ("flow-capable engagement").
- **Politics:** the system remains non-coercive and local-first; user pause/quit and language autonomy are preserved as governance constraints.

### 0.2 Unequal doctrine of the mean (flow-centered, procrastination-weighted)
v5 adopts an **asymmetric** mean:
- left deviation: procrastination / avoidant delay (primary risk to feasibility and agency)
- center: engaged, sustainable progress (flow-compatible)
- right deviation: over-control / excessive pressure / annoyance burden

The asymmetry is intentional:
- procrastination receives stronger corrective priority than mild annoyance when feasibility is threatened,
- but hard safety constraints prevent collapsing into coercive over-control.

Operational rule:
- optimize toward the center with unequal penalties, typically `penalty(procrastination) > penalty(annoyance)`,
- while maintaining viability guards (`slack != neg`, recovery not persistently slow).

### 0.3 Analogical atlas (Mihajlo Alas -> Aristotelian final cluster)
From the system viewpoint, the same regulator can be read through multiple mathematical-physics analogies:

1) **Dynamical-system analogy**  
- state trajectory in a constrained phase space; procrastination is drift away from viable attractors.

2) **Thermodynamic/entropy analogy**  
- disturbances increase disorder in task execution; regulation injects structured work (negative entropy budget).

3) **Variational/FEP analogy**  
- action selection minimizes expected free-energy-style score under ethical asymmetry.

4) **Information-theoretic analogy**  
- observation channel reduces uncertainty; control channel must supply requisite response variety.

5) **Homological/phenomenological analogy (Mihajlo Alas/Petrović)**  
- episodes are grouped by invariant structural signatures; homologous cases share transferable corrective forms.

6) **Control-theoretic analogy**  
- homeostat defines admissible set; policy and phenomenology choose within constraints to preserve essential variables.

7) **Cybernetic demon analogy (Maxwell-style, algorithmic)**  
- the regulator acts as a bounded demon: it sorts disturbances by observation class and applies differential actions.

8) **Autopoietic analogy**  
- the system reproduces its own regulatory organization by recursive logging, memory updates, and policy retuning.

Aristotelian final cluster (teleological interpretation):
- **telos-feasible:** preserve viability (`slack >= 0`)
- **telos-engaged:** sustain flow-capable action
- **telos-proportion:** apply unequal mean (stronger anti-procrastination correction without coercive excess)
- **telos-political:** keep agency rights (pause/quit/local autonomy)

Algedonic coupling:
- algedonic alarm (`ok/warning/critical`) is the bridge between these analogies and operation:
  it signals when the demon-like sorter/autopoietic updater must shift from routine correction to structural redesign.

---

## 1) Ashby closed-loop formulation

### 1.1 Sets
- Domains:  d ∈ 𝒟
- Tasks:    τ ∈ 𝒯 with τ = (id, d, name, deadline, workload)
- Plant state (person + context): s_t ∈ 𝒮_d
- Disturbance: w_t ∈ 𝒲
- Observation: u_t ∈ 𝒰
- Regulator state: x_t ∈ 𝒳
- Action: a_t ∈ 𝒜
- Essential variables (must stay in bounds): e_t ∈ ℝ^m

### 1.2 Plant + observations
Plant (unknown, stochastic):
- s_{t+1} ~ T_d(s_t, a_t, w_t)

Observation channel:
- u_t = H(s_t, w_t)

Essential variables extracted from plant:
- e_t = E(s_t)

**Core essential variables (study outcomes):**
- feasibility / slack ≥ 0 (capacity remaining − blocks remaining), per task
- progress slope (remaining workload decreases fast enough)
- recovery time after drift/procrastination
- dropout / “annoyance” events (disable checks / quit / reported burden)

---

## 2) Regulator as an Ashby transducer

A regulator instance is parameterized by a discrete “genome” g:
- R_g := (𝒳, 𝒰, 𝒜, F_g, G_g)

Update + output:
- x_{t+1} = F_g(x_t, u_t)
- a_t     = G_g(x_t, u_t)

---

## 3) Regulator state x_t (explicit decomposition)

Represent:
- x_t = (T_t, a_t, c_t, h_t, m_t, Θ_t, Π_t, ρ_t, λ_t, D_t, A_t)

### 3.1 Task set T_t (NEW)
- T_t = { τ_i } where each task τ_i = (id, domain, name, deadline, B_tot, B_rem, θ_i, recentDrift_i)
- a_t = active_task_id (optional)

### 3.2 Constraints c_t
- c_t = (W, 𝓑, B_max)

### 3.3 Habit/momentum h_t
- h_t = (minDone ∈ {0,1}, streak ∈ ℕ)

### 3.4 Mode m_t
- m_t ∈ {idle, engaged, paused, recovering}

### 3.5 Discrete TMT per task Θ_t (NEW)
- θ_i = (E_i, V_i, D_i, Γ_i) for each task
  - E ∈ {L, M, H} expectancy
  - V ∈ {L, M, H} value
  - D ∈ {N, F}   reward horizon near/far
  - Γ ∈ {L, H}   impulsivity/sensitivity

### 3.6 Active policy Π_t
- π_i = (b_i, Δ^check_i) derived from θ_i
  - b_i ∈ {10,25,35,50} block length (minutes)
  - Δ^check_i = (Δ_min, Δ_max) check gap range

### 3.7 User fitness / goal profile ρ_t
ρ_t captures what the user considers “good regulation” in this context.

Minimal representation (study-ready, no weights required):
- ρ_t = ordered tiers of priorities (lexicographic), plus tolerances

Example:
- tier1: feasibility
- tier2: recover_fast
- tier3: low_annoyance
- tier4: streak_up

Optional:
- annoyance tolerance threshold (how often checks can fire before “too much”)
- preferred outcomes for the current domain (e.g., learning quality vs speed)

### 3.8 Language pack λ_t (NEW)
- λ_t is a mapping from canonical labels → user-facing terms/templates.

Examples:
- “drift” → “slip”
- “blocked” → “stuck”
- recovery script templates in user-preferred phrasing

**Design rule:** canonical internals stay stable; λ_t is the empirical skin.

### 3.9 Domain recursion state D_t (NEW)
- D_t maps domain → state:
  - preferred block length override (optional)
  - check-gap overrides (optional)
  - drift EMA (domain-level stability)
  - recovery-family counters (which families worked most often)

### 3.10 Algedonic channel A_t (NEW)
- A_t ∈ {ok, warning, critical}
- Computed from real signals:
  - negative task slack (infeasible)
  - drift clusters (recentDrift_i high) or domain drift EMA high
- When A_t = critical, the CLI must instruct a **structural fix** (capacity, scope, deadline).

---

## 4) Observations u_t (what the CLI can sense)

Minimal components:
- time ticks + elapsed session time
- check prompts fired
- check responses: {e,d,p,b}
- self-reports: {d,p,b,r,c,q}
- progress logs: minutes done, miss minimum
- optional trigger tag r_t
- optional Haghbin flags (fast prompts)

Study mode adds:
- burden rating (e.g., 1–5) after interventions
- perceived success (e.g., “did the suggestion help?” yes/no)
- reason for quit (optional)

Trigger tag:
- r_t ∈ {confusion, boredom, distraction, anxiety, fatigue, other}

---

## 5) Haghbin classifier (procrastination justification)

Encode “necessary elements” as flags:
- η_t = (I_t, C_t, N_t, F_t)  (minimal queried set)
  - I: intention present?
  - C: expected cost if not done?
  - N: needless/irrational delay acknowledged?
  - F: negative affect present?

Classifier label:
- ℓ_t = HAG(η_t) =
  - PROCRASTINATION if I=1 ∧ C=1 ∧ N=1
  - DRIFT          if I=1 ∧ (C=0 or N=0)
  - NO_INTENT      if I=0

**Note:** Haghbin’s “voluntary delay” and “awareness” can be (a) inferred structurally from interaction,
and (b) measured explicitly in study mode if needed.

---

## 6) Discrete TMT router (Haghbin + trigger → θ update)

Update operator:
- θ_{t+1} = U_θ(θ_t, ℓ_t, r_t, h_t)

Starter rules (hard-coded v5 defaults):
- if r ∈ {confusion, anxiety} → E ↓
- if r = boredom              → V ↓
- if r = distraction          → Γ → H
- if ℓ = PROCRASTINATION      → D → F
- if recovery succeeds:
  - E ↑ or V ↑ (depending on recovery family)
  - Γ → L
  - D → N if a quick-win artifact was produced

**Empirical substitution point:** replace these rules with learned transition tables from diary/logs.

---

## 7) Planner / replanner (multi-task allocation)

Compute daily capacity (blocks) from constraints:
- κ_t = CAP(c_t, day t) ∈ ℕ

Per-task needed blocks:
- need_i = ceil(B_rem_i / block_min)

Minimal allocator (priority fill):
- Sort tasks by earliest deadline, then lowest slack.
- Fill each day’s capacity with blocks for tasks in that priority order.

Outputs:
- feasible? (Σ κ_t ≥ Σ need_i)
- peak cap P*
- allocation plan `plan[date][task_id] = blocks`
- per-task slack summary (capacity until deadline − need_i)

---

## 8) Immediate effects (seconds–minutes control)

State skeleton:
- ENGAGED → (check/self-report) → DRIFTING / PROCRASTINATING / BLOCKED
- (DRIFTING/PROCRASTINATING/BLOCKED) → RECOVERING → ENGAGED
- any → PAUSED → ENGAGED (resume)

Immediate effects when drift/procrastination happens:
1) classify ℓ_t (Haghbin)
2) update θ_t (discrete TMT bins)
3) select recovery family (E-first vs V-first vs Γ-first vs mixed)
4) tighten check policy (smaller gaps), within annoyance tolerances from ρ_t
5) optionally enforce near-reward artifact (push D→N)
6) replan if threshold crossed

---

## 9) Homomorphism across domains (structure preserved)

Domain state:
- 𝒳_d = 𝒟 × 𝒳, with x_d = (d, x)

Mappings:
- h_I : I_d → 𝒰      (translate domain-language inputs to canonical observations)
- r_d : 𝒜 → 𝒜_d      (render canonical actions in domain language via λ_t)
- h_X(d,x) = x        (forget domain skin)

Homomorphism conditions:
- h_X(F_d((d,x), i_d)) = F_g(x, h_I(i_d))
- G_d((d,x), i_d) = r_d(G_g(x, h_I(i_d)))

**Empirical layer:** focus groups and usability testing primarily modify λ_t and the trigger vocabulary.

### 9.1 Phenomenological homology memory (implemented extension)
To operationalize Petrović-style mathematical phenomenology, define a finite signature map:
- σ_t = SIG(d, ℓ_t, r_t, θ_t) ∈ Σ
- Example encoding: `domain|label|trigger|E{E}V{V}D{D}G{G}`

Each signature class stores empirical recovery outcomes:
- M(σ) = { per-action tries, helped-rate, burden stats, sample count }

Selection rule in v5 extension:
1) rank candidates by fitness DSL (and goal profile ρ)
2) restrict to top-K candidates (K=3 in current implementation)
3) if M(σ_t) has enough evidence (n ≥ n_min), allow phenomenology to re-rank within top-K
4) apply only if empirical score exceeds influence threshold τ

Interpretation:
- this preserves structural homology by transferring action choice across episodes with the same invariant signature
- it increases effective control precision without changing canonical state/action alphabets
- all memory is local-first and auditable in state/log files

### 9.2 Algorithm visual (layer interaction)
```text
DISTURBANCE/EPISODE
  -> state update (TMT/drift/slack/alarm)
  -> gene-policy candidate generation
  -> homeostat safety filter (require/veto)
  -> fitness+goals ranking
  -> phenomenology signature check
       if n>=n_min and score>=tau: re-rank within top-K
  -> execute action (+ tighten checks / replan)
  -> log outcome
  -> update phenomenology memory (mid loop)
  -> update NK/gene tuning (slow loop)
```

### 9.3 Sailor analogy (operational mental model)
```text
Watchman spots trouble
  -> Navigator checks charts (feasibility/route)
  -> Engine room exposes possible maneuvers (genes)
  -> Safety officer blocks dangerous moves (homeostat)
  -> First Mate ranks remaining maneuvers (fitness+goals)
  -> Old Sailor checks logbook of similar seas (phenomenology)
       computes one scalar per maneuver:
         phi = p_viable - lambda_procrast*p_procrast - mu_overcontrol*p_overcontrol
         with lambda_procrast > mu_overcontrol
       if enough evidence and phi exceeds threshold: refine choice within approved maneuvers
  -> Crew executes
  -> Logkeeper records outcome
  -> Old Sailor updates logbook; Captain updates doctrine over time (NK tuning)
```

Interpretation of the Old Sailor's scalar:
- `p_viable`: empirical proxy that maneuver restores viable engagement
- `p_procrast`: empirical risk of remaining in procrastinative delay
- `p_overcontrol`: empirical risk of high burden/over-control
- asymmetry (`lambda_procrast > mu_overcontrol`) implements the unequal mean:
  procrastination is weighted as the primary deviation, while anti-coercion guardrails remain active.

Optional FEP-style mode:
- `score = phi - beta_ambiguity*ambiguity + eta_epistemic*epistemic`
- this preserves the same ethical asymmetry while explicitly accounting for uncertainty and controlled exploration.

---

## 10) NK layer: fitness landscape over regulator designs

### 10.1 What NK is over (important)
NK is over a finite, implementable family of controllers produced by choosing discrete knobs in this template.

- Genotype: g (gene vector selecting policy knobs)
- Phenotype: observed trajectory τ under g for a user/context
- Fitness: Φ(g) computed from measurable outcomes and/or ρ-weighted utility

### 10.2 Minimal learnable configuration
Start:
- N = 10 genes
- K = 2 (hypothesis; estimated later)

### 10.3 Genome g (explicit, buildable)
- g = (g1..g10) ∈ Π_i 𝒢_i

Recommended genes (unchanged):
1) g1: block-length map θ→b ∈ {10,25,35,50}
2) g2: check-gap map θ→(Δmin,Δmax) ∈ {(2,5),(4,8),(6,14)}
3) g3: tightening strength ∈ {none,mild,strong}
4) g4: classifier strictness ∈ {lenient,balanced,strict}
5) g5: recovery routing ∈ {E-first,V-first,trigger,mixed}
6) g6: enforce near-reward artifact ∈ {off,on}
7) g7: artifact type ∈ {none,3problems,5flashcards,5bullets}
8) g8: replan threshold ∈ {0,1,2}
9) g9: pause policy ∈ {soft,hard}
10) g10: exploration rate ∈ {off,low,med,high}  (study mode typically uses off/low)

### 10.4 NK fitness decomposition
For user type z:
- Φ_z(g) = (1/N) Σ_{i=1..N} f_{i,z}(g_i, g_{N_i})
with |N_i| = K.

Each f_{i,z} is a small table learned from outcomes.

**Empirical substitution point:** estimate K and interaction neighborhoods N_i using controlled policy perturbations.

---

### 10.5 Homeostat DSL (discrete)
Homeostat rules are expressed over **discrete bins** of essential variables (no continuous math). The DSL *defines acceptable states* and can **veto** bad states.

Default bins:
- slack: neg | low | ok | high
- progress: down | flat | up | surge
- recovery: slow | ok | fast | instant
- annoyance: high | med | low | none
- drift: high | med | low | none
- stability: fragile | wobbly | steady | stable

Rules:
- `require` = **hard veto** (candidates violating are excluded if any candidates satisfy it)
- `penalty` = soft cost (higher is worse)
- `prefer` = soft bonus (lower is better)

Unequal-mean guidance (default):
- encode stronger penalties for procrastination/intent-gap than for moderate annoyance,
- keep anti-coercion guardrails via `require` constraints and pause/quit rights.

Integration:
- The homeostat score is applied **before** fitness-DSL ranking.
- NK exploration minimizes homeostat violations/penalties when selecting genomes.

---

## 11) Fitness: measurable + user-definable (DSL + goal profile)

### 11.1 Measurable outcome features from logs
From trajectory τ:
- J1(τ) = 1[slack ≥ 0]              (feasible)
- J2(τ) = completion / progress rate
- J3(τ) = mean time-to-recover
- J4(τ) = P(miss minimum)
- J5(τ) = P(quit/disable checks)    (annoyance/attrition proxy)
- J6(τ) = mean burden rating        (study mode)
- J7(τ) = perceived helpfulness     (study mode)

Store a signed feature vector, e.g.:
- J(τ) = (J1, J2, -J3, -J4, -J5, -J6, J7)

### 11.2 User-defined fitness without numeric weights
Two compatible mechanisms:

A) **Fitness DSL (prokfit)**: lexicographic priorities over predicates.

B) **Goal profile ρ_t**: a user’s preferred ordering of outcomes + tolerances.

**Combination rule (v5 default):**
- DSL defines system-wide safety/feasibility requirements (e.g., slack != neg).
- ρ_t breaks ties among feasible candidates and limits “annoying” policies.

---

## 12) CCA placeholder (OFF in v5)

CCA is reserved for future personalization linking questionnaire space ↔ regulator parameter space.

v5 requirements:
- Interface exists: `cca_map(features) -> priors`
- Default: `use_cca = false`, and `cca_map` is identity/no-op.
- No runtime behavior may depend on CCA unless explicitly enabled in a later version.

---

## 13) Study mode requirements (v5)

When `study_mode = true`, the CLI must:
- record participant/session identifiers
- prompt for minimal burden/helpfulness ratings after interventions
- record quit reasons (optional)
- export logs + state as a portable bundle

Study mode must be optional and must not block normal use.

---

## 14) Empirical substitution points (roadmap)

Hard-coded in v5, learnable later:
1) Haghbin prompts frequency/strictness → learn from false alarms + outcomes
2) θ update rules (U_θ) → learn transition tables
3) transform selection → contextual bandits / MRT causal estimates
4) check tightening policy → learn annoyance thresholds
5) gene interactions (K) → estimate from controlled perturbations
6) language pack λ_t → user testing + focus group validation
7) ρ_t goal profile → preference elicitation (BWS/paired comparisons), stable scoring in RCQ
8) Recovery transform library 𝒜 (strategies) is elicited from focus groups/diaries (open coding → canonical transform families → domain renderings via λₜ), then refined empirically via logs/experiments.
