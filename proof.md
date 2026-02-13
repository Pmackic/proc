# Proof (Information/Variety Framing)

This document gives a mathematically logical, physics-flavored proof sketch using Ashby's Law of Requisite Variety for PROK v5. The goal is to show that regulator variety must meet or exceed disturbance variety to preserve feasibility.

---

## 1. Formal Model (Discrete-Time)

We consider a discrete-time system with time index `t = 0,1,2,...`.

State variables:
- `s_t`: vector of per-task slack values at time `t`.
  For task `i = 1..N`, define:

```
s_t^i = C_t^i - R_t^i
```

where `C_t^i` is total capacity from time `t` until task `i`'s deadline, and `R_t^i` is the remaining required blocks for task `i`.
- `x_t`: regulator internal state (policy, domain recursion, task bins).

Disturbance and observation:
- `w_t`: external disturbance (triggered drift/procrastination context).
- `u_t`: observation of the plant (self-reports/check responses).

Action:
- `a_t`: regulator action (recovery strategy + check policy).

The essential variable is `s_t` (slack vector). Feasibility is preserved if `s_t >= 0` componentwise for all `t`.

---

## 2. Variety Sets (Ashby)

Define finite sets:
- Disturbance variety: `W = {w_1, ..., w_m}`.
- Regulator action variety: `A = {a_1, ..., a_n}`.

We consider `m = |W|` and `n = |A|`.

---

## 3. Claim (Ashby Requisite Variety for Feasibility)

**Theorem (Requisite Variety Condition).**  
If the regulator can maintain `s_t >= 0` componentwise for all admissible disturbance sequences, then its action variety must satisfy:

```
|A| >= |W_eff|
```

where `W_eff` is the effective disturbance variety that is distinguishable through the observation channel `u_t`.

Equivalently, if `|A| < |W_eff|`, then there exists a disturbance sequence that forces some component of `s_t` negative at some time.

---

## 4. Proof Sketch

We model the regulator as a mapping from observations to actions:

```
f: U -> A
```

Let `h: W -> U` be the observation map. The effective disturbance variety is:

```
W_eff = h(W)
```

Assume `|A| < |W_eff|`. By the pigeonhole principle, there exist distinct disturbances
`w_i != w_j` such that `h(w_i) = h(w_j)` and thus `f(h(w_i)) = f(h(w_j))`.
So the regulator applies the same action to both disturbances.

If there exists at least one pair `(w_i, w_j)` such that the same action cannot
keep all components of the slack vector nonnegative under both cases, then feasibility fails in at least one case.

Since the disturbance variety exceeds the regulator variety, such an indistinguishable
pair exists in the worst case, yielding a trajectory with `s_t < 0`.

Therefore, maintaining componentwise feasibility for all admissible disturbances requires:

```
|A| >= |W_eff|.
```

---

## 5. Mapping to PROK v5

Concrete varieties:
- `W_eff`: combinations of trigger class × drift/procrastination label × context bins.
- `A`: recovery family × check-gap policy × block sizing (from genome bins).

Thus, to avoid feasibility collapse in worst-case contexts, the number of distinct
policy responses must match or exceed the number of distinguishable disturbances.

---

## 6. Physics Interpretation (Entropy/Variety)

Treat the disturbance variety as an entropy budget (in bits):

```
H(W_eff) = log2 |W_eff|
```

and regulator action variety as available control entropy:

```
H(A) = log2 |A|.
```

The theorem implies a thermodynamic-style inequality:

```
H(A) >= H(W_eff).
```

If control entropy is lower than disturbance entropy, the system cannot constrain all
essential variables, and some component of `s_t` must cross the feasibility boundary.

---

## 7. Lemma (PROK v5 Variety Lower Bound)

Define:
- `F` = set of recovery families (from the recovery candidate library).
- `G` = set of check-gap policies (distinct `(min_gap, max_gap)` pairs that can be selected).
- `B` = set of block-size outputs that can be selected for a task.

Then the action variety satisfies:

```
|A| >= |F| * |G| * |B|.
```

**Justification.**  
In PROK v5, a recovery action is specified by a recovery family (which selects the recovery script and updated TMT bins), a check-gap policy, and a block-size recommendation. Distinct combinations yield distinct actions in the Ashby sense. Therefore the total action variety is at least the product of these component varieties.

**Concrete mapping (from code defaults).**  
The recovery candidate generator defines four families: `E_RECOVERY`, `V_RECOVERY`, `G_RECOVERY`, `MIXED_RECOVERY`. The check-gap policies and block-size outputs are drawn from discrete maps in the genome and domain overrides. Thus the bound becomes:

```
|A| >= 4 * |G| * |B|.
```

This gives a simple audit: if the effective disturbance variety `|W_eff|` exceeds the achievable `|A|`, feasibility cannot be guaranteed in the worst case.

---

## 8. Lemma (Effective Disturbance Variety)

Let:
- `T` = set of trigger labels (confusion, boredom, distraction, anxiety, fatigue, other).
- `L` = set of disturbance labels (DRIFT, PROCRASTINATION, BLOCKED).
- `C` = set of context bins (task domain, time-of-day bin, or other coarse bins the observation channel preserves).

Then the effective disturbance variety satisfies:

```
|W_eff| >= |T| * |L| * |C|.
```

**Justification.**  
Each distinct combination of trigger label, disturbance label, and context bin is a distinguishable observation class for the regulator. Therefore the number of effective disturbances is at least the product of these component varieties.

With PROK v5 defaults, `|T| = 6` and `|L| = 3`, so:

```
|W_eff| >= 18 * |C|.
```

Combined with Lemma 7, feasibility guarantees in the worst case require:

```
4 * |G| * |B| >= 18 * |C|.
```

If the context bins are refined, the required action variety increases accordingly.

**Example (concrete `|C|`).**  
If context bins are `domain × time-of-day`, and there are `D` domains and `K` time bins, then:

```
|C| = D * K
```

so the bound becomes:

```
4 * |G| * |B| >= 18 * D * K.
```

**Design implication.**  
Increasing the number of recovery families, diversifying check-gap policies, or adding block-size options increases `|A|` and therefore raises the upper bound on the disturbance variety the regulator can handle without feasibility loss. Conversely, refining context bins or adding new triggers increases `|W_eff|` and should be matched by added policy variety to maintain the inequality.

---

## 9. Bounded Rationality and Maxwell's Demon (Analogy)

**Bounded rationality.**  
The regulator can only distinguish a finite number of disturbance classes and can only execute a finite number of distinct actions. This is a formal statement of bounded rationality: limited variety implies limited control entropy.

**Maxwell's demon analogy.**  
In statistical physics, a demon can lower entropy only if it can *observe* microstates and *act* differentially. Here, the observation channel `u_t` is the demon's sensor, and the action set `A` is its actuator repertoire. If observation collapses distinct disturbances into the same class or if the actuator repertoire is too small, the demon cannot maintain all essential variables (here, slack components) within bounds. The requisite-variety inequality

```
H(A) >= H(W_eff)
```

is the direct analog: without sufficient informational and action capacity, feasibility losses are unavoidable.

---

## 10. Mihajlo Petrović's Mathematical Phenomenology (Analogy)

In the spirit of Petrović's mathematical phenomenology, we treat each disturbance class as a point in a finite phenomenological space `W_eff`, and each admissible action as a point in the control space `A`. The requisite-variety inequality then states that the control space must be at least as rich (in cardinality/entropy) as the phenomenological space it seeks to stabilize.

Put plainly: if the phenomenological classification (disturbance taxonomy) is finer than the control classification (action taxonomy), then the mapping from phenomena to actions necessarily merges distinct mechanisms, and at least one mechanism will be misregulated.

---

## 10.1 Lemma (Phenomenological Refinement Improves Effective Control)

Let:
- `Sigma` be the finite signature space of invariant classes `sigma = SIG(domain, label, trigger, theta)`.
- `M` be a memory map assigning empirical action statistics to each `sigma`.
- `A_top(sigma)` be the top-K fitness-ranked candidate actions for signature `sigma`.

Define the deployed policy:
- baseline: `f0: U -> A` (fitness DSL only)
- refined:  `fM: U -> A`, where `fM(u)` may replace `f0(u)` by an action in `A_top(sigma(u))` if:
  - evidence `n(sigma) >= n_min`
  - empirical score exceeds threshold `tau`.

Then, for every signature class where the conditions hold:
1) action choice remains inside the admissible set `A_top(sigma)` (safety envelope preserved),
2) expected recovery utility is non-decreasing relative to baseline ranking under the empirical score order induced by `M`.

Sketch:
- Constraint (1) follows directly from `fM(u) in A_top(sigma(u))`.
- For (2), replacement occurs only when memory score ranks an alternative above baseline and passes `tau`; otherwise `fM = f0`.
- Therefore refinement is monotone on classes with sufficient evidence and neutral elsewhere.

Design implication:
- phenomenological memory does not require expanding the global action alphabet,
- but increases discrimination of the observation-to-action map by conditioning on learned invariant classes.

---

## 11. Homologija (Model–Observation)

We define a homology-style correspondence between model space and observation space:

```
h: W -> U
```

with `W_eff = h(W)` the effective phenomenological space of observed disturbances.
The regulator acts on `U` through:

```
f: U -> A.
```

The composed map `f ∘ h` is the operational homology between the model's disturbance classes and the regulator's action classes. If `h` collapses distinct phenomena (non-injective), or if `f` collapses distinct observation classes (non-injective), then distinct mechanisms become homologous under control and cannot be separated by action. This is exactly the condition that forces `|A| < |W_eff|` and yields inevitable feasibility loss for some disturbances.

---

## 12. Sailor Metaphor (Operational Interpretation)

To make the control architecture concrete:

- **Navigator (control core)** monitors feasibility/slack and redraws route when needed.
- **First Mate (policy layer)** chooses baseline maneuvers from explicit ship rules (fitness DSL + goals).
- **Old Sailor (phenomenology layer)** consults the logbook of similar wind/current signatures and may switch to a better maneuver only when evidence is sufficient and only within approved top maneuvers.

Interaction cycle:
1) disturbance is observed,  
2) Navigator checks whether ports are still reachable,  
3) First Mate ranks standard responses,  
4) Old Sailor optionally refines choice under evidence gates,  
5) crew executes,  
6) outcome is logged for future signatures.

Meaning in proof terms:
- safety envelope is preserved because refinement is constrained to admissible candidates,
- adaptation improves through local empirical memory,
- Ashby's variety condition remains the governing feasibility constraint.

---

## 13. Aristotelian Ethical Clarification (Unequal Mean)

We model three regulation bands:
- `P`: procrastinative delay band (deficiency of timely action),
- `M`: engaged/flow-capable band (practical mean),
- `O`: over-control burden band (excess pressure/annoyance).

Define an asymmetric ethical loss:

```
L = a * Pr(P) + b * Pr(O) + c * Pr(s_t < 0)
```

with default ordering:

```
a > b >= 0,  c >> a
```

Interpretation:
- `a > b` encodes the **unequal doctrine of the mean**: procrastination is penalized more strongly than moderate burden,
- `c >> a` preserves feasibility primacy (slack violations dominate all soft tradeoffs).

Proposition (asymmetric-center policy):
If two admissible policies have equal feasibility risk and one has lower `Pr(P)` but slightly higher `Pr(O)`, then it is ethically preferred whenever:

```
a * DeltaPr(P) > b * DeltaPr(O).
```

This formalizes a procrastination-focused but non-coercive center:
- not maximal intensity,
- not permissive delay,
- but a biased mean toward timely engaged activity, constrained by hard viability bounds.
