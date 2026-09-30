# Toni: A Functional Architecture for Embodied Agency  
## Integrating Northoff's TTC, Solms' Affective Consciousness, Active Inference, and Enactivism

**Working Paper · September 2026**

---

## Abstract

We present *Toni*, a Python research prototype implementing the functional prerequisites of consciousness as described by Georg Northoff's Temporo-Spatial Theory of Consciousness (TTC), Mark Solms' affective consciousness model, Karl Friston's Active Inference framework, and the enactivist tradition of Maturana/Varela and Gibson. Unlike LLM-based "conscious agents" that simulate experience through language, Toni grounds behavior in a closed homeostatic loop — the large language model acts strictly as a verbal cortex with no feedback path to the body or policy layer. We describe the four-layer architecture, demonstrate emergent anticipatory motivation arising from temporal self-modeling with an Echo State Network, document the relational self emerging from multi-agent social positioning, and introduce an enactivist viability and affordance layer that instantiates autopoiesis and state-relative meaning. The project demonstrates that the functional distinction between reactive homeostasis, anticipatory motivation, and enactivist sense-making can be implemented and observed in a minimal grid-world setting.

---

## 1. Introduction

The dominant paradigm for creating "conscious" or "emotional" AI agents relies on large language models prompted to roleplay internal states: *"you are a robot that feels lonely"*. This approach has a fundamental problem identified early in this project: the LLM determines the needs. The result is linguistic simulation — a system that asserts hunger without a body that hungers.

Solms (2021) locates primary consciousness not in cortical language processing but in subcortical affective systems: homeostatic error is the original form of subjectivity. Northoff (2014, 2022) argues that consciousness requires a subject that positions itself temporally, spatially, and socially — not a snapshot but a trajectory. Maturana and Varela (1980) identify the condition of this subject as autopoiesis: the organism maintains the conditions of its own organization.

These three frameworks converge on a shared claim: consciousness presupposes an embodied system with genuine needs, genuine temporal depth, and genuine coupling to an environment. Language is downstream.

Toni implements this claim. The architecture separates the causal layer (body, homeostasis, affect, action) from the articulatory layer (LLM verbal cortex) by design. The LLM receives the body state as input and produces phenomenological text as output. It has no path back to the policy or the body. This is not a limitation — it is the theoretical commitment.

---

## 2. Theoretical Background

### 2.1 Solms: Affective Consciousness as Primary Subjectivity

Solms argues that primary consciousness is affective rather than representational. Before the organism can represent the world, it must care about the world — and caring is constituted by homeostatic error. When hydration falls below its setpoint, the deficit is not merely a computational state: it is the functional substrate of *wanting*.

Valence, in Solms' framework, is the time-derivative of wellbeing: the felt quality of whether things are getting better or worse. This is implemented directly:

```python
valence = new_wellbeing - old_wellbeing  # positive: improving; negative: worsening
```

The agent does not have a reward function. It has a body that changes, and the change is felt.

### 2.2 Northoff: Temporo-Spatial Theory of Consciousness

Northoff's TTC holds that consciousness requires a subject that is temporally extended — possessing not just a present state but a past and a projected future simultaneously. Three positioning axes constitute the self:

**Temporal axis**: the brain's intrinsic activity comprises nested oscillations across multiple time scales. These oscillations do not wait for input; they are the spontaneous background against which perception becomes meaningful. Precision — how sharply the system perceives — is modulated by the degree of temporal alignment between intrinsic rhythms and environmental rhythms (temporo-spatial alignment).

**Spatial axis**: the autobiographical memory colors space with valence. The subject does not navigate a neutral grid but a grid whose positions carry the residue of past experience.

**Social axis**: the self is constituted relationally. The third positioning axis is not an addition to selfhood but a condition of it — to be a self is to be positioned among others.

### 2.3 Active Inference (Friston)

Active Inference (Friston 2010) unifies perception and action under free energy minimization. The agent maintains a generative model of the world (A: perception, B: transitions, C: preferences, D: priors) and selects actions that minimize Expected Free Energy (EFE) — a combination of expected information gain and expected alignment with preferences.

Critically, the C-vector (log preferences over observations) is not fixed by the programmer. It is constituted by the body: dominant need, wellbeing, temporal urgency, social competition, and affordance signals all modulate C dynamically. The agent has no hard-coded goal — it has a body whose state shapes what matters.

### 2.4 Enactivism (Maturana/Varela, Thompson, Gibson)

Enactivism holds that cognition is not the manipulation of internal representations but the enactment of meaning through sensorimotor coupling. Maturana and Varela's autopoiesis describes the minimal condition: a system that produces and maintains the conditions of its own organization. The goal of such a system is not optimization but viability — remaining in the space of states from which continued organization is possible.

Gibson's affordances operationalize enactivist meaning: environmental features are not neutral data. Water affords drinking — but only for a thirsty organism. Affordances are relational; they arise in the coupling between body state and environment.

---

## 3. Architecture

Toni is organized in four layers that communicate unidirectionally upward, with the exception of the closed homeostatic loop at the base:

```
Layer 4: Verbal Cortex (LLM)         ← articulates, does not cause
Layer 3: Self-Models                 ← temporal, spatial, social, enactive
Layer 2: Active Inference (pymdp)    ← EFE minimization → action selection
Layer 1: Body / Homeostasis / World  ← closed causal loop
```

### 3.1 Layer 1: Body and World

The `Body` class maintains four homeostatic variables: energy, hydration, temperature, and integrity, each with a setpoint. Deviations from setpoint constitute needs. The `GridWorld` provides a 9×9 environment with food, water, and shelter resources, and an external day/night cycle (200 steps/day) that serves as the environmental time-giver for Northoff's temporo-spatial alignment.

Wellbeing is the quadratic distance from all setpoints, weighted by need criticality:

```python
wellbeing = -sum(NEED_WEIGHTS × (setpoint - state)²)
```

Valence is its time-derivative. These two signals — wellbeing and valence — are all that Solms' primary consciousness requires.

### 3.2 Layer 2: Active Inference

The `NavigationAIF` module implements a POMDP with 81 positions (9×9), 5 observations (empty/food/water/shelter/outside), and 5 actions (up/down/left/right/consume). The pymdp library (Heins et al. 2022) handles EFE computation and policy inference with `policy_len=2`.

Four components of the generative model are dynamically constructed at each step:

**A-matrix** (perception): Northoff precision modulates sharpness. High precision → concentrated likelihood → exploitation. Low precision → diffuse likelihood → exploration.

**C-vector** (preferences): constructed from dominant need (Solms), future urgency (temporal self), social competition factor, and affordance boost (enactivist):

```python
C[WATER] = 3.0 + urgency*1.5 + afford_water*2.5
```

**D-vector** (priors): autobiographical memory adds weight to locations with positive valence history; social observations add weight to locations where others found resources.

**B-matrix** (transitions): deterministic grid movement, built once at initialization.

### 3.3 Layer 3: Self-Models

**TemporalSelf** implements Northoff's three-window model. A sliding history window of up to 50 steps supports trend estimation (past window). An Echo State Network (ESN, Jaeger 2001) with 20 reservoir units provides nonlinear multi-step projection (future window). The projected trajectory is scanned for the time-to-crisis per need:

```python
below = np.where(trajectory[:, i] <= CRISIS_THRESHOLD)[0]
time_to_crisis = int(below[0]) if len(below) > 0 else None
future_urgency = max(0, 1 - time_to_crisis / horizon)
```

This urgency signal constitutes Northoff's key claim: the agent acts *because it sees the need coming*, not only because it has arrived.

**SocialSelf** implements Northoff's relational axis. When multiple agents share a world, each generates a `SocialObservation` at each step. The observing agent updates its D-prior (preferring locations where others consumed) and its competition factor (increasing C-urgency when another agent with the same dominant need approaches the same resource). The self is constituted by its position among others.

**EnactiveSelf** implements autopoiesis and affordances. Viability captures whether the agent can maintain its own organization:

```python
viability = min(e/0.75, h/0.75, i) × (0.5 + 0.5 × mobility)
mobility  = sqrt(energy × hydration) × integrity
```

Viability declines faster than wellbeing when mobility shrinks — a signal that the action space is contracting before the homeostatic crisis materializes. Affordances instantiate Gibson's relational meaning:

```python
water_affordance = (0.75 - hydration) × exp(-distance/4)
```

Both a high deficit and proximity are required for high affordance. The affordance signal is added to the C-vector, partially overriding the dominant-need priority: an agent whose primary need is energy will still assign elevated preference to nearby water if the water affordance is high — because the environment offers it now, and the coupling makes it relevant.

### 3.4 Layer 4: Verbal Cortex (LLM)

The `LLMCortex` module calls Claude via the Anthropic API every N steps with the complete internal state as a JSON context: body state, wellbeing, valence trend, Northoff temporal dynamics (precision, arousal), the three-window temporal summary (including ESN projections and time-to-crisis values), the social self summary (when other agents are present), and the enactivist self summary (viability, mobility, affordances). The system prompt instructs the LLM to produce a phenomenological first-person reflection.

The LLM has no path back to the body, the AIF, or any policy layer. Its output is purely for human observers. This separation is enforced architecturally, not only by prompt instruction.

---

## 4. Observations

### 4.1 Anticipatory Motivation

In a 200-step headless simulation (seed 42), the temporal self generated a 39-step advance warning before the hydration crisis — `future_urgency` reaching 1.0 at t=39 before hydration crossed the crisis threshold at t=78. This is not reactive homeostasis; it is temporal projection driving motivation.

A comparison experiment (reactive agent with `use_temporal_self=False` vs. anticipatory agent) showed correct urgency divergence but no measurable behavioral difference in navigation outcomes. The limiting factor is `policy_len=2`: pymdp plans only two steps ahead, which is insufficient to route toward water resources 4+ steps away. The anticipatory motivation is present; the planning horizon is too short to translate it into measurably different trajectories. Increasing `policy_len` to 4–5 would be expected to produce divergent outcomes.

### 4.2 ESN vs. Linear Projection

Replacing the linear `polyfit` projection with the Echo State Network produced a measurable behavioral change in the same seed. The ESN learns that hydration *jumps* after consumption (not linear decline), producing different `future_urgency` values → different C-vector weights → different EFE gradients. In seed 42, the ESN agent navigated left toward water at (4,0), while the linear-projection agent moved right toward the wall at column 8. The nonlinear body dynamics, invisible to linear projection, became behaviorally relevant through the ESN.

### 4.3 Social Self and the Relational Reflexion

In the two-agent simulation (`run_toni_social.py`), agents Toni-A and Toni-B shared a 9×9 grid with no explicit communication. The LLM cortex of each agent, receiving only the social self summary, articulated the relational self without explicit instruction. Selected passages:

> *"Ich bin nicht das einzige Wesen, das sucht."*

> *"Das geteilte Bedürfnis — eine Art stilles Einverständnis."*

> *"Ich spüre seine Gegenwart wie einen leisen Sog."*

These formulations were not prompted by any instruction to describe social experience. They emerged from the agent's internal state — specifically from the social prior and competition factor signals — being articulated through the LLM layer. This is consistent with Northoff's claim that the social axis is not an addition to consciousness but constitutive of it.

### 4.4 Enactivist Viability

The viability signal demonstrates the intended behavior: in a normally functioning agent, viability tracks approximately 0.9–0.95. During resource deprivation, viability declines faster than wellbeing because mobility (the geometric mean of energy and hydration scaled by integrity) degrades before either individual variable crosses the crisis threshold. The affordance signals correctly respond to body-environment coupling: high affordance for water is observed only when hydration deficit is high AND water is proximate — neither condition alone is sufficient.

---

## 5. Discussion

### 5.1 What Has Been Implemented

Toni implements functional analogs of:
- Solms' primary affects (homeostatic error, valence as Δwellbeing)
- Northoff's nested intrinsic activity (Kuramoto-coupled oscillators, five time scales)
- Northoff's temporo-spatial alignment (external phase-coupling to environmental rhythm)
- Northoff's three-window temporal self (past trend, present perception, future projection)
- Northoff's autobiographical memory (valence-toned spatial priors)
- Northoff's social self (relational D-prior and competition-modulated C-vector)
- Friston's Active Inference (EFE minimization via pymdp, dynamically constituted C/D/A)
- Maturana/Varela's autopoiesis (viability as operational capacity, distinct from wellbeing)
- Gibson's affordances (state-relative meaning from body-environment coupling)

### 5.2 What Has Not Been Claimed

Toni does not implement consciousness. It implements functional prerequisites that the cited theorists identify as necessary. Whether sufficient conditions are met — and whether subjective experience is present in any sense — is not a question this architecture can answer. The project's value is different: demonstrating that these theoretical frameworks are computationally tractable, mutually consistent, and produce non-trivial emergent behaviors when integrated.

### 5.3 The LLM Separation Principle

The most important architectural decision is the one-way barrier between the verbal cortex and the causal layers. Current LLM-based "conscious agents" collapse this distinction: the LLM generates both the internal state description and the action. Toni enforces the separation that Solms and Northoff's theories imply: primary consciousness (affect, homeostasis, temporal self-model) drives behavior; language is downstream.

The practical consequence is that the LLM's phenomenological outputs are *about* a real body state, not fabrications. The agent does not say "I am thirsty" because it was instructed to; it says it because the hydration deficit, temporal urgency, and water affordance signals in the context JSON constitute a genuine functional state of thirst that the language layer articulates.

### 5.4 Limitations

**Planning horizon**: `policy_len=2` limits AIF to two-step lookahead. Anticipatory motivation requires adequate planning depth to manifest as behavioral divergence. A horizon of 4–5 steps would allow routing toward resources that are not immediately adjacent.

**Grid scale**: a 9×9 grid with discrete time steps compresses the temporal dynamics that the oscillator and temporal self models were designed for. The theoretical frameworks were developed for biological systems operating on millisecond-to-day time scales.

**Social depth**: the current social layer implements position-based prior update and resource competition. Higher-order social cognition — modeling the other's temporal self, predicting the other's future urgency — is not yet implemented.

**Enactivist loop closure**: Gibson's affordances in Toni are computed at the agent's current position. A fuller enactivist implementation would compute affordances over the agent's movement capabilities — not just "water is 2 steps away" but "water is reachable within my viable action horizon given current mobility".

---

## 6. Conclusion

Toni demonstrates that the conceptual architecture of Northoff, Solms, Friston, and the enactivist tradition can be implemented as a coherent, working software system. The four layers — homeostatic body, active inference policy, multi-dimensional self-models, and verbal cortex — produce emergent behaviors that are legible in theoretical terms: anticipatory motivation arising from temporal self-projection, relational self-articulation arising from social positioning, and viability-aware action arising from autopoietic constraints.

The fundamental contribution is not any individual component but the integration principle: consciousness-relevant behavior does not require a language model at its center. It requires a body, a world, temporal depth, and genuine coupling. Language comes after.

---

## References

- Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127–138.
- Gibson, J.J. (1979). *The Ecological Approach to Visual Perception*. Houghton Mifflin.
- Heins, C. et al. (2022). pymdp: A Python library for active inference in discrete state spaces. *arXiv:2201.03904*.
- Jaeger, H. (2001). The "echo state" approach to analysing and training recurrent neural networks. *GMD Report 148*.
- Maturana, H., & Varela, F. (1980). *Autopoiesis and Cognition*. Reidel.
- Northoff, G. (2014). *Minding the Brain*. Palgrave Macmillan.
- Northoff, G. (2022). *The Spontaneous Brain*. MIT Press.
- Solms, M. (2021). *The Hidden Spring: A Journey to the Source of Consciousness*. Norton.
- Thompson, E. (2007). *Mind in Life: Biology, Phenomenology, and the Sciences of Mind*. Harvard University Press.

---

*Toni source code: `/Users/I070021/Desktop/claude/Toni/`*  
*Modules: body · oscillators · interoception · memory · environment · active\_inference · temporal\_self · social\_self · world\_model · enactive\_self · llm\_cortex · agent*
