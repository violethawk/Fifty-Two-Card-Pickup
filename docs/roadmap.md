# 52 Card Pickup — Product Roadmap
## A Multi-Agent Systems Hello World

---

## Vision

Establish the canonical "hello world" for multi-agent LLM systems — a project simple enough that anyone can understand in 5 minutes, deep enough to teach every concept that matters, and extensible enough to grow into a real orchestration playground.

The 52 Card Pickup metaphor maps directly to the three fundamentals of multi-agent systems:
1. **Define agent roles** — scatter, pick up, track time, verify
2. **Define agent interactions** — sequencing, shared state, coordination
3. **Set constraints** — 52 cards must all be accounted for, no duplicates, no losses

---

## Phase 0 — Foundation

**Intent:** Project setup and architecture decisions established.

**Deliverables:**
- Repository initialized with README, LICENSE, and project structure
- LangGraph selected and validated as orchestration framework
- Shared state schema defined and documented
- Development environment reproducible (requirements.txt / pyproject.toml)
- Phase 1 implementation prompt written (Tier 2 semi-formal)

**GO signal:** Repository exists, dependencies install cleanly, prompt is reviewed.

---

## Phase 1 — Deterministic Multi-Agent Orchestration

**Intent:** Prove the multi-agent pattern works with zero LLM involvement.

**Deliverables:**
- 4 deterministic agents: Scatter, Pickup, Timer, Verifier
- LangGraph state graph with defined node sequencing
- Single-agent pickup working end-to-end with verifier passing
- Scaling experiment: 1, 2, and 4 pickup agents with timing comparison
- Summary table output showing scaling results
- All agents operate on shared state with no side channels

**Key Concepts Demonstrated:**
- Agent role definition
- Shared state as coordination mechanism
- Sequential and parallel execution
- Constraint verification (verifier agent)
- Scaling behavior under task decomposition

**GO signal:** Verifier passes 100% of runs across all agent configurations. Summary table shows measurable time improvement with additional agents.

---

## Phase 2 — LLM-Powered Supervisor

**Intent:** Introduce intelligent orchestration while keeping worker agents deterministic.

**Deliverables:**
- Supervisor agent powered by an LLM (Claude or GPT-4)
- Supervisor observes the scatter pattern and decides:
  - How many pickup agents to deploy (1, 2, or 4)
  - How to divide the grid among agents (spatial partitioning strategy)
- Worker agents remain deterministic — the LLM only makes strategic decisions
- Supervisor explains its reasoning in natural language
- Comparison: supervisor's choices vs. brute-force configurations from Phase 1

**Key Concepts Demonstrated:**
- Hybrid architecture: intelligent supervisor + deterministic workers
- LLM as decision-maker, not executor
- Strategic reasoning about resource allocation
- Human-readable orchestration rationale

**GO signal:** Supervisor makes reasonable deployment decisions that match or beat the average brute-force configuration. Reasoning is coherent and auditable.

---

## Phase 3 — LLM-Powered Pickup Agents

**Intent:** Give agents intelligence and observe coordination challenges.

**Deliverables:**
- Pickup agents powered by LLMs with spatial awareness
- Agents can strategize: prioritize clusters, optimize paths, avoid conflicts
- Conflict resolution: when two agents target the same card, a protocol resolves it
- Agent communication channel: agents can broadcast intentions ("I'm heading to the northeast quadrant")
- Comparison: LLM-powered agents vs. deterministic agents from Phase 1

**Key Concepts Demonstrated:**
- Agent autonomy and emergent behavior
- Conflict detection and resolution
- Inter-agent communication protocols
- Tradeoff: intelligence vs. coordination overhead vs. cost

**GO signal:** LLM-powered agents complete pickup successfully. Conflict resolution works without deadlocks. Cost-per-run documented.

---

## Phase 4 — Observability and Governance

**Intent:** Add the monitoring and quality infrastructure that production multi-agent systems need.

**Deliverables:**
- Real-time dashboard showing agent positions, card states, and progress
- Event log: every agent action recorded with timestamp and state snapshot
- Replay capability: re-run any session from its event log
- Performance metrics: cards per second per agent, idle time, conflict rate
- Anomaly detection: flag runs where agents behave unexpectedly
- Governance layer: rules that halt execution if invariants are violated (e.g., card count changes mid-run)

**Key Concepts Demonstrated:**
- Multi-agent observability
- Audit trails for agent behavior
- Governance constraints as runtime guardrails
- The difference between monitoring (what happened) and governance (what's allowed)

**GO signal:** Every run produces a complete, replayable event log. Dashboard shows live progress. Governance layer catches injected faults.

---

## Phase 5 — Extensibility and Teaching

**Intent:** Package the project as a teaching tool and extensibility platform.

**Deliverables:**
- Tutorial series: one document per phase, building incrementally
- "Add Your Own Agent" guide: how to create a 5th agent and wire it into the graph
- Benchmark suite: standardized scatter patterns for reproducible comparisons
- Plugin architecture: swap in different LLM providers, different strategies, different coordination protocols
- Conference talk / blog post: "52 Card Pickup: The Multi-Agent Hello World"

**Key Concepts Demonstrated:**
- How to teach multi-agent systems progressively
- Extensibility as a design goal
- Benchmarking and reproducibility in multi-agent research

**GO signal:** A developer with no multi-agent experience can complete Phase 1 from the tutorial in under 2 hours.

---

## Cross-Phase Principles

**Simplicity over sophistication.** Every phase must remain understandable to someone who hasn't done the previous phases. If a concept can't be explained with cards on a floor, it doesn't belong here.

**Correctness over performance.** The verifier agent runs after every phase, every configuration, every experiment. If 52 cards don't come back intact, nothing else matters.

**Teach the pattern, not the framework.** LangGraph is the implementation choice but the concepts transfer to any orchestration framework. Documentation always explains the "why" before the "how."

**Progressive complexity.** Phase 1 has no LLMs. Phase 2 has one. Phase 3 has many. Phase 4 adds monitoring. Phase 5 packages it. Each phase adds exactly one layer of complexity.

---

## Relationship to the 3 Tier Coding Method

This roadmap was built using the method:

- **Tier 1 (Informal):** The 52 Card Pickup concept emerged from informal conversation about what a multi-agent hello world should look like.
- **Tier 2 (Semi-formal):** The Phase 1 implementation prompt was engineered before any code was written. Each subsequent phase will get its own Tier 2 prompt before implementation begins.
- **Tier 3 (Formal):** As the project matures, recurring quality checks (verifier logic, benchmark runs, governance rules) will be formalized into versioned prompts that run throughout development.

The project itself is a demonstration of the method that produced it.
