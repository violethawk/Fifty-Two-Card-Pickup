# 52 Card Pickup: The Multi-Agent Hello World

Every programming discipline has its "hello world" — the simplest possible example that demonstrates the core concepts. Web development has the todo app. Machine learning has MNIST. But multi-agent systems? The examples are either too trivial (two chatbots talking) or too complex (autonomous research teams with 15 agents and a PhD's worth of configuration).

We built 52 Card Pickup to fill that gap.

## The Metaphor

Imagine 52 playing cards scattered on a floor. You need to pick them all up. With one person, you pick up everything yourself. With two, you split the room. With four, you each take a quadrant. Simple enough that a five-year-old understands the problem. Deep enough that it maps to every fundamental challenge in multi-agent coordination.

The metaphor gives us:
- **Agent roles** — scatter, pick up, time, verify
- **Shared state** — the cards on the floor, who picked up what
- **Coordination** — who covers which area, what happens when two people reach for the same card
- **Constraints** — 52 cards must come back intact, no duplicates, no losses

## Five Phases, Five Lessons

### Phase 1: No LLMs at All

The first version uses pure Python functions as agents, orchestrated by LangGraph. No LLM, no AI, no API keys. Four agents (Scatter, Pickup, Timer, Verifier) operate on shared state. The pickup agent uses greedy nearest-neighbor search. A scaling experiment compares 1, 2, and 4 agents.

**Lesson**: The orchestration pattern, shared state, and verification work without any intelligence. This is the foundation.

### Phase 2: One LLM, Many Workers

A Claude-powered Supervisor agent analyzes the scatter pattern and decides how many pickup agents to deploy. The workers stay deterministic. The supervisor receives spatial metrics (quadrant density, spread, nearest-neighbor distance) and returns a decision with reasoning.

**Lesson**: The hybrid architecture — intelligent supervisor, deterministic workers — is the most practical multi-agent pattern today. Use LLMs for decisions that require reasoning, not for repetitive execution.

### Phase 3: Intelligent Agents, Real Conflicts

The pickup agents get their own LLM (Haiku, for cost efficiency). Each agent plans moves, broadcasts intentions, and resolves conflicts when two agents target the same card. A round-based loop coordinates the process: plan, broadcast, resolve, execute, repeat.

**Lesson**: Agent autonomy creates coordination challenges. Conflicts are not bugs — they're the fundamental problem. The resolution protocol (closest wins, deterministic fallback) is simple but instructive. And intelligence is expensive: LLM agents are 800x slower than deterministic ones for this problem.

### Phase 4: You Can't Manage What You Can't Measure

Event logging records every action. Governance checks enforce invariants after every round. Performance metrics quantify throughput, efficiency, and conflict rate. Anomaly detection flags unexpected behavior. A terminal dashboard shows live progress.

**Lesson**: Observability and governance are not optional extras — they're requirements for any system where agents act autonomously. The distinction between monitoring (what happened) and governance (what's allowed) is critical.

### Phase 5: Make It Teachable

A benchmark suite with six standardized scatter patterns enables reproducible comparisons. A plugin architecture lets you swap LLM providers and pickup strategies without modifying core code. Tutorials walk through each phase. You're reading the result.

**Lesson**: A good example is one that others can learn from, extend, and build on.

## What We Learned

### The Intelligence Tradeoff Is Real

For 52 cards on a 10x10 grid, greedy nearest-neighbor is hard to beat. The LLM agents demonstrate genuine spatial reasoning — they identify clusters, avoid conflicts, plan efficient paths — but each decision costs 2 seconds of API latency. Intelligence has a price, and the problem has to be complex enough to justify it.

### Hybrid Architectures Win

The most useful configuration isn't "all LLM" or "no LLM." It's one smart supervisor making strategic decisions, with deterministic workers executing efficiently. This maps directly to real-world systems: an LLM decides what to do, a code pipeline does it.

### Verifier-First Development

The Verifier agent runs after every experiment, every configuration, every trial. If 52 cards don't come back intact, nothing else matters. This constraint-first approach catches bugs immediately and builds confidence in the system as it grows more complex.

### Benchmarks Reveal the Truth

The benchmark suite shows that different scatter patterns favor different configurations. Clustered cards? One agent wins. Evenly spread? Four agents win. The supervisor's job is to recognize this — and the benchmarks are how we verify it does.

## Try It Yourself

```bash
git clone https://github.com/violethawk/Fifty-Two-Card-Pickup.git
cd Fifty-Two-Card-Pickup
pip install -r requirements.txt

# Phase 1 — no API key needed
python -m card_pickup --phase 1

# Benchmarks — all patterns, all configs
python -m card_pickup --benchmark

# Full suite — needs ANTHROPIC_API_KEY
python -m card_pickup
```

The code is designed to be read, modified, and extended. Add your own agent (there's a guide). Swap in your own LLM provider (there's an interface). Create your own scatter pattern (there's a template). Run the benchmarks and see what wins.

52 cards. N agents. One shared floor. The rest is coordination.
