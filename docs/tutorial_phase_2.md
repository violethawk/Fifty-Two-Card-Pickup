# Phase 2 Tutorial: LLM-Powered Supervisor

## What You'll Learn

- The hybrid architecture: intelligent supervisor + deterministic workers
- How to use an LLM as a decision-maker, not an executor
- How to provide structured data to an LLM for reasoning
- How to compare LLM decisions against brute-force baselines

## Prerequisites

- Completed Phase 1 tutorial
- An Anthropic API key (set `ANTHROPIC_API_KEY` environment variable)
- `pip install anthropic` (already in requirements.txt)

## The Pattern

Phase 1 hardcodes the number of pickup agents (1, 2, or 4). But which is best? It depends on how the cards are scattered.

Phase 2 adds a **supervisor** — a single LLM-powered agent that looks at the scatter pattern and decides how many workers to deploy. The workers themselves stay deterministic. This is the **hybrid pattern**: one intelligent agent orchestrating several simple ones.

This is the most practical pattern in production multi-agent systems today. LLMs are expensive and slow. Use them for decisions that require reasoning, not for repetitive execution.

## The Supervisor Node

The supervisor sits between `scatter` and `timer_start` in the graph:

```
START -> scatter -> supervisor -> timer_start -> pickup -> timer_stop -> verify -> END
```

It reads the scattered card positions, computes spatial metrics, sends them to Claude, and gets back a decision.

### Spatial Analysis (`_analyze_scatter`)

Instead of dumping 52 card positions into the LLM prompt, we compute meaningful metrics:

- **Quadrant counts**: How many cards in each quarter of the grid
- **Left/right split**: Balance between halves
- **Spatial spread**: Standard deviation of x and y coordinates
- **Nearest-neighbor distance**: Average distance to the closest card
- **Balance ratio**: min(quadrant) / max(quadrant)

These metrics give the LLM the information it needs to reason about parallelism without overwhelming it with raw data.

### The Prompt

The supervisor's system prompt (line ~198) explains the tradeoffs:

- 1 agent: best for clustered cards
- 2 agents: good for left/right spread
- 4 agents: good for even distribution

It asks for a JSON response with `agents` (the count) and `reasoning` (why).

### Fallback

If the API call fails, the supervisor defaults to 2 agents. Always have a fallback — LLM calls can fail for many reasons (network, rate limits, invalid responses).

## Running It

```bash
export ANTHROPIC_API_KEY=your-key-here
python card_pickup.py --phase 2
```

The comparison shows the supervisor's choice against brute-force:

```
| Trial | Supervisor Choice | Supervisor Time | Best Brute-Force | Match? |
|-------|-------------------|-----------------|------------------|--------|
| 1     | 4 agents          | 0.1541s         | 4 agents 0.1475s | Yes    |
```

## Key Concept: LLM as Decision-Maker

The supervisor doesn't pick up any cards. It makes one decision ("how many agents?") and explains why. This is the sweet spot for LLMs in multi-agent systems:

- **High impact**: The decision affects the entire run
- **Low frequency**: One call per run (cheap)
- **Auditable**: The reasoning is stored in state and can be reviewed

Contrast this with having the LLM pick up each card individually (Phase 3) — that's 52+ calls, much more expensive, and the reasoning matters less per-call.

## Exercises

1. **Test with benchmarks**: Run `python card_pickup.py --benchmark` and check if the supervisor would make the right call for each pattern
2. **Modify the prompt**: Add a guideline for when 2 agents is optimal. Does the supervisor start picking 2?
3. **Change the model**: Try using Haiku instead of Sonnet for the supervisor. Is the reasoning quality noticeably different?
4. **Add a metric**: Compute a new spatial metric (e.g., average distance from center) and add it to the analysis. Does the supervisor use it?

## Next

[Phase 3: LLM-Powered Pickup Agents](tutorial_phase_3.md) — Give the pickup agents intelligence and watch coordination challenges emerge.
