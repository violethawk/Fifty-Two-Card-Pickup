# Phase 3 Tutorial: LLM-Powered Pickup Agents

## What You'll Learn

- Agent autonomy: each agent plans its own moves
- Conflict detection and resolution between agents
- Inter-agent communication via broadcast intentions
- The cost of intelligence: API calls, latency, and dollars

## Prerequisites

- Completed Phase 2 tutorial
- An Anthropic API key

## The Shift

In Phases 1-2, agents had fixed regions and couldn't collide. Phase 3 removes the boundaries. All agents share the entire grid, choose their own targets, and can conflict when two agents want the same card.

This is where multi-agent coordination gets interesting.

## The Round-Based Loop

The pickup phase becomes a multi-round process. Each round has four steps:

### Step 1: Plan

Each agent calls Claude Haiku with:
- Its current position
- All remaining unpicked cards (with coordinates)
- Other agents' positions and announced intentions

The LLM returns a batch of 5-10 target cards in priority order, plus a strategy explanation.

### Step 2: Broadcast

Each agent announces its first-choice target to all other agents. This is the **intention** — what the agent will try to do this round.

### Step 3: Resolve Conflicts

If two agents target the same card, a **deterministic** protocol resolves it:
- The agent closest to the card wins
- The loser falls back to their next planned target
- If all targets conflict, the agent waits

No LLM calls here — conflict resolution is a pure function. This is deliberate: you don't want to pay for an API call to decide who gets to pick up the 7 of hearts.

### Step 4: Execute

Each agent with a resolved target travels to the card (simulated travel time) and picks it up.

Then repeat until all 52 cards are collected.

## Cost Awareness

Phase 3 uses Haiku (not Sonnet) because pickup agents make many calls:
- 4 agents x ~13 rounds = ~52 LLM calls per run
- At Haiku pricing: ~$0.08 per run
- Compare to Sonnet: ~$0.62 per run

The choice of model is a design decision. Pickup agents don't need deep reasoning — they need fast spatial planning. Haiku is the right tool.

## Running It

```bash
export ANTHROPIC_API_KEY=your-key-here
python card_pickup.py --phase 3
```

Output includes per-agent strategies and conflict counts:

```
LLM: 4 agents, 124.44s, 52 calls, 19 conflicts, $0.0786
Brute-force best: 0.15s
```

The LLM agents are ~800x slower than deterministic ones. That's the tradeoff: intelligence costs time and money.

## Key Concept: Intelligence vs Overhead

The deterministic greedy approach picks the nearest card every time. It's fast, cheap, and works well. The LLM agents can reason about clusters, avoid conflicts, and plan ahead — but each decision costs ~2 seconds of API latency.

For 52 cards on a 10x10 grid, the greedy approach is hard to beat. But imagine a larger problem: 10,000 items across a warehouse with obstacles, varying priorities, and dynamic constraints. That's where LLM planning starts to win.

The lesson: **match the intelligence to the problem complexity**. Don't use an LLM when a for-loop will do.

## Key Concept: Conflict as Coordination

Conflicts are not bugs — they're the fundamental coordination challenge. When agents independently decide what to do, they sometimes want the same thing. The system needs a protocol to resolve this.

Our protocol is simple: closest agent wins. But you could design others:
- **Priority-based**: agents with fewer cards get priority
- **Auction**: agents bid on cards based on their utility
- **Negotiation**: agents exchange messages to agree (more LLM calls)

Each protocol trades off fairness, efficiency, and cost.

## Exercises

1. **Use the mock provider**: Run with `--provider mock` to see how deterministic the LLM strategy is without actual LLM calls
2. **Count conflicts per pattern**: Run the clustered benchmark — do 4 agents conflict more when cards are in one corner?
3. **Change conflict resolution**: Modify `_resolve_conflicts` so the agent with *fewer* total pickups wins. Does this balance workload better?
4. **Single LLM agent**: Run Phase 3 with 1 agent. There are no conflicts — is the LLM's path better than greedy nearest-neighbor?

## Next

[Phase 4: Observability and Governance](tutorial_phase_4.md) — Add monitoring, guardrails, and a live dashboard.
