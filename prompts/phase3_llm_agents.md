# Phase 3 Implementation Prompt — LLM-Powered Pickup Agents

## Tier 2 (Semi-Formal)

---

## Context

Phase 2 is complete. The system has a deterministic LangGraph pipeline with an LLM-powered supervisor (Claude Sonnet) that decides how many pickup agents to deploy. Worker agents use greedy nearest-neighbor search within fixed grid regions. Simulated travel cost (0.005s per distance unit) makes parallelism meaningful.

Phase 3 replaces the deterministic pickup logic with LLM-powered agents that choose their own targets, communicate intentions, and resolve conflicts when two agents want the same card.

## Objective

Replace the fixed-region greedy pickup with an LLM-driven pickup loop where:

1. Agents share the entire grid — no fixed region boundaries.
2. Each agent uses Claude Haiku to plan its next batch of moves (5-10 cards at a time).
3. Agents broadcast their intentions before moving.
4. A deterministic conflict resolution protocol handles collisions.
5. Agents adapt their plans based on what other agents announce.

## Architecture: Round-Based Pickup Loop

The pickup phase becomes a multi-round loop. Each round:

### Step 1 — Plan
Each agent receives:
- Its current position (x, y)
- The list of all remaining unpicked cards (with positions)
- Other agents' current positions
- Other agents' most recent broadcast intentions

Each agent calls Claude Haiku and receives a **batch plan**: an ordered list of 5-10 card targets (by suit+rank identifier). The agent explains its strategy briefly.

### Step 2 — Broadcast
Each agent's first target from its plan is broadcast to all other agents. This is the "intention" — the card the agent will try to pick up this round.

### Step 3 — Conflict Resolution (Deterministic)
If two or more agents target the same card:
- The agent **closest** to the card wins.
- Losing agents fall back to their next planned target. If that also conflicts, continue down the plan.
- If an agent's entire batch conflicts, it skips this round (waits).

### Step 4 — Execute
Each agent with a resolved target:
- Travels to the card (simulated travel cost applies).
- Picks up the card (marks picked_up=True, picked_up_by=agent_id).
- Updates its position to the card's location.

### Step 5 — Repeat
If unpicked cards remain, go to Step 1. Agents re-plan every round (their batch may be partially stale due to other agents' actions).

Note: Agents re-plan every round rather than executing their full batch because other agents' actions invalidate stale plans. The batch serves as the LLM's strategic thinking — "here's my priority order" — but only the first non-conflicting target executes per round.

## LLM Prompt Design (Haiku)

### System Prompt
```
You are a pickup agent in a 52 Card Pickup simulation on a 10x10 grid.
You are at position ({x}, {y}). Plan your next moves to efficiently
collect cards while avoiding conflicts with other agents.

Prioritize:
- Cards close to your current position (minimize travel distance)
- Clusters of nearby cards (plan a path through them)
- Cards that other agents are NOT heading toward

Respond with ONLY a JSON object (no markdown):
{"targets": ["<rank> of <suit>", ...], "strategy": "<brief explanation>"}

List 5-10 cards in the order you want to collect them.
```

### User Message
Contains: remaining cards with positions, other agents' positions and announced intentions.

## State Schema Changes

Add to `AppState`:
- `agent_positions: dict[str, Tuple[float, float]]` — current position of each agent
- `agent_intentions: dict[str, str]` — each agent's announced next target
- `agent_strategies: dict[str, str]` — each agent's strategy explanation (for observability)
- `llm_calls: int` — total LLM API calls made during pickup (for cost tracking)
- `conflicts_resolved: int` — total conflicts detected and resolved

## Graph Changes

The Phase 3 graph uses the same structure as Phase 2:
`START -> scatter -> supervisor -> timer_start -> pickup -> timer_stop -> verify -> END`

The supervisor still decides how many agents. The `pickup_node` implementation changes internally to use the round-based LLM loop instead of greedy nearest-neighbor.

The Phase 1/2 deterministic pickup remains available via `build_graph(with_supervisor=..., llm_pickup=False)` for comparison.

## Conflict Resolution Details

Deterministic, no LLM calls:
1. Collect all agents' first-choice targets.
2. Identify conflicts (same card targeted by 2+ agents).
3. For each conflict, compute distance from each competing agent to the card. Closest agent wins. Ties broken by agent_id (lower wins).
4. Losing agents try their next planned target. If it's also taken (by another agent's resolved target), continue down the list.
5. An agent with no viable target this round does nothing (stays in place).

## Comparison Experiment

Run both Phase 1 (deterministic) and Phase 3 (LLM-powered) with the same seeds:

Report:
- Pickup time (both)
- Total LLM calls made
- Total conflicts resolved
- Estimated API cost
- Verifier pass/fail
- Whether LLM agents found more efficient paths than greedy nearest-neighbor

## Cost Tracking

Track and report:
- Number of LLM calls per run
- Input/output tokens (from API response metadata)
- Estimated cost using Haiku pricing ($0.80/$4.00 per million tokens)

## Implementation Constraints

- Haiku model: `claude-haiku-4-5-20251001`
- Supervisor stays on Sonnet: `claude-sonnet-4-20250514`
- Batch size: request 5-10 targets per LLM call, but only execute 1 per round
- The round loop runs in the main process (no ProcessPoolExecutor) since the bottleneck is API latency, not CPU
- Travel cost remains 0.005s per distance unit
- Fallback: if an agent's LLM call fails, use greedy nearest-neighbor for that round
- Verifier must still pass 100%

## File Structure

All code in `card_pickup.py`. The `build_graph` function gains an `llm_pickup` parameter. `main()` updated to run Phase 3 comparison when API key is available.

## Success Criteria

- LLM agents complete pickup successfully (verifier passes)
- Conflict resolution works without deadlocks (all 52 cards always collected)
- Agents demonstrate spatial awareness in their strategies (not random picking)
- Cost-per-run documented and within Haiku budget expectations
- A developer reading the code can clearly see: planning, broadcasting, conflict resolution, and execution as distinct steps
