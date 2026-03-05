# Phase 2 Implementation Prompt — LLM-Powered Supervisor

## Tier 2 (Semi-Formal)

---

## Context

Phase 1 of the 52 Card Pickup project is complete. Four deterministic agents (Scatter, Timer, Pickup, Verifier) operate on shared state via a LangGraph StateGraph. The scaling experiment runs 1, 2, and 4 pickup agents with timing and verification across 10 trials each. All verifier checks pass.

Phase 2 introduces one LLM-powered agent — a **Supervisor** — that makes strategic decisions about how to deploy the deterministic pickup workers. The workers themselves do not change.

## Objective

Add a Supervisor node to the LangGraph state graph that:

1. Receives the scattered card positions from the Scatter node.
2. Analyzes the spatial distribution of the 52 cards on the 10x10 grid.
3. Decides how many pickup agents to deploy (1, 2, or 4).
4. Optionally recommends a partitioning strategy (full grid, left/right halves, quadrants).
5. Explains its reasoning in natural language.
6. Stores its decision and reasoning in the shared state.

The Supervisor uses the **Anthropic Python SDK** to call **Claude** for its decision.

## Graph Modification

**Phase 1 graph:**
`START -> scatter -> timer_start -> pickup -> timer_stop -> verify -> END`

**Phase 2 graph:**
`START -> scatter -> supervisor -> timer_start -> pickup -> timer_stop -> verify -> END`

The Supervisor node reads `state["cards"]` and writes:
- `state["pickup_agents"]` — integer (1, 2, or 4)
- `state["supervisor_reasoning"]` — string (natural language explanation)

## Supervisor Prompt Design

The Supervisor receives a structured summary of the scatter pattern, NOT the raw list of 52 card positions. The summary should include metrics that are relevant to the parallelism decision:

- **Card density per quadrant** — count of cards in each of the four quadrants (x<5/y<5, x>=5/y<5, x<5/y>=5, x>=5/y>=5)
- **Spatial spread** — standard deviation of x and y coordinates
- **Clustering** — average nearest-neighbor distance across all cards
- **Balance ratio** — how evenly cards are distributed across quadrants (e.g., min/max quadrant count)

The LLM is asked to reason about whether the distribution benefits from parallelism and to return a structured response with its agent count decision and explanation.

## State Schema Changes

Add to `AppState`:
- `supervisor_reasoning: Optional[str]` — the Supervisor's natural language explanation

The existing `pickup_agents: int` field is now set by the Supervisor instead of being hardcoded in the initial state.

## Comparison Experiment

After the Supervisor-driven runs, compare against Phase 1 brute-force results:

1. Run the Supervisor configuration N times (seeded for reproducibility).
2. For each run, also execute the brute-force configurations (1, 2, and 4 agents) with the same seed.
3. Report whether the Supervisor's choice matched the fastest brute-force configuration.
4. Print a comparison table showing:
   - Supervisor's choice and reasoning (summarized)
   - Actual time with Supervisor's choice
   - Best brute-force time and which configuration achieved it
   - Whether Supervisor matched optimal

## Implementation Constraints

- The Anthropic API key is read from the `ANTHROPIC_API_KEY` environment variable.
- The Supervisor must handle API failures gracefully — if the LLM call fails, fall back to a default of 2 agents and note the failure in the reasoning field.
- The Supervisor's LLM call is the ONLY non-deterministic element. All other agents remain pure functions.
- The verifier still runs after every configuration and must pass 100%.
- Keep the Supervisor's prompt minimal and focused. Do not over-engineer the spatial analysis — the point is to demonstrate the hybrid pattern, not to build a perfect optimizer.

## Dependencies

Add `anthropic` to `requirements.txt`.

## File Structure

All Phase 2 code goes in `card_pickup.py` alongside the existing Phase 1 code. The `main()` function is updated to run both the Supervisor experiment and the brute-force comparison. Phase 1's standalone mode (hardcoded agent counts) should remain accessible for reference.

## Success Criteria

- Supervisor makes reasonable deployment decisions (not always the same answer).
- Reasoning is coherent and references the spatial metrics provided.
- Supervisor choices match or come close to the optimal brute-force configuration.
- Verifier passes 100% of all runs.
- A developer reading the code can clearly see the boundary between "LLM decides" and "deterministic agents execute."
