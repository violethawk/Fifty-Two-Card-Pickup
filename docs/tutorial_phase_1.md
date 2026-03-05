# Phase 1 Tutorial: Deterministic Multi-Agent Orchestration

## What You'll Learn

- How to define agent roles as Python functions
- How to use LangGraph's `StateGraph` to orchestrate agents
- How shared state coordinates agents without side channels
- How to scale from 1 to N agents with task decomposition

## Prerequisites

- Python 3.11+
- Basic Python knowledge (functions, dicts, lists)
- No multi-agent or LLM experience needed

## Setup

```bash
git clone https://github.com/violethawk/Fifty-Two-Card-Pickup.git
cd Fifty-Two-Card-Pickup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## The Metaphor

Imagine 52 playing cards scattered on a floor. You need to pick them all up. That's it — that's the entire problem.

Now imagine you have helpers. With one person, you pick up everything yourself. With two, you split the room in half. With four, you each take a quadrant. More workers means less walking per person, but you need to coordinate who covers what.

This is the fundamental multi-agent problem: **divide work, coordinate agents, verify results**.

## The Agents

Open `card_pickup.py`. The simulation has four agents, each a Python function:

### 1. Scatter Agent (`scatter_node`, line ~108)

Creates 52 cards with random positions on a 10x10 grid. Each card is a dict:

```python
card = {
    "suit": "hearts",
    "rank": "7",
    "x": 3.45,        # position on grid
    "y": 7.82,
    "picked_up": False,
    "picked_up_by": None,
}
```

### 2. Timer Agent (`timer_start_node` / `timer_stop_node`)

Records timestamps before and after pickup. Simple but important — it measures performance without interfering with the work.

### 3. Pickup Agent (`pickup_node`, line ~629)

The worker. It picks up cards using greedy nearest-neighbor: always go to the closest unpicked card. With multiple agents, the grid is partitioned into regions so agents don't step on each other.

### 4. Verifier Agent (`verify_node`, line ~700)

The constraint checker. After everything is done, it verifies:
- Exactly 52 cards exist
- Every card is marked as picked up
- No duplicates

If any check fails, the run is marked FAIL.

## The State

All agents share a single `AppState` dict (defined as a `TypedDict`):

```python
class AppState(TypedDict):
    cards: List[Card]
    phase: str
    start_time: Optional[float]
    end_time: Optional[float]
    pickup_agents: int
    result: Optional[str]
```

This is the coordination mechanism. No agent talks to another agent directly — they all read and write the shared state. LangGraph passes the state from one node to the next.

## The Graph

The `build_graph` function (line ~731) wires the agents into a pipeline:

```
START -> scatter -> timer_start -> pickup -> timer_stop -> verify -> END
```

Each arrow is an edge. LangGraph executes nodes in this order, passing the state along. The graph is compiled once and can be invoked many times with different initial states.

## Running It

```bash
python card_pickup.py --phase 1
```

You'll see output like:

```
=== Phase 1: Brute-Force Scaling Experiment ===

| Agents | Avg Time (s) | Best (s) | Worst (s) | Verifier |
|--------|--------|--------|--------|--------|
| 1 | 0.3300 | 0.2900 | 0.3900 | 10/10 |
| 2 | 0.2000 | 0.1800 | 0.2300 | 10/10 |
| 4 | 0.1400 | 0.1200 | 0.1500 | 10/10 |
```

Key observations:
- **More agents = faster pickup** — 4 agents is ~2.4x faster than 1
- **Verifier passes 100%** — the constraint is never violated
- **Travel cost matters** — agents simulate physical movement (0.005s per unit distance)

## How Parallelism Works

With 1 agent, the entire grid is one region. The agent starts at (0,0) and picks up all 52 cards.

With 2 agents, the grid splits at x=5:
- Agent 0 handles x < 5 (left half)
- Agent 1 handles x >= 5 (right half)

With 4 agents, the grid splits into quadrants at x=5, y=5.

Each agent runs in a separate process (`ProcessPoolExecutor`) so they truly work in parallel.

## Key Concept: The Verifier Pattern

The most important pattern in this phase is the **verifier agent**. It runs after every configuration, every trial. It's the invariant that guarantees correctness.

In production multi-agent systems, this is governance. You can't trust that agents did their job correctly — you need an independent check. The verifier is that check.

## Exercises

1. **Change the grid size**: What happens if you increase the grid to 100x100? Does the scaling ratio change?
2. **Add a 3-agent configuration**: Modify `_determine_region` to split the grid into 3 regions. Does 3 agents beat 2?
3. **Change the strategy**: Instead of nearest-neighbor, try picking up cards in a fixed order (e.g., by suit). How does it compare?
4. **Break the verifier**: Temporarily modify `pickup_node` to skip one card. Does the verifier catch it?

## Next

[Phase 2: LLM-Powered Supervisor](tutorial_phase_2.md) — Add an intelligent supervisor that decides how many agents to deploy.
