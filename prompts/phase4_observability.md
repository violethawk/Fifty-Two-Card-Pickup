# Phase 4 Implementation Prompt — Observability and Governance

## Tier 2 (Semi-Formal)

---

## Context

Phases 1-3 are complete. The system has deterministic agents, an LLM supervisor, and LLM-powered pickup agents with conflict resolution. All run through LangGraph state graphs. The code lives in `card_pickup.py`.

Phase 4 adds the infrastructure layer: event logging, replay, performance metrics, governance guardrails, anomaly detection, and a terminal TUI dashboard.

## Objective

Instrument the existing simulation so that every agent action is recorded, monitored, and governed — without changing agent behavior. This is observability and governance bolted onto the existing system.

## Architecture

### 1. Event Log

Every meaningful action produces an `Event` record:

```python
@dataclass
class Event:
    timestamp: float          # time.perf_counter() relative to run start
    event_type: str           # scatter | plan | broadcast | conflict | pickup | verify | governance
    agent_id: Optional[str]   # which agent, if applicable
    data: dict                # event-specific payload
```

Events are collected in an `EventLog` (list of Events) stored in state as serializable dicts. Key events:

- `scatter`: card positions assigned
- `plan`: agent planned targets (with strategy)
- `broadcast`: agent announced intention
- `conflict`: conflict detected + resolution (winner, losers)
- `pickup`: agent picked up a card (card key, distance traveled, travel time)
- `move`: agent changed position
- `verify`: verifier result
- `governance`: invariant check result (pass or violation)

### 2. Governance Layer

Runtime invariant checks that run after every round of pickup. Violations halt execution immediately.

Invariants:
- **Card count**: `len(cards) == 52` at all times
- **No double pickup**: a card marked `picked_up=True` must not be picked up again
- **No phantom cards**: every `picked_up_by` agent must exist in the agent list
- **Monotonic progress**: the number of picked-up cards must never decrease between rounds

The governance check is a function called after each round in both deterministic and LLM pickup. On violation, it logs a `governance` event and raises `GovernanceViolation`.

### 3. Performance Metrics

Computed from the event log after a run completes:

- **Cards per second per agent**: total cards picked by agent / elapsed time
- **Idle rounds per agent**: rounds where agent had no viable target (waited)
- **Conflict rate**: conflicts / total pickup attempts
- **Total travel distance per agent**: sum of distances from pickup events
- **Average planning time**: mean time spent in LLM calls (Phase 3 only)

Metrics are printed as a summary table after each run.

### 4. Anomaly Detection

Flag runs where agents behave unexpectedly. Checks applied post-run:

- **Unbalanced workload**: one agent picked up >60% of cards (with 2+ agents)
- **Excessive conflicts**: conflict rate >30%
- **Stalled agent**: an agent went 5+ consecutive rounds without picking up a card
- **Path inefficiency**: agent's total travel distance exceeds 2x the optimal greedy distance

Each anomaly generates a warning message in the run output.

### 5. Terminal TUI Dashboard

A live-updating terminal display during simulation runs. Uses `curses` (stdlib, no extra deps).

Layout:
```
+------------------------------------------+
| 52 Card Pickup — Live Dashboard          |
+------------------------------------------+
|  10x10 Grid          |  Agent Status     |
|  . . . . . . . . . . |  agent_0: (2.3,4.1) cards:7  |
|  . . C . . . . . . . |  agent_1: (8.1,2.5) cards:5  |
|  . . . . A . . C . . |  agent_2: (1.0,9.3) cards:6  |
|  . . . . . . . . . . |  agent_3: (5.5,7.2) cards:8  |
|  ... (grid rows)      |                   |
+------------------------------------------+
|  Progress: 26/52 cards | Conflicts: 3    |
|  Elapsed: 0.045s       | Round: 7/~13    |
+------------------------------------------+
|  Last events:                            |
|  [0.012] agent_0 picked up 7 of hearts   |
|  [0.015] CONFLICT: agent_1 vs agent_2    |
|  [0.018] agent_2 picked up K of spades   |
+------------------------------------------+
```

Grid display:
- `.` = empty cell
- `C` = unpicked card
- `0-3` = agent position (by agent number)
- `*` = card being picked up this round

The TUI is optional — enabled via `--dashboard` CLI flag or `dashboard=True` parameter. When disabled, the existing text output works as before. The TUI updates after each round in the pickup loop.

### 6. Replay

An event log can be saved to a JSON file and replayed:

- `save_event_log(events, filepath)` — writes the event log to disk
- `replay_event_log(filepath, speed=1.0)` — reads and replays events through the TUI at the given speed multiplier

Replay re-creates the dashboard display from recorded events without re-running the simulation or making API calls.

## State Schema Changes

Add to `AppState`:
- `event_log: List[dict]` — serialized Event records

## Implementation Plan

### New file: `observability.py`

Keep the observability infrastructure separate from agent logic:
- `Event` dataclass
- `EventLog` class (append, query, serialize, save/load)
- `GovernanceChecker` class (invariant checks)
- `MetricsCalculator` class (compute from event log)
- `AnomalyDetector` class (post-run checks)
- `Dashboard` class (curses TUI)
- `replay_event_log()` function

### Changes to `card_pickup.py`

- Import from `observability`
- Add `event_log` to `AppState` and `_make_initial_state`
- Instrument `pickup_node` and `llm_pickup_node` to emit events
- Add governance checks after each pickup round
- Add `--dashboard` flag to `main()`
- Add metrics/anomaly summary after each run
- Wire up event log saving

## Dependencies

No new pip dependencies. `curses` is in the Python stdlib.

## CLI Changes

```
python card_pickup.py                    # existing behavior, text output
python card_pickup.py --dashboard        # live TUI during runs
python card_pickup.py --replay log.json  # replay a saved event log
python card_pickup.py --save-log         # save event logs to files
```

Use `argparse` for CLI parsing.

## Success Criteria

- Every run produces a complete event log
- Governance layer catches injected faults (test by temporarily breaking an invariant)
- Metrics table printed after each run
- Anomaly detection flags unbalanced or inefficient runs
- TUI dashboard shows live progress during runs (when enabled)
- Replay faithfully re-creates a run's dashboard from its event log
- All existing Phase 1-3 functionality unchanged when dashboard is off
