# Phase 4 Tutorial: Observability and Governance

## What You'll Learn

- How to instrument a multi-agent system with event logging
- Runtime governance: invariant checks that halt on violations
- Performance metrics and anomaly detection
- Building a live terminal dashboard for agent monitoring
- Event replay for debugging and analysis

## Prerequisites

- Completed Phase 3 tutorial
- Understanding of the round-based pickup loop

## Why Observability Matters

In Phases 1-3, we built agents and ran experiments. But when something goes wrong in a multi-agent system, how do you know? And more importantly, how do you figure out *why*?

Observability answers both questions. It's the difference between "the run failed" and "agent_2 picked up the same card twice in round 7 because the conflict resolver didn't account for the greedy fallback path."

## The Event Log

Every meaningful action in the simulation produces an `Event` record:

```python
@dataclass
class Event:
    timestamp: float       # seconds since run start
    event_type: str        # scatter, plan, broadcast, conflict, pickup, verify, governance
    agent_id: Optional[str]
    data: dict             # event-specific payload
```

Events are collected in an `EventLog` stored in state. A typical run produces ~180 events:
- 1 scatter, 52 plans, 52 broadcasts, 52 pickups, ~13 rounds, ~11 conflicts, 1 verify

## Governance: Runtime Guardrails

The `GovernanceChecker` runs after every pickup round and enforces four invariants:

1. **Card count**: Always exactly 52 cards
2. **No double pickup**: A picked-up card can't be picked again
3. **No phantom agents**: Every `picked_up_by` must be a real agent
4. **Monotonic progress**: The number of picked-up cards never decreases

If any invariant is violated, execution halts immediately with a `GovernanceViolation`. This is the difference between monitoring ("we noticed something bad") and governance ("we stopped something bad").

## Performance Metrics

After each run, `MetricsCalculator` derives from the event log:

- **Cards per second per agent**: throughput
- **Total distance per agent**: efficiency
- **Idle rounds per agent**: wasted time
- **Conflict rate**: coordination overhead

```
Performance Metrics (elapsed: 119.69s)
Agent       Cards  Cards/s   Distance  Idle Rounds
agent_0        13      0.1      21.73            0
agent_1        13      0.1      22.13            0
agent_2        13      0.1      19.43            0
agent_3        13      0.1      34.34            0
Conflicts: 11 (rate: 21.2%)
```

## Anomaly Detection

`AnomalyDetector` flags unusual behavior:

- **Unbalanced workload**: One agent did >60% of the work (with 2+ agents)
- **Excessive conflicts**: Conflict rate >30%
- **Stalled agent**: 5+ consecutive idle rounds
- **Path inefficiency**: Agent traveled 2x the optimal distance

These aren't errors — they're signals that something unexpected happened and warrants investigation.

## The Terminal Dashboard

The `Dashboard` class uses Python's `curses` library (stdlib, no extra deps) to show a live view during simulation:

```
 52 Card Pickup - Live Dashboard
 . . . . . . . . . .   Agent Status
 . . C . . . . . . .   agent_0: (2.3,4.1) cards:7
 . . . . 0 . . C . .   agent_1: (8.1,2.5) cards:5
 . . . . . . . . . .   ...
 Progress: [###########---------] 26/52
 Elapsed: 0.045s  Round: 7  Conflicts: 3
 Recent events:
 [0.012] agent_0 picked up 7 of hearts
 [0.015] CONFLICT over K of spades: agent_1 wins vs agent_2
```

Enable it with `--dashboard`. The grid shows `.` for empty cells, `C` for cards, and agent numbers for agent positions.

## Running It

```bash
# Phase 3 with observability, save event log
python -m card_pickup --phase 3 --save-log

# Replay saved event log through dashboard
python -m card_pickup --replay event_log_trial_1.json

# Live dashboard during run
python -m card_pickup --phase 3 --dashboard
```

## Event Log Save/Replay

Event logs are saved as JSON files. The replay function reads events and re-creates the dashboard display at real-time speed (adjustable with speed multiplier).

This is invaluable for debugging: run once, replay as many times as needed. No API calls required for replay.

## Key Concept: Monitoring vs Governance

**Monitoring** tells you what happened. The event log, metrics, and anomaly detection are monitoring — they observe after the fact.

**Governance** prevents bad things from happening. The invariant checks run *during* execution and can halt the system. They're guardrails, not reports.

Production multi-agent systems need both. You monitor to learn and improve. You govern to ensure safety.

## The Architecture

All observability code lives in `observability.py`, separate from agent logic in `card_pickup/_core.py`. This separation is deliberate:

- Agents don't depend on observability
- Observability can be turned on/off without changing agents
- New monitoring can be added without touching agent code

The connection is lightweight: agents check `_active_event_log` and emit events if it's set.

## Exercises

1. **Inject a fault**: Temporarily modify `llm_pickup_node` to set `card["picked_up"] = False` after picking up a card. Does the governance checker catch it?
2. **Add a new invariant**: Write a check that no agent holds more than 20 cards. Add it to `GovernanceChecker`.
3. **Add a new metric**: Track "average time between pickups" per agent. Add it to `MetricsCalculator`.
4. **Customize the dashboard**: Add a column showing each agent's total distance traveled.

## Next

[Phase 5: Extensibility and Teaching](tutorial_phase_5.md) — Benchmark suite, plugin architecture, and packaging for others.
