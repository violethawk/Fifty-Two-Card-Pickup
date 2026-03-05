# Pickup Agent Delivery to Verifier — Visual Convergence Mechanic
## Tier 2 Semi-Formal | Phase Enhancement (All Phases)

---

### Context

The 52 Card Pickup project currently has pickup agents that collect cards in their assigned territory and mark them as `picked_up: true` in shared state. The verifier then reads shared state and confirms all 52 cards are accounted for. This works correctly but is invisible — the verifier is a state check, not a presence on the grid.

This prompt adds a delivery mechanic: pickup agents physically carry their collected cards to the verifier agent's position on the grid. The verifier becomes a visible station that agents converge on. Cards are not verified until delivered.

This creates two visible phases in every run:
1. **Fan out** — agents spread from their starting positions into their territories, picking up cards
2. **Converge** — agents travel to the verifier station and deliver their collected cards

---

### What to Change

**1. Verifier Gets a Grid Position**

The verifier agent is placed at a fixed position on the grid. Default: `(5, 5)` — center of the 10x10 grid.

Add to shared state:
```python
"verifier_position": {"x": 5.0, "y": 5.0},
"cards_delivered": 0,          # increments as agents deliver
"cards_in_transit": 0,         # cards picked up but not yet delivered
"deliveries": [
    {
        "agent_id": str,
        "cards_delivered": int,
        "delivery_time": float,
        "travel_distance": float   # from last pickup to verifier
    }
]
```

**2. Pickup Agent Lifecycle Changes**

Current lifecycle:
```
start → pick up nearest card → repeat until territory empty → done
```

New lifecycle:
```
start → pick up nearest card → repeat until territory empty → travel to verifier → deliver cards → done
```

After an agent picks up its last card, it:
- Calculates distance from its current position to the verifier position
- Adds simulated travel time for that distance (same travel cost model used for card-to-card movement)
- Moves to the verifier position
- Delivers all collected cards: `cards_delivered += agent.cards_collected`
- The agent's delivery is logged as an event

**3. Verifier Behavior Changes**

Current behavior:
- Runs once after all pickup is complete
- Reads shared state, checks 52 cards present, reports PASS/FAIL

New behavior:
- The verifier is active throughout the delivery phase
- As each agent arrives, it receives that agent's cards and adds them to its verified count
- After all agents have delivered, it performs the final integrity check:
  - Exactly 52 cards delivered
  - All suit-rank combinations present, no duplicates
  - Cards delivered == cards picked up (no cards lost in transit)
- Reports PASS/FAIL with per-agent delivery breakdown

The verifier does NOT start checking until the first agent arrives. It does NOT declare PASS until all agents have delivered.

**4. Timer Adjustments**

The timer now captures three intervals:
```python
"timing": {
    "pickup_start": float,
    "pickup_end": float,       # last card picked up by any agent
    "delivery_end": float,     # last agent arrives at verifier
    "pickup_duration": float,  # pickup_end - pickup_start
    "delivery_duration": float, # delivery_end - pickup_end
    "total_duration": float    # delivery_end - pickup_start
}
```

**5. Event Log Additions**

New event types for the observability layer.

**6. Visualization Updates**

TUI Dashboard, Streamlit App, and scatter pattern visualizations all updated.

**7. Benchmark Impact**

Update benchmark output to show delivery cost with Delivery % column.

---

### Acceptance Criteria

1. Pickup agents travel to verifier position after collecting all cards in their territory
2. Verifier receives cards from each agent and counts incrementally
3. Final verifier check passes 100% of runs — delivery mechanic introduces no new failure modes
4. Timer reports pickup duration, delivery duration, and total duration separately
5. Event log shows agent departure, travel, arrival, and verifier receipt events
6. TUI dashboard shows verifier position on grid and delivery phase visually
7. Streamlit app animates both fan-out and convergence phases
8. Benchmark table includes delivery time and delivery percentage columns
9. Verifier position visible on scatter pattern visualizations
