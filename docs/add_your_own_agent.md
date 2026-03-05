# Add Your Own Agent

This guide walks you through adding a new agent to the 52 Card Pickup simulation. By the end, you'll have a working **Sorter agent** that reorders collected cards by suit after pickup, wired into the LangGraph state graph with full event logging.

## Prerequisites

- You've read through Phase 1 and understand how the state graph works
- You can run `python -m card_pickup --phase 1` successfully

## Step 1: Define Your Agent's Role

Before writing code, answer three questions:

1. **What does your agent do?** The Sorter agent organizes picked-up cards by suit.
2. **What state does it read?** `cards` (the list of all cards).
3. **What state does it write?** `cards` (reordered) and a new field `sort_order`.

Every agent in this system is a Python function that receives the shared `AppState` dict and returns it (possibly modified).

## Step 2: Update AppState (if needed)

If your agent needs a new field, add it to the `AppState` TypedDict in `card_pickup/_core.py`:

```python
class AppState(TypedDict):
    # ... existing fields ...
    sort_order: Optional[str]  # suit ordering used by the sorter agent
```

And add a default value in `_make_initial_state`:

```python
def _make_initial_state(num_agents: int = 1) -> AppState:
    return {
        # ... existing fields ...
        "sort_order": None,
    }
```

## Step 3: Write the Node Function

Add your agent function in `card_pickup/_core.py`, near the other node functions:

```python
SUIT_ORDER = {"hearts": 0, "diamonds": 1, "clubs": 2, "spades": 3}


def sorter_node(state: AppState) -> AppState:
    """Sort collected cards by suit after pickup.

    This agent demonstrates post-processing: it runs after the pickup
    phase and reorganizes the card list by suit. It doesn't change
    which cards were picked up or by whom — only the ordering.
    """
    cards = state["cards"]

    # Separate picked and unpicked (there shouldn't be unpicked ones
    # after a successful run, but be defensive)
    picked = [c for c in cards if c["picked_up"]]
    unpicked = [c for c in cards if not c["picked_up"]]

    # Sort by suit, then by rank within suit
    rank_order = {r: i for i, r in enumerate(RANKS)}
    picked.sort(key=lambda c: (SUIT_ORDER.get(c["suit"], 99),
                                rank_order.get(c["rank"], 99)))

    state["cards"] = picked + unpicked
    state["sort_order"] = " -> ".join(SUIT_ORDER.keys())

    # Emit event for observability
    if _active_event_log is not None:
        _active_event_log.emit("sort", data={
            "order": state["sort_order"],
            "cards_sorted": len(picked),
        })

    return state
```

Key points:
- The function signature is `(state: AppState) -> AppState`
- Read what you need from state, modify it, return it
- Emit events if observability is active (check `_active_event_log`)

## Step 4: Register the Node in build_graph

In the `build_graph` function, add your node and an edge:

```python
def build_graph(with_supervisor=False, llm_pickup=False, with_sorter=False):
    builder = StateGraph(AppState)

    # ... existing nodes ...
    builder.add_node("verify", verify_node)

    if with_sorter:
        builder.add_node("sorter", sorter_node)

    # ... existing edges ...
    if with_sorter:
        builder.add_edge("verify", "sorter")
        builder.add_edge("sorter", END)
    else:
        builder.add_edge("verify", END)

    return builder.compile()
```

The graph now flows: `... -> verify -> sorter -> END`

## Step 5: Test It

Write a quick test:

```python
import random
from card_pickup import build_graph, _make_initial_state, SUIT_ORDER

random.seed(42)
graph = build_graph(with_sorter=True)
state = graph.invoke(_make_initial_state(num_agents=2))

print(f"Verifier: {state['result']}")
print(f"Sort order: {state['sort_order']}")

# Check that cards are actually sorted by suit
cards = state["cards"]
suits_seen = [c["suit"] for c in cards]
suit_indices = [SUIT_ORDER[s] for s in suits_seen]
assert suit_indices == sorted(suit_indices), "Cards not sorted!"
print("Sort verified!")
```

## Step 6: Add a CLI Flag (Optional)

In `main()`, add an argument:

```python
parser.add_argument("--sort", action="store_true",
                    help="Enable the Sorter agent after verification")
```

And pass it to `build_graph`:

```python
graph = build_graph(with_sorter=args.sort)
```

## Summary

Adding an agent takes four steps:

1. **Define the role** — what does it read and write?
2. **Write the function** — `(AppState) -> AppState`
3. **Register in the graph** — `add_node` + `add_edge`
4. **Test with the verifier** — the verifier still runs, so you know you didn't break anything

The pattern is always the same: state in, state out. LangGraph handles the sequencing. The verifier ensures correctness. Event logging gives you observability for free.
