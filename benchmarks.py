"""
benchmarks.py
=============

Standardized scatter patterns and a benchmark runner for reproducible
comparisons across pickup strategies and agent configurations.

Each pattern function returns a list of 52 Card dicts with predetermined
positions.  Patterns are deterministic — no randomness involved.

Usage::

    python card_pickup.py --benchmark
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Tuple

from card_pickup import (
    AppState,
    Card,
    SUITS,
    RANKS,
    TRAVEL_COST_PER_UNIT,
    _card_key,
    _extract_elapsed,
    _extract_timing,
    _make_initial_state,
    build_graph,
)
from observability import EventLog
import card_pickup as _cp


# ---------------------------------------------------------------------------
# Scatter patterns
# ---------------------------------------------------------------------------

def _make_deck(positions: List[Tuple[float, float]]) -> List[Card]:
    """Build a 52-card deck with the given (x, y) positions."""
    cards: List[Card] = []
    idx = 0
    for suit in SUITS:
        for rank in RANKS:
            x, y = positions[idx % len(positions)]
            cards.append({
                "suit": suit,
                "rank": rank,
                "x": x,
                "y": y,
                "picked_up": False,
                "picked_up_by": None,
            })
            idx += 1
    return cards


def pattern_uniform() -> List[Card]:
    """Cards evenly spaced across the grid in a spiral-like pattern."""
    positions = []
    for i in range(52):
        # Distribute across grid using golden-ratio spacing
        angle = i * 2.399963  # golden angle in radians
        r = math.sqrt(i / 52.0) * 5.0  # radius scales to fill grid
        x = 5.0 + r * math.cos(angle)
        y = 5.0 + r * math.sin(angle)
        x = max(0.0, min(10.0, x))
        y = max(0.0, min(10.0, y))
        positions.append((x, y))
    return _make_deck(positions)


def pattern_clustered() -> List[Card]:
    """All 52 cards concentrated in the bottom-left quadrant."""
    positions = []
    for i in range(52):
        row = i // 8
        col = i % 8
        x = 0.3 + col * 0.5
        y = 0.3 + row * 0.5
        positions.append((x, y))
    return _make_deck(positions)


def pattern_two_clusters() -> List[Card]:
    """26 cards near (1, 1), 26 cards near (9, 9)."""
    positions = []
    for i in range(26):
        row = i // 6
        col = i % 6
        positions.append((0.5 + col * 0.4, 0.5 + row * 0.4))
    for i in range(26):
        row = i // 6
        col = i % 6
        positions.append((7.5 + col * 0.4, 7.5 + row * 0.4))
    return _make_deck(positions)


def pattern_four_clusters() -> List[Card]:
    """13 cards in each quadrant corner."""
    positions = []
    corners = [(1.0, 1.0), (9.0, 1.0), (1.0, 9.0), (9.0, 9.0)]
    for cx, cy in corners:
        for i in range(13):
            row = i // 4
            col = i % 4
            positions.append((cx - 0.6 + col * 0.4, cy - 0.6 + row * 0.4))
    return _make_deck(positions)


def pattern_diagonal() -> List[Card]:
    """Cards distributed along the diagonal from (0,0) to (10,10)."""
    positions = []
    for i in range(52):
        t = i / 51.0
        # Slight jitter perpendicular to diagonal
        offset = (i % 3 - 1) * 0.3
        x = t * 10.0 + offset * 0.707
        y = t * 10.0 - offset * 0.707
        x = max(0.0, min(10.0, x))
        y = max(0.0, min(10.0, y))
        positions.append((x, y))
    return _make_deck(positions)


def pattern_edge() -> List[Card]:
    """Cards distributed along the perimeter of the grid."""
    positions = []
    perimeter = 40.0  # 4 sides of 10
    for i in range(52):
        t = (i / 52.0) * perimeter
        if t < 10.0:
            positions.append((t, 0.0))
        elif t < 20.0:
            positions.append((10.0, t - 10.0))
        elif t < 30.0:
            positions.append((30.0 - t, 10.0))
        else:
            positions.append((0.0, 40.0 - t))
    return _make_deck(positions)


PATTERNS: Dict[str, callable] = {
    "uniform": pattern_uniform,
    "clustered": pattern_clustered,
    "two_clusters": pattern_two_clusters,
    "four_clusters": pattern_four_clusters,
    "diagonal": pattern_diagonal,
    "edge": pattern_edge,
}


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def _run_pattern_benchmark(
    pattern_name: str,
    cards: List[Card],
    num_agents: int,
    save_log: bool = False,
) -> Tuple[float, float, float, bool]:
    """Run a single benchmark: given cards + agent count, return (elapsed, passed)."""
    graph = build_graph(with_supervisor=False, llm_pickup=False)

    state = _make_initial_state(num_agents)
    # Inject pre-built cards instead of scattering
    state["cards"] = [dict(c) for c in cards]  # deep copy
    state["phase"] = "pickup"

    # Skip scatter node — invoke from timer_start onward.
    # Since we can't skip nodes in the compiled graph, we build a
    # mini graph without scatter.
    from langgraph.graph import END, START, StateGraph
    from card_pickup import (
        timer_start_node, pickup_node, delivery_node, timer_stop_node, verify_node,
    )

    builder = StateGraph(AppState)
    builder.add_node("timer_start", timer_start_node)
    builder.add_node("pickup", pickup_node)
    builder.add_node("delivery", delivery_node)
    builder.add_node("timer_stop", timer_stop_node)
    builder.add_node("verify", verify_node)
    builder.add_edge(START, "timer_start")
    builder.add_edge("timer_start", "pickup")
    builder.add_edge("pickup", "delivery")
    builder.add_edge("delivery", "timer_stop")
    builder.add_edge("timer_stop", "verify")
    builder.add_edge("verify", END)
    mini_graph = builder.compile()

    # Wire up event logging if requested
    elog = None
    if save_log:
        elog = EventLog()
        _cp._active_event_log = elog

    try:
        final_state = mini_graph.invoke(state)
    finally:
        _cp._active_event_log = None

    elapsed = _extract_elapsed(final_state)
    timing = _extract_timing(final_state)
    passed = final_state.get("result", "").startswith("PASS")

    if elog and save_log:
        filename = f"event_log_bench_{pattern_name}_{num_agents}ag.json"
        elog.save(filename)

    return elapsed, timing["pickup_duration"], timing["delivery_duration"], passed


def run_benchmarks(save_log: bool = False) -> None:
    """Run all patterns against all agent configurations and print results."""
    configs = [1, 2, 4]

    print("=== Benchmark Suite ===\n")
    header = ["Pattern", "Agents", "Pickup", "Delivery", "Total", "Delivery %"]
    print("| " + " | ".join(f"{h:>14}" for h in header) + " |")
    print("|" + "----------------|" * len(header))

    all_passed = True
    all_results: Dict[str, List] = {}

    for name, pattern_fn in PATTERNS.items():
        cards = pattern_fn()
        pattern_results = []

        for n in configs:
            elapsed, pickup_t, delivery_t, passed = _run_pattern_benchmark(
                name, cards, n, save_log=save_log,
            )
            pattern_results.append((n, elapsed, pickup_t, delivery_t, passed))
            if not passed:
                all_passed = False

        all_results[name] = pattern_results
        best = min(pattern_results, key=lambda r: r[1])

        for n, elapsed, pickup_t, delivery_t, passed in pattern_results:
            marker = " *" if n == best[0] else "  "
            status = "" if passed else " FAIL"
            del_pct = (delivery_t / elapsed * 100) if elapsed > 0 else 0
            print(
                f"| {name:>14} | {n:>14} | {pickup_t:>10.4f}s  "
                f"| {delivery_t:>10.4f}s  | {elapsed:>10.4f}s{marker}{status}"
                f" | {del_pct:>10.1f}%  |"
            )

    print()
    # Summary: best config per pattern
    print("Best configs:")
    for name, results in all_results.items():
        best = min(results, key=lambda r: r[1])
        print(f"  {name}: {best[0]} agent{'s' if best[0] > 1 else ''} ({best[1]:.4f}s)")

    print()
    if all_passed:
        print("All verifier checks passed.")
    else:
        print("WARNING: Some verifier checks failed!")
