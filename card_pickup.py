"""
card_pickup.py
================

This script implements a small, deterministic multi‑agent simulation
using the `langgraph` library.  It acts as a "hello world" for
multi‑actor systems by modelling the classic **52 card pickup** game
as a simple state machine.  Four agents operate on a shared piece of
state:

* **Scatter agent** – places a standard deck of cards at random
  positions on a 10×10 grid.  Each card is marked as scattered and
  unpicked.
* **Timer agent (start)** – records the wall–clock start time just
  before the pickup begins.
* **Pickup agent(s)** – one or more agents that repeatedly pick up
  cards.  Each agent starts at the origin `(0, 0)` and picks the
  closest unpicked card in its assigned region of the grid, moves to
  that card and marks it as collected.  Regions are assigned based on
  the number of pickup agents (the whole grid, left/right halves or
  quadrants).  Agents run concurrently when there are multiple
  regions.
* **Timer agent (stop)** – records the end time once all cards have
  been collected.
* **Verifier agent** – checks that exactly 52 unique cards exist,
  confirms that every card has been picked up and that there are no
  duplicates.  It writes a human‑readable PASS/FAIL message into the
  state.

The state of the graph is defined with a `TypedDict` to make the
structure explicit.  Each node accepts the state as input, performs
some transformation and returns an updated state.  LangGraph uses
these nodes and explicit edges to orchestrate the flow of data
between them【410095119139183†L728-L796】.  For a simple loop within
the pickup agent we rely on standard Python control flow rather than
LangGraph’s conditional edges (which you can see used in other
examples【913856775414019†L68-L140】).

At the bottom of the file there is a small driver that runs the
simulation with one, two and four pickup agents.  It seeds the random
number generator for reproducibility (each trial uses a different
seed) and runs ten trials per configuration.  After the runs, it
prints a summary table showing the average, best and worst pickup
times along with whether the verifier passed every run.

Run this script with a recent version of Python (≥ 3.11) after
installing the dependencies listed in `requirements.txt`.
"""

from __future__ import annotations

import json
import math
import os
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, TypedDict, Tuple

import anthropic
from langgraph.graph import END, START, StateGraph


class Card(TypedDict):
    """Representation of a single playing card in the simulation."""

    suit: str  # hearts, diamonds, clubs, spades
    rank: str  # 2-10, J, Q, K, A
    x: float   # horizontal position (0.0–10.0)
    y: float   # vertical position (0.0–10.0)
    picked_up: bool
    picked_up_by: Optional[str]  # agent identifier that picked up the card


class AppState(TypedDict):
    """Shared state carried through the graph.

    Each node reads the state, possibly mutates it and returns it.  The
    schema defined here mirrors the state described in the specification.
    """

    cards: List[Card]
    phase: str               # scatter | pickup | verify | done
    start_time: Optional[float]
    end_time: Optional[float]
    pickup_agents: int       # number of concurrent pickup agents
    supervisor_reasoning: Optional[str]  # natural language explanation from the supervisor
    result: Optional[str]    # textual report from the verifier


SUITS = ["hearts", "diamonds", "clubs", "spades"]
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]

# Simulated travel cost: seconds per unit of distance.  An agent moving across
# the full diagonal of the 10x10 grid (~14.14 units) would spend ~0.07s.  This
# makes parallelism meaningful because total travel time dominates process overhead.
TRAVEL_COST_PER_UNIT = 0.005


def scatter_node(state: AppState) -> AppState:
    """Scatter a fresh deck of 52 cards onto a 10×10 grid.

    Each card receives a random `(x, y)` position in the unit square scaled to
    `[0.0, 10.0]`.  Cards are marked as unpicked.  This function only runs
    once, immediately after the START node.  When finished it switches the
    phase to 'pickup'.
    """
    cards: List[Card] = []
    for suit in SUITS:
        for rank in RANKS:
            card: Card = {
                "suit": suit,
                "rank": rank,
                "x": random.random() * 10.0,
                "y": random.random() * 10.0,
                "picked_up": False,
                "picked_up_by": None,
            }
            cards.append(card)
    # update state
    state["cards"] = cards
    state["phase"] = "pickup"
    return state


def _analyze_scatter(cards: List[Card]) -> dict:
    """Compute spatial metrics about the card distribution.

    Returns a dictionary with metrics the supervisor LLM uses to decide
    how many pickup agents to deploy.
    """
    xs = [c["x"] for c in cards]
    ys = [c["y"] for c in cards]

    # Card counts per quadrant
    quadrants = [0, 0, 0, 0]  # Q0: x<5,y<5  Q1: x>=5,y<5  Q2: x<5,y>=5  Q3: x>=5,y>=5
    for c in cards:
        left = c["x"] < 5.0
        lower = c["y"] < 5.0
        if left and lower:
            quadrants[0] += 1
        elif not left and lower:
            quadrants[1] += 1
        elif left and not lower:
            quadrants[2] += 1
        else:
            quadrants[3] += 1

    # Left/right split
    left_count = quadrants[0] + quadrants[2]
    right_count = quadrants[1] + quadrants[3]

    # Spatial spread
    std_x = statistics.stdev(xs)
    std_y = statistics.stdev(ys)

    # Average nearest-neighbor distance
    nn_distances = []
    for i, c in enumerate(cards):
        min_dist = float("inf")
        for j, other in enumerate(cards):
            if i == j:
                continue
            d = math.hypot(c["x"] - other["x"], c["y"] - other["y"])
            if d < min_dist:
                min_dist = d
        nn_distances.append(min_dist)
    avg_nn_dist = statistics.mean(nn_distances)

    # Balance ratio: min quadrant count / max quadrant count
    q_min = min(quadrants)
    q_max = max(quadrants)
    balance_ratio = q_min / q_max if q_max > 0 else 0.0

    return {
        "total_cards": len(cards),
        "quadrant_counts": {
            "Q0_bottom_left": quadrants[0],
            "Q1_bottom_right": quadrants[1],
            "Q2_top_left": quadrants[2],
            "Q3_top_right": quadrants[3],
        },
        "left_right_split": {"left": left_count, "right": right_count},
        "spatial_spread": {"std_x": round(std_x, 3), "std_y": round(std_y, 3)},
        "avg_nearest_neighbor_distance": round(avg_nn_dist, 3),
        "quadrant_balance_ratio": round(balance_ratio, 3),
    }


SUPERVISOR_SYSTEM_PROMPT = """\
You are a supervisor agent in a 52 Card Pickup simulation. Your job is to \
decide how many pickup agents to deploy (1, 2, or 4) based on the spatial \
distribution of cards on a 10x10 grid.

Each agent starts at (0, 0) and picks up cards by traveling to the nearest \
unpicked card in its assigned region. Travel takes real time proportional to \
distance (0.005 seconds per unit). With multiple agents, each agent only \
covers its own region and agents run in parallel:
- 2 agents: grid split into left (x<5) and right (x>=5) halves.
- 4 agents: grid split into four quadrants.

Guidelines:
- 1 agent: best when cards are tightly clustered near the origin or in one area.
- 2 agents: good when cards are spread across the grid with a roughly even \
left/right split. Saves travel time by halving each agent's territory.
- 4 agents: good when cards are spread evenly across all four quadrants. \
Each agent covers a smaller area, reducing total travel significantly.

More agents add a small fixed overhead (~20ms for process spawning), but the \
travel time savings from shorter distances within each region usually outweigh \
this when cards are spread out.

Respond with ONLY a JSON object (no markdown, no extra text):
{"agents": <1|2|4>, "reasoning": "<one or two sentences explaining your choice>"}
"""


def supervisor_node(state: AppState) -> AppState:
    """LLM-powered supervisor that decides how many pickup agents to deploy.

    Analyzes the scatter pattern and calls Claude to make a strategic decision
    about resource allocation.  Falls back to 2 agents if the API call fails.
    """
    cards = state["cards"]
    metrics = _analyze_scatter(cards)

    user_message = (
        "Here is the spatial analysis of the current card scatter:\n\n"
        + json.dumps(metrics, indent=2)
        + "\n\nHow many pickup agents should I deploy?"
    )

    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            system=SUPERVISOR_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        reply = response.content[0].text.strip()
        decision = json.loads(reply)
        num_agents = int(decision["agents"])
        reasoning = decision.get("reasoning", "No reasoning provided.")
        if num_agents not in (1, 2, 4):
            reasoning = f"LLM suggested {num_agents} agents (invalid). Falling back to 2. Original reasoning: {reasoning}"
            num_agents = 2
    except Exception as e:
        num_agents = 2
        reasoning = f"Supervisor API call failed ({type(e).__name__}: {e}). Falling back to 2 agents."

    state["pickup_agents"] = num_agents
    state["supervisor_reasoning"] = reasoning
    return state


def timer_start_node(state: AppState) -> AppState:
    """Record the start time of the pickup phase."""
    state["start_time"] = time.perf_counter()
    return state


def _determine_region(card: Card, num_agents: int) -> int:
    """Assign a card to a region based on its coordinates and number of agents.

    With 1 agent all cards fall into region 0.  With 2 agents the plane is
    divided vertically at x = 5.  With 4 agents the plane is divided into
    quadrants at x = 5 and y = 5.  This deterministic partitioning ensures
    agents operate on disjoint subsets of the deck.
    """
    if num_agents <= 1:
        return 0
    # two agents: left (<5) and right (>=5)
    if num_agents == 2:
        return 0 if card["x"] < 5.0 else 1
    # four agents: quadrants
    # index 0: x < 5, y < 5; 1: x >= 5, y < 5; 2: x < 5, y >= 5; 3: x >= 5, y >= 5
    if num_agents == 4:
        left = card["x"] < 5.0
        lower = card["y"] < 5.0
        if left and lower:
            return 0
        elif not left and lower:
            return 1
        elif left and not lower:
            return 2
        else:
            return 3
    # fallback: assign all cards to region 0 for unsupported agent counts
    return 0


def _pickup_region(args: Tuple[int, List[Tuple[int, float, float]], str]) -> List[int]:
    """Internal helper executed in a separate process for one agent's region.

    The function receives a tuple `(region_id, positions, agent_id)` where
    `positions` is a list of `(card_index, x, y)` tuples belonging to this
    agent’s region.  It simulates picking up cards by repeatedly selecting
    the nearest unpicked card to the agent’s current position (starting
    from the origin) until none remain.  It returns the ordered list of
    card indices to update in the main process.  The heavy work of
    computing distances happens here so that different regions can run
    concurrently across multiple processes.
    """
    region_id, positions, agent_id = args
    # local copy of unpicked positions; each element is (index, x, y)
    remaining = list(positions)
    current_x, current_y = 0.0, 0.0
    pickup_order: List[int] = []
    while remaining:
        # find the nearest card to the current position
        nearest_idx = None
        nearest_dist = float("inf")
        nearest_pos = None
        for idx, (card_index, x, y) in enumerate(remaining):
            # Euclidean distance
            dist = math.hypot(x - current_x, y - current_y)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_idx = idx
                nearest_pos = (card_index, x, y)
        assert nearest_idx is not None and nearest_pos is not None
        # Simulate physical travel time proportional to distance
        time.sleep(nearest_dist * TRAVEL_COST_PER_UNIT)
        # record the card index and update current position
        pickup_order.append(nearest_pos[0])
        current_x, current_y = nearest_pos[1], nearest_pos[2]
        # remove from remaining
        remaining.pop(nearest_idx)
    return pickup_order


def pickup_node(state: AppState) -> AppState:
    """Pick up all scattered cards using one or more concurrent agents.

    This node inspects `state['pickup_agents']` to determine how many
    concurrent pickup threads/processes to run.  The cards are partitioned
    into disjoint regions of the grid (whole, halves or quadrants).  Each
    region is processed in parallel by a separate worker using
    `concurrent.futures.ProcessPoolExecutor`.  Each worker picks up its
    assigned cards by greedy nearest‑neighbour search and returns the
    ordered list of card indices to mark as picked.  Once all workers
    complete, this node updates the main card list with `picked_up=True`
    and the `picked_up_by` agent identifier.  Finally it switches the
    phase to 'verify'.
    """
    cards = state["cards"]
    num_agents = max(1, int(state.get("pickup_agents", 1)))

    # Partition cards into regions
    region_positions: List[List[Tuple[int, float, float]]] = [[] for _ in range(num_agents)]
    for idx, card in enumerate(cards):
        region_id = _determine_region(card, num_agents)
        if 0 <= region_id < num_agents:
            region_positions[region_id].append((idx, card["x"], card["y"]))
        else:
            # default fallback
            region_positions[0].append((idx, card["x"], card["y"]))

    # Prepare arguments for each worker
    tasks: List[Tuple[int, List[Tuple[int, float, float]], str]] = []
    for region_id, positions in enumerate(region_positions):
        agent_id = f"agent_{region_id}"
        tasks.append((region_id, positions, agent_id))

    # Use a ProcessPoolExecutor only if multiple agents are requested; using
    # processes circumvents Python’s GIL for CPU‑bound work.  For a single
    # agent we run the pickup synchronously in the main process.
    pickup_results: List[Tuple[int, List[int]]] = []  # (region_id, order list)
    if num_agents > 1 and any(region_positions):
        with ProcessPoolExecutor(max_workers=num_agents) as executor:
            future_to_region = {
                executor.submit(_pickup_region, task): task[0] for task in tasks
            }
            for future in as_completed(future_to_region):
                region_id = future_to_region[future]
                order = future.result()
                pickup_results.append((region_id, order))
    else:
        # Synchronous execution for one agent or no cards
        for region_id, positions, agent_id in tasks:
            order = _pickup_region((region_id, positions, agent_id))
            pickup_results.append((region_id, order))

    # Mark picked cards in the main state; preserve the pick ordering per agent
    for region_id, order in pickup_results:
        agent_id = f"agent_{region_id}"
        for card_idx in order:
            card = cards[card_idx]
            card["picked_up"] = True
            card["picked_up_by"] = agent_id

    state["cards"] = cards
    state["phase"] = "verify"
    return state


def timer_stop_node(state: AppState) -> AppState:
    """Record the end time of the pickup phase."""
    state["end_time"] = time.perf_counter()
    return state


def verify_node(state: AppState) -> AppState:
    """Validate the final state and record a PASS/FAIL message.

    The verifier checks that there are exactly 52 cards, all of which are
    marked as picked up, and that every suit–rank combination appears
    exactly once.  Any discrepancy results in a FAIL message with
    details; otherwise a PASS message is stored in the `result` field.
    """
    cards = state["cards"]
    result_lines = []
    # Check card count
    if len(cards) != 52:
        result_lines.append(f"Expected 52 cards, found {len(cards)}.")
    # Check uniqueness of suit-rank pairs
    seen = set()
    for card in cards:
        key = (card["suit"], card["rank"])
        if key in seen:
            result_lines.append(f"Duplicate card detected: {card['rank']} of {card['suit']}.")
        seen.add(key)
        if not card["picked_up"]:
            result_lines.append(f"Card not picked up: {card['rank']} of {card['suit']}.")
    # Compose result message
    if result_lines:
        state["result"] = "FAIL: " + "; ".join(result_lines)
    else:
        state["result"] = "PASS"
    state["phase"] = "done"
    return state


def build_graph(with_supervisor: bool = False) -> StateGraph:
    """Assemble and compile the LangGraph for the card pickup simulation.

    When *with_supervisor* is True the graph includes an LLM-powered supervisor
    node between scatter and timer_start that decides how many pickup agents to
    deploy.  Otherwise the graph is the same deterministic Phase 1 graph.
    """
    builder: StateGraph = StateGraph(AppState)
    # register nodes
    builder.add_node("scatter", scatter_node)
    if with_supervisor:
        builder.add_node("supervisor", supervisor_node)
    builder.add_node("timer_start", timer_start_node)
    builder.add_node("pickup", pickup_node)
    builder.add_node("timer_stop", timer_stop_node)
    builder.add_node("verify", verify_node)
    # define edges
    builder.add_edge(START, "scatter")
    if with_supervisor:
        builder.add_edge("scatter", "supervisor")
        builder.add_edge("supervisor", "timer_start")
    else:
        builder.add_edge("scatter", "timer_start")
    builder.add_edge("timer_start", "pickup")
    builder.add_edge("pickup", "timer_stop")
    builder.add_edge("timer_stop", "verify")
    builder.add_edge("verify", END)
    # compile into a runnable graph
    return builder.compile()


def _make_initial_state(num_agents: int = 1) -> AppState:
    """Create a fresh initial state for a simulation run."""
    return {
        "cards": [],
        "phase": "scatter",
        "start_time": None,
        "end_time": None,
        "pickup_agents": num_agents,
        "supervisor_reasoning": None,
        "result": None,
    }


def _extract_elapsed(final_state: AppState) -> float:
    """Compute elapsed pickup time from a completed state."""
    start = final_state["start_time"]
    end = final_state["end_time"]
    if start is not None and end is not None:
        return end - start
    return float("nan")


def run_trials(num_agents: int, trials: int = 10) -> Tuple[List[float], int]:
    """Execute multiple runs of the simulation and gather timing statistics.

    :param num_agents: the number of concurrent pickup agents
    :param trials: how many times to run the simulation
    :returns: a tuple containing the list of elapsed times and the count of
              successful verifier passes.
    """
    times: List[float] = []
    passes = 0
    graph = build_graph(with_supervisor=False)
    for i in range(trials):
        seed = 42 + i
        random.seed(seed)
        final_state = graph.invoke(_make_initial_state(num_agents))
        times.append(_extract_elapsed(final_state))
        if final_state.get("result", "").startswith("PASS"):
            passes += 1
    return times, passes


def run_supervisor_comparison(trials: int = 5) -> None:
    """Run the supervisor experiment and compare against brute-force configs.

    For each trial (with a fixed seed), the supervisor picks an agent count,
    then we also run all three brute-force configurations with the same seed
    to see which was actually fastest.
    """
    supervisor_graph = build_graph(with_supervisor=True)
    brute_graph = build_graph(with_supervisor=False)

    print("\n=== Phase 2: Supervisor vs. Brute-Force Comparison ===\n")
    header = ["Trial", "Seed", "Supervisor Choice", "Supervisor Time (s)",
              "Best Brute-Force", "Best BF Time (s)", "Match?"]
    print("| " + " | ".join(header) + " |")
    print("|" + "--------|" * len(header))

    matches = 0
    for i in range(trials):
        seed = 42 + i

        # --- Supervisor run ---
        random.seed(seed)
        sup_state = supervisor_graph.invoke(_make_initial_state(num_agents=1))
        sup_elapsed = _extract_elapsed(sup_state)
        sup_agents = sup_state["pickup_agents"]
        sup_reasoning = sup_state.get("supervisor_reasoning", "")
        sup_passed = sup_state.get("result", "").startswith("PASS")

        # --- Brute-force runs (same seed each time) ---
        bf_results: List[Tuple[int, float, bool]] = []
        for n in (1, 2, 4):
            random.seed(seed)
            bf_state = brute_graph.invoke(_make_initial_state(n))
            bf_elapsed = _extract_elapsed(bf_state)
            bf_passed = bf_state.get("result", "").startswith("PASS")
            bf_results.append((n, bf_elapsed, bf_passed))

        best_bf = min(bf_results, key=lambda r: r[1])
        matched = sup_agents == best_bf[0]
        if matched:
            matches += 1

        print(
            f"| {i+1} | {seed} | {sup_agents} agents | {sup_elapsed:.4f} "
            f"| {best_bf[0]} agents | {best_bf[1]:.4f} "
            f"| {'Yes' if matched else 'No'} |"
        )
        print(f"  Reasoning: {sup_reasoning}")
        if not sup_passed:
            print(f"  WARNING: Supervisor run verifier did not pass!")

    print(f"\nSupervisor matched optimal: {matches}/{trials}")


def print_summary(results: List[Tuple[int, List[float], int]]) -> None:
    """Pretty-print a summary table for the scaling experiment."""
    header = ["Agents", "Avg Time (s)", "Best (s)", "Worst (s)", "Verifier"]
    print("| " + " | ".join(header) + " |")
    print("|" + "--------|" * len(header))
    for num_agents, times, passes in results:
        if times:
            avg_time = sum(times) / len(times)
            best_time = min(times)
            worst_time = max(times)
        else:
            avg_time = best_time = worst_time = float("nan")
        verifier_str = f"{passes}/{len(times)}" + (" ✓" if passes == len(times) else " ✗")
        print(
            f"| {num_agents} | {avg_time:.4f} | {best_time:.4f} | {worst_time:.4f} | {verifier_str} |"
        )


def main() -> None:
    """Run the Phase 1 scaling experiment and the Phase 2 supervisor comparison."""
    # Phase 1: brute-force scaling experiment
    print("=== Phase 1: Brute-Force Scaling Experiment ===\n")
    configurations = [1, 2, 4]
    summary_results: List[Tuple[int, List[float], int]] = []
    for num_agents in configurations:
        times, passes = run_trials(num_agents)
        summary_results.append((num_agents, times, passes))
    print_summary(summary_results)

    # Phase 2: supervisor comparison
    if os.environ.get("ANTHROPIC_API_KEY"):
        run_supervisor_comparison(trials=5)
    else:
        print("\n=== Phase 2: Skipped (ANTHROPIC_API_KEY not set) ===")


if __name__ == "__main__":
    main()