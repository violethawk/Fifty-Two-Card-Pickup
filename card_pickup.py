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

import math
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional, TypedDict, Tuple

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
    result: Optional[str]    # textual report from the verifier


SUITS = ["hearts", "diamonds", "clubs", "spades"]
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]


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


def build_graph() -> StateGraph:
    """Assemble and compile the LangGraph for the card pickup simulation."""
    builder: StateGraph = StateGraph(AppState)
    # register nodes
    builder.add_node("scatter", scatter_node)
    builder.add_node("timer_start", timer_start_node)
    builder.add_node("pickup", pickup_node)
    builder.add_node("timer_stop", timer_stop_node)
    builder.add_node("verify", verify_node)
    # define edges: START → scatter → timer_start → pickup → timer_stop → verify → END
    builder.add_edge(START, "scatter")
    builder.add_edge("scatter", "timer_start")
    builder.add_edge("timer_start", "pickup")
    builder.add_edge("pickup", "timer_stop")
    builder.add_edge("timer_stop", "verify")
    builder.add_edge("verify", END)
    # compile into a runnable graph
    return builder.compile()


def run_trials(num_agents: int, trials: int = 10) -> Tuple[List[float], int]:
    """Execute multiple runs of the simulation and gather timing statistics.

    :param num_agents: the number of concurrent pickup agents
    :param trials: how many times to run the simulation
    :returns: a tuple containing the list of elapsed times and the count of
              successful verifier passes.
    """
    times: List[float] = []
    passes = 0
    graph = build_graph()
    for i in range(trials):
        # vary the seed for reproducibility across runs; first run uses 42
        seed = 42 + i
        random.seed(seed)
        initial_state: AppState = {
            "cards": [],
            "phase": "scatter",
            "start_time": None,
            "end_time": None,
            "pickup_agents": num_agents,
            "result": None,
        }
        final_state = graph.invoke(initial_state)
        start = final_state["start_time"]
        end = final_state["end_time"]
        # compute elapsed time; guard against missing values
        if start is not None and end is not None:
            elapsed = end - start
        else:
            elapsed = float("nan")
        times.append(elapsed)
        if final_state.get("result", "").startswith("PASS"):
            passes += 1
    return times, passes


def print_summary(results: List[Tuple[int, List[float], int]]) -> None:
    """Pretty‑print a summary table for the scaling experiment."""
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
    """Run the scaling experiment and display results."""
    configurations = [1, 2, 4]
    summary_results: List[Tuple[int, List[float], int]] = []
    for num_agents in configurations:
        times, passes = run_trials(num_agents)
        summary_results.append((num_agents, times, passes))
    print_summary(summary_results)


if __name__ == "__main__":
    main()