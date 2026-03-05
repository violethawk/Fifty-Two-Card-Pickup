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

import argparse
import json
import math
import os
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, TypedDict, Tuple

import anthropic
from langgraph.graph import END, START, StateGraph

from observability import (
    AnomalyDetector,
    Dashboard,
    EventLog,
    GovernanceChecker,
    GovernanceViolation,
    MetricsCalculator,
    replay_event_log,
)

# Module-level optional dashboard — pickup nodes check this for live updates.
_active_dashboard: Optional[Dashboard] = None
_active_event_log: Optional[EventLog] = None


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
    agent_positions: Optional[Dict[str, List[float]]]  # current (x, y) per agent
    agent_intentions: Optional[Dict[str, str]]  # announced next target per agent
    agent_strategies: Optional[Dict[str, str]]  # strategy explanation per agent
    llm_calls: int           # total LLM API calls during pickup
    conflicts_resolved: int  # total conflicts detected and resolved
    total_input_tokens: int  # cumulative input tokens for cost tracking
    total_output_tokens: int # cumulative output tokens for cost tracking
    event_log: Optional[List[dict]]  # serialized event records (Phase 4)
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
    # Emit scatter event
    if _active_event_log is not None:
        _active_event_log.emit("scatter", data={
            "cards": [{"rank": c["rank"], "suit": c["suit"],
                       "x": round(c["x"], 2), "y": round(c["y"], 2)} for c in cards]
        })
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
        if reply.startswith("```"):
            reply = reply.split("\n", 1)[1] if "\n" in reply else reply[3:]
            if reply.endswith("```"):
                reply = reply[:-3].strip()
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


PICKUP_AGENT_SYSTEM_PROMPT = """\
You are a pickup agent in a 52 Card Pickup simulation on a 10x10 grid.
You are at position ({agent_x:.1f}, {agent_y:.1f}).

Plan your next moves to efficiently collect cards. Each move costs time \
proportional to the distance traveled (0.005s per unit).

Prioritize:
- Cards close to your current position (minimize travel distance)
- Clusters of nearby cards (plan a path through them)
- Cards that other agents are NOT heading toward (check their intentions)

Respond with ONLY a JSON object (no markdown, no extra text):
{{"targets": ["<rank> of <suit>", ...], "strategy": "<brief explanation>"}}

List 5-10 cards in the order you want to collect them. Use exact card names \
like "7 of hearts" or "K of spades".
"""


def _card_key(card: Card) -> str:
    """Return a human-readable identifier for a card."""
    return f"{card['rank']} of {card['suit']}"


def _find_card_by_key(cards: List[Card], key: str) -> Optional[int]:
    """Find the index of an unpicked card matching the given key string."""
    for idx, card in enumerate(cards):
        if not card["picked_up"] and _card_key(card) == key:
            return idx
    return None


def _greedy_nearest_card(cards: List[Card], x: float, y: float) -> Optional[int]:
    """Fallback: find the nearest unpicked card to (x, y)."""
    best_idx = None
    best_dist = float("inf")
    for idx, card in enumerate(cards):
        if card["picked_up"]:
            continue
        d = math.hypot(card["x"] - x, card["y"] - y)
        if d < best_dist:
            best_dist = d
            best_idx = idx
    return best_idx


def _plan_agent_moves(
    client: anthropic.Anthropic,
    agent_id: str,
    agent_x: float,
    agent_y: float,
    cards: List[Card],
    all_positions: Dict[str, List[float]],
    all_intentions: Dict[str, str],
) -> Tuple[List[str], str, int, int]:
    """Call Haiku to plan the next batch of moves for one agent.

    Returns (target_keys, strategy, input_tokens, output_tokens).
    On failure, returns a greedy fallback plan.
    """
    remaining = [c for c in cards if not c["picked_up"]]
    remaining_desc = [
        f"  {_card_key(c)} at ({c['x']:.1f}, {c['y']:.1f})"
        for c in remaining
    ]

    other_info = []
    for aid, pos in all_positions.items():
        if aid == agent_id:
            continue
        intention = all_intentions.get(aid, "none")
        other_info.append(
            f"  {aid} at ({pos[0]:.1f}, {pos[1]:.1f}), heading toward: {intention}"
        )

    user_message = (
        f"Remaining cards ({len(remaining)}):\n"
        + "\n".join(remaining_desc)
        + "\n\nOther agents:\n"
        + ("\n".join(other_info) if other_info else "  (none)")
    )

    system = PICKUP_AGENT_SYSTEM_PROMPT.format(agent_x=agent_x, agent_y=agent_y)

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        reply = response.content[0].text.strip()
        # Strip markdown code fences if present
        if reply.startswith("```"):
            reply = reply.split("\n", 1)[1] if "\n" in reply else reply[3:]
            if reply.endswith("```"):
                reply = reply[:-3].strip()
        decision = json.loads(reply)
        targets = decision.get("targets", [])
        strategy = decision.get("strategy", "No strategy provided.")
        if not targets:
            raise ValueError("Empty targets list")
        return targets, strategy, input_tokens, output_tokens
    except Exception:
        # Fallback: greedy nearest-neighbor plan
        fallback_targets = []
        fx, fy = agent_x, agent_y
        temp_remaining = list(remaining)
        for _ in range(min(10, len(temp_remaining))):
            best = None
            best_dist = float("inf")
            for c in temp_remaining:
                d = math.hypot(c["x"] - fx, c["y"] - fy)
                if d < best_dist:
                    best_dist = d
                    best = c
            if best is None:
                break
            fallback_targets.append(_card_key(best))
            fx, fy = best["x"], best["y"]
            temp_remaining.remove(best)
        return fallback_targets, "Fallback: greedy nearest-neighbor", 0, 0


def _resolve_conflicts(
    agent_targets: Dict[str, str],
    agent_positions: Dict[str, List[float]],
    cards: List[Card],
) -> Dict[str, Optional[int]]:
    """Resolve conflicts when multiple agents target the same card.

    Returns a mapping of agent_id -> card_index (or None if agent must wait).
    Closest agent wins; ties broken by agent_id (lexicographic).
    """
    # Group agents by their target
    target_to_agents: Dict[str, List[str]] = {}
    for agent_id, target_key in agent_targets.items():
        target_to_agents.setdefault(target_key, []).append(agent_id)

    resolved: Dict[str, Optional[int]] = {}
    claimed_indices: set = set()

    for target_key, competing_agents in target_to_agents.items():
        card_idx = _find_card_by_key(cards, target_key)
        if card_idx is None:
            # Card doesn't exist or already picked — all agents miss
            for aid in competing_agents:
                resolved[aid] = None
            continue

        if len(competing_agents) == 1:
            resolved[competing_agents[0]] = card_idx
            claimed_indices.add(card_idx)
        else:
            # Conflict: closest agent wins
            card = cards[card_idx]
            competing_agents.sort(key=lambda aid: (
                math.hypot(
                    card["x"] - agent_positions[aid][0],
                    card["y"] - agent_positions[aid][1],
                ),
                aid,  # tiebreaker
            ))
            resolved[competing_agents[0]] = card_idx
            claimed_indices.add(card_idx)
            for aid in competing_agents[1:]:
                resolved[aid] = None

    return resolved


def llm_pickup_node(state: AppState) -> AppState:
    """Pick up cards using LLM-powered agents with planning and conflict resolution.

    Each round: agents plan (LLM call), broadcast intentions, resolve conflicts
    deterministically, then execute one pickup each. Repeats until all cards collected.
    """
    cards = state["cards"]
    num_agents = max(1, int(state.get("pickup_agents", 1)))
    client = anthropic.Anthropic()

    agent_ids = [f"agent_{i}" for i in range(num_agents)]
    positions: Dict[str, List[float]] = {aid: [0.0, 0.0] for aid in agent_ids}
    intentions: Dict[str, str] = {aid: "none" for aid in agent_ids}
    strategies: Dict[str, str] = {}

    llm_calls = 0
    conflicts_resolved = 0
    total_input_tokens = 0
    total_output_tokens = 0

    elog = _active_event_log
    dash = _active_dashboard
    governance = GovernanceChecker(elog, agent_ids) if elog else None

    max_rounds = 200  # safety limit to prevent infinite loops
    round_num = 0

    while any(not c["picked_up"] for c in cards) and round_num < max_rounds:
        round_num += 1

        # Step 1: Plan — each agent calls Haiku for a batch of targets
        agent_plans: Dict[str, List[str]] = {}
        for aid in agent_ids:
            targets, strategy, in_tok, out_tok = _plan_agent_moves(
                client, aid,
                positions[aid][0], positions[aid][1],
                cards, positions, intentions,
            )
            agent_plans[aid] = targets
            strategies[aid] = strategy
            llm_calls += 1
            total_input_tokens += in_tok
            total_output_tokens += out_tok
            if elog:
                elog.emit("plan", agent_id=aid, data={
                    "targets": targets[:3],
                    "strategy": strategy[:100],
                })

        # Step 2: Broadcast — each agent announces first valid target
        round_targets: Dict[str, str] = {}
        for aid in agent_ids:
            for target_key in agent_plans[aid]:
                if _find_card_by_key(cards, target_key) is not None:
                    round_targets[aid] = target_key
                    break
            # If no valid target found in plan, try greedy fallback
            if aid not in round_targets:
                fallback_idx = _greedy_nearest_card(
                    cards, positions[aid][0], positions[aid][1]
                )
                if fallback_idx is not None:
                    round_targets[aid] = _card_key(cards[fallback_idx])

        if not round_targets:
            break  # no more cards to pick up

        intentions = {aid: round_targets.get(aid, "none") for aid in agent_ids}

        for aid, target in round_targets.items():
            if elog:
                elog.emit("broadcast", agent_id=aid, data={"target": target})

        # Step 3: Resolve conflicts
        resolved = _resolve_conflicts(round_targets, positions, cards)

        # Count conflicts (agents that got None despite having a target)
        for aid in round_targets:
            if resolved.get(aid) is None:
                conflicts_resolved += 1
                if elog:
                    # Find who won the card this agent wanted
                    wanted = round_targets[aid]
                    winner = None
                    for other_aid, other_idx in resolved.items():
                        if other_idx is not None and other_aid != aid:
                            c = cards[other_idx]
                            if _card_key(c) == wanted:
                                winner = other_aid
                                break
                    elog.emit("conflict", data={
                        "card": wanted,
                        "winner": winner or "unknown",
                        "losers": [aid],
                    })
                # Try fallback: walk down the plan for an unclaimed card
                claimed = {idx for idx in resolved.values() if idx is not None}
                for target_key in agent_plans[aid]:
                    idx = _find_card_by_key(cards, target_key)
                    if idx is not None and idx not in claimed:
                        resolved[aid] = idx
                        claimed.add(idx)
                        break
                # Last resort: greedy
                if resolved.get(aid) is None:
                    for idx, c in enumerate(cards):
                        if not c["picked_up"] and idx not in claimed:
                            resolved[aid] = idx
                            claimed.add(idx)
                            break

        # Step 4: Execute — agents travel and pick up their resolved cards
        active_agents = []
        idle_agents = []
        for aid in agent_ids:
            card_idx = resolved.get(aid)
            if card_idx is None:
                idle_agents.append(aid)
                continue
            active_agents.append(aid)
            card = cards[card_idx]
            dist = math.hypot(
                card["x"] - positions[aid][0],
                card["y"] - positions[aid][1],
            )
            time.sleep(dist * TRAVEL_COST_PER_UNIT)
            card["picked_up"] = True
            card["picked_up_by"] = aid
            positions[aid] = [card["x"], card["y"]]
            if elog:
                elog.emit("pickup", agent_id=aid, data={
                    "card": _card_key(card),
                    "distance": round(dist, 3),
                    "position": [round(card["x"], 2), round(card["y"], 2)],
                })

        # Emit round summary event
        if elog:
            picked_count = sum(1 for c in cards if c["picked_up"])
            elog.emit("round", data={
                "round": round_num,
                "picked": picked_count,
                "active_agents": active_agents,
                "idle_agents": idle_agents,
            })

        # Governance check
        if governance:
            governance.check(cards)  # raises GovernanceViolation on failure

        # Dashboard update
        if dash:
            picked_count = sum(1 for c in cards if c["picked_up"])
            dash.update(
                cards, positions, picked_count, 52,
                elog.events[-10:] if elog else [],
                {"elapsed": time.perf_counter() - (state.get("start_time") or 0),
                 "round": round_num,
                 "conflicts": conflicts_resolved},
            )

    state["cards"] = cards
    state["phase"] = "verify"
    state["agent_positions"] = positions
    state["agent_intentions"] = intentions
    state["agent_strategies"] = strategies
    state["llm_calls"] = state.get("llm_calls", 0) + llm_calls
    state["conflicts_resolved"] = state.get("conflicts_resolved", 0) + conflicts_resolved
    state["total_input_tokens"] = state.get("total_input_tokens", 0) + total_input_tokens
    state["total_output_tokens"] = state.get("total_output_tokens", 0) + total_output_tokens
    if elog:
        state["event_log"] = elog.serialize()
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
    _region_id, positions, _agent_id = args
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
        prev_x, prev_y = 0.0, 0.0
        for card_idx in order:
            card = cards[card_idx]
            dist = math.hypot(card["x"] - prev_x, card["y"] - prev_y)
            card["picked_up"] = True
            card["picked_up_by"] = agent_id
            if _active_event_log is not None:
                _active_event_log.emit("pickup", agent_id=agent_id, data={
                    "card": _card_key(card),
                    "distance": round(dist, 3),
                    "position": [round(card["x"], 2), round(card["y"], 2)],
                })
            prev_x, prev_y = card["x"], card["y"]

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
    if _active_event_log is not None:
        _active_event_log.emit("verify", data={"result": state["result"]})
        state["event_log"] = _active_event_log.serialize()
    return state


def build_graph(with_supervisor: bool = False, llm_pickup: bool = False) -> StateGraph:
    """Assemble and compile the LangGraph for the card pickup simulation.

    When *with_supervisor* is True the graph includes an LLM-powered supervisor
    node between scatter and timer_start that decides how many pickup agents to
    deploy.  When *llm_pickup* is True the pickup node uses LLM-powered agents
    with planning and conflict resolution instead of deterministic greedy pickup.
    """
    builder: StateGraph = StateGraph(AppState)
    # register nodes
    builder.add_node("scatter", scatter_node)
    if with_supervisor:
        builder.add_node("supervisor", supervisor_node)
    builder.add_node("timer_start", timer_start_node)
    builder.add_node("pickup", llm_pickup_node if llm_pickup else pickup_node)
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
        "agent_positions": None,
        "agent_intentions": None,
        "agent_strategies": None,
        "llm_calls": 0,
        "conflicts_resolved": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "event_log": None,
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


HAIKU_INPUT_COST_PER_M = 0.80   # dollars per million input tokens
HAIKU_OUTPUT_COST_PER_M = 4.00  # dollars per million output tokens


def _estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """Estimate API cost in dollars for Haiku usage."""
    return (
        input_tokens * HAIKU_INPUT_COST_PER_M / 1_000_000
        + output_tokens * HAIKU_OUTPUT_COST_PER_M / 1_000_000
    )


def run_llm_comparison(trials: int = 3) -> None:
    """Run LLM-powered agents and compare against deterministic Phase 1.

    For each trial, runs both the LLM pickup (with supervisor) and the
    deterministic brute-force (best of 1/2/4 agents) with the same seed.
    """
    llm_graph = build_graph(with_supervisor=True, llm_pickup=True)
    brute_graph = build_graph(with_supervisor=False, llm_pickup=False)

    print("\n=== Phase 3: LLM Agents vs. Deterministic Comparison ===\n")
    header = ["Trial", "LLM Agents", "LLM Time (s)", "BF Best Time (s)",
              "LLM Calls", "Conflicts", "Est. Cost", "Verifier"]
    print("| " + " | ".join(header) + " |")
    print("|" + "--------|" * len(header))

    for i in range(trials):
        seed = 42 + i

        # --- LLM pickup run ---
        random.seed(seed)
        llm_state = llm_graph.invoke(_make_initial_state(num_agents=1))
        llm_elapsed = _extract_elapsed(llm_state)
        llm_agents = llm_state["pickup_agents"]
        llm_calls = llm_state.get("llm_calls", 0)
        conflicts = llm_state.get("conflicts_resolved", 0)
        in_tok = llm_state.get("total_input_tokens", 0)
        out_tok = llm_state.get("total_output_tokens", 0)
        cost = _estimate_cost(in_tok, out_tok)
        llm_passed = llm_state.get("result", "").startswith("PASS")

        # --- Brute-force baseline (same seed) ---
        bf_times = []
        for n in (1, 2, 4):
            random.seed(seed)
            bf_state = brute_graph.invoke(_make_initial_state(n))
            bf_times.append(_extract_elapsed(bf_state))
        best_bf = min(bf_times)

        verifier_str = "PASS" if llm_passed else "FAIL"
        print(
            f"| {i+1} | {llm_agents} | {llm_elapsed:.4f} | {best_bf:.4f} "
            f"| {llm_calls} | {conflicts} | ${cost:.4f} | {verifier_str} |"
        )

        reasoning = llm_state.get("supervisor_reasoning", "")
        if reasoning:
            print(f"  Supervisor: {reasoning}")

        strategies = llm_state.get("agent_strategies", {})
        for aid, strat in sorted(strategies.items()):
            print(f"  {aid} strategy: {strat}")

        if not llm_passed:
            print(f"  WARNING: {llm_state.get('result', '')}")

    print(f"\n  Token usage last run: {in_tok} input, {out_tok} output")


def _run_with_observability(
    graph,
    initial_state: AppState,
    dashboard: bool = False,
    save_log: Optional[str] = None,
    label: str = "",
) -> AppState:
    """Run a graph invocation with event logging, governance, dashboard, and metrics."""
    global _active_event_log, _active_dashboard

    elog = EventLog()
    _active_event_log = elog

    dash = None
    if dashboard:
        dash = Dashboard()
        dash.start()
        _active_dashboard = dash

    try:
        final_state = graph.invoke(initial_state)
    except GovernanceViolation as e:
        print(f"\n  GOVERNANCE VIOLATION: {e}")
        final_state = initial_state
    finally:
        if dash:
            dash.stop()
        _active_dashboard = None
        _active_event_log = None

    elapsed = _extract_elapsed(final_state)

    # Metrics
    if elog.events:
        mc = MetricsCalculator(elog, elapsed)
        mc.print_summary()

        # Anomaly detection
        num_agents = final_state.get("pickup_agents", 1)
        ad = AnomalyDetector(elog, num_agents)
        ad.print_warnings()

    # Save event log if requested
    if save_log:
        elog.save(save_log)
        print(f"\n  Event log saved to {save_log}")

    return final_state


def main() -> None:
    """Run all phase experiments."""
    parser = argparse.ArgumentParser(description="52 Card Pickup — Multi-Agent Simulation")
    parser.add_argument("--dashboard", action="store_true",
                        help="Enable live terminal TUI dashboard")
    parser.add_argument("--replay", type=str, metavar="FILE",
                        help="Replay a saved event log through the dashboard")
    parser.add_argument("--save-log", action="store_true",
                        help="Save event logs to JSON files")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=0,
                        help="Run only the specified phase (default: all)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run the benchmark suite across all scatter patterns")
    parser.add_argument("--strategy", type=str, choices=["greedy", "llm"],
                        default="greedy",
                        help="Pickup strategy plugin (default: greedy)")
    parser.add_argument("--provider", type=str, choices=["anthropic", "mock"],
                        default="anthropic",
                        help="LLM provider plugin (default: anthropic)")
    args = parser.parse_args()

    # Replay mode
    if args.replay:
        replay_event_log(args.replay)
        return

    # Benchmark mode
    if args.benchmark:
        from benchmarks import run_benchmarks
        run_benchmarks()
        return

    run_phase1 = args.phase == 0 or args.phase == 1
    run_phase2 = args.phase == 0 or args.phase == 2
    run_phase3 = args.phase == 0 or args.phase == 3
    has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))

    # Phase 1: brute-force scaling experiment
    if run_phase1:
        print("=== Phase 1: Brute-Force Scaling Experiment ===\n")
        configurations = [1, 2, 4]
        summary_results: List[Tuple[int, List[float], int]] = []
        for num_agents in configurations:
            times, passes = run_trials(num_agents)
            summary_results.append((num_agents, times, passes))
        print_summary(summary_results)

    if has_api_key:
        # Phase 2: supervisor comparison
        if run_phase2:
            run_supervisor_comparison(trials=5)

        # Phase 3: LLM agents comparison (with observability)
        if run_phase3:
            print("\n=== Phase 3: LLM Agents vs. Deterministic Comparison ===\n")
            llm_graph = build_graph(with_supervisor=True, llm_pickup=True)
            brute_graph = build_graph(with_supervisor=False, llm_pickup=False)

            for i in range(3):
                seed = 42 + i
                print(f"--- Trial {i+1} (seed={seed}) ---")

                random.seed(seed)
                log_path = f"event_log_trial_{i+1}.json" if args.save_log else None
                llm_state = _run_with_observability(
                    llm_graph,
                    _make_initial_state(num_agents=1),
                    dashboard=args.dashboard,
                    save_log=log_path,
                    label=f"Trial {i+1}",
                )

                llm_elapsed = _extract_elapsed(llm_state)
                llm_agents = llm_state.get("pickup_agents", 1)
                llm_calls = llm_state.get("llm_calls", 0)
                conflicts = llm_state.get("conflicts_resolved", 0)
                in_tok = llm_state.get("total_input_tokens", 0)
                out_tok = llm_state.get("total_output_tokens", 0)
                cost = _estimate_cost(in_tok, out_tok)
                passed = llm_state.get("result", "").startswith("PASS")

                # Brute-force baseline
                bf_times = []
                for n in (1, 2, 4):
                    random.seed(seed)
                    bf_state = brute_graph.invoke(_make_initial_state(n))
                    bf_times.append(_extract_elapsed(bf_state))
                best_bf = min(bf_times)

                print(f"  LLM: {llm_agents} agents, {llm_elapsed:.4f}s, "
                      f"{llm_calls} calls, {conflicts} conflicts, ${cost:.4f}")
                print(f"  Brute-force best: {best_bf:.4f}s")
                print(f"  Verifier: {'PASS' if passed else 'FAIL'}")

                reasoning = llm_state.get("supervisor_reasoning", "")
                if reasoning:
                    print(f"  Supervisor: {reasoning}")
                print()
    else:
        if run_phase2:
            print("\n=== Phase 2: Skipped (ANTHROPIC_API_KEY not set) ===")
        if run_phase3:
            print("=== Phase 3: Skipped (ANTHROPIC_API_KEY not set) ===")


if __name__ == "__main__":
    main()