"""
observability.py
================

Phase 4 infrastructure: event logging, governance guardrails, performance
metrics, anomaly detection, terminal TUI dashboard, and replay.

All classes operate on or produce event dicts that are stored in
``AppState["event_log"]``.  Nothing here changes agent behaviour — it
only *observes* and *constrains*.
"""

from __future__ import annotations

import curses
import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Event
# ---------------------------------------------------------------------------

@dataclass
class Event:
    """A single recorded action in the simulation."""
    timestamp: float
    event_type: str
    agent_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Event":
        return cls(**d)


# ---------------------------------------------------------------------------
# EventLog
# ---------------------------------------------------------------------------

class EventLog:
    """Ordered collection of events with helpers for query and persistence."""

    def __init__(self) -> None:
        self._events: List[Event] = []
        self._run_start: float = time.perf_counter()

    def set_start(self, t: float) -> None:
        self._run_start = t

    def emit(
        self,
        event_type: str,
        agent_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Event:
        evt = Event(
            timestamp=time.perf_counter() - self._run_start,
            event_type=event_type,
            agent_id=agent_id,
            data=data or {},
        )
        self._events.append(evt)
        return evt

    @property
    def events(self) -> List[Event]:
        return list(self._events)

    def serialize(self) -> List[dict]:
        return [e.to_dict() for e in self._events]

    @classmethod
    def deserialize(cls, data: List[dict]) -> "EventLog":
        log = cls()
        log._events = [Event.from_dict(d) for d in data]
        return log

    def save(self, filepath: str) -> None:
        with open(filepath, "w") as f:
            json.dump(self.serialize(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "EventLog":
        with open(filepath) as f:
            return cls.deserialize(json.load(f))

    def by_type(self, event_type: str) -> List[Event]:
        return [e for e in self._events if e.event_type == event_type]

    def by_agent(self, agent_id: str) -> List[Event]:
        return [e for e in self._events if e.agent_id == agent_id]


# ---------------------------------------------------------------------------
# Governance
# ---------------------------------------------------------------------------

class GovernanceViolation(Exception):
    """Raised when a runtime invariant is violated."""
    pass


class GovernanceChecker:
    """Runtime invariant checks executed after every pickup round."""

    def __init__(self, event_log: EventLog, agent_ids: List[str]) -> None:
        self._log = event_log
        self._agent_ids = set(agent_ids)
        self._prev_picked = 0

    def check(self, cards: List[dict]) -> None:
        """Run all invariant checks.  Raises GovernanceViolation on failure."""
        self._check_card_count(cards)
        self._check_no_double_pickup(cards)
        self._check_no_phantom_agents(cards)
        self._check_monotonic_progress(cards)

    def _check_card_count(self, cards: List[dict]) -> None:
        if len(cards) != 52:
            self._violation(f"Card count changed: expected 52, found {len(cards)}")

    def _check_no_double_pickup(self, cards: List[dict]) -> None:
        picked_keys = []
        for c in cards:
            if c["picked_up"]:
                key = f"{c['rank']} of {c['suit']}"
                picked_keys.append(key)
        if len(picked_keys) != len(set(picked_keys)):
            self._violation("Duplicate pickup detected")

    def _check_no_phantom_agents(self, cards: List[dict]) -> None:
        for c in cards:
            if c["picked_up"] and c["picked_up_by"] not in self._agent_ids:
                self._violation(
                    f"Phantom agent '{c['picked_up_by']}' picked up "
                    f"{c['rank']} of {c['suit']}"
                )

    def _check_monotonic_progress(self, cards: List[dict]) -> None:
        current_picked = sum(1 for c in cards if c["picked_up"])
        if current_picked < self._prev_picked:
            self._violation(
                f"Progress went backwards: {self._prev_picked} -> {current_picked}"
            )
        self._prev_picked = current_picked

    def _violation(self, msg: str) -> None:
        self._log.emit("governance", data={"violation": msg})
        raise GovernanceViolation(msg)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class MetricsCalculator:
    """Compute performance metrics from a completed event log."""

    def __init__(self, event_log: EventLog, elapsed: float) -> None:
        self._log = event_log
        self._elapsed = elapsed

    def compute(self) -> Dict[str, Any]:
        pickups = self._log.by_type("pickup")
        agents: Dict[str, Dict[str, Any]] = {}

        for evt in pickups:
            aid = evt.agent_id or "unknown"
            if aid not in agents:
                agents[aid] = {"cards": 0, "distance": 0.0, "idle_rounds": 0}
            agents[aid]["cards"] += 1
            agents[aid]["distance"] += evt.data.get("distance", 0.0)

        # Idle rounds from round events
        for evt in self._log.by_type("round"):
            idle_agents = evt.data.get("idle_agents", [])
            for aid in idle_agents:
                if aid in agents:
                    agents[aid]["idle_rounds"] += 1

        conflicts = self._log.by_type("conflict")
        total_pickups = len(pickups)

        return {
            "elapsed": self._elapsed,
            "agents": {
                aid: {
                    "cards_picked": d["cards"],
                    "cards_per_second": d["cards"] / self._elapsed if self._elapsed > 0 else 0,
                    "total_distance": round(d["distance"], 2),
                    "idle_rounds": d["idle_rounds"],
                }
                for aid, d in sorted(agents.items())
            },
            "total_conflicts": len(conflicts),
            "conflict_rate": len(conflicts) / total_pickups if total_pickups > 0 else 0,
        }

    def print_summary(self) -> None:
        m = self.compute()
        print(f"\n  Performance Metrics (elapsed: {m['elapsed']:.4f}s)")
        print(f"  {'Agent':<10} {'Cards':>6} {'Cards/s':>8} {'Distance':>10} {'Idle Rounds':>12}")
        print(f"  {'-'*10} {'-'*6} {'-'*8} {'-'*10} {'-'*12}")
        for aid, ad in m["agents"].items():
            print(
                f"  {aid:<10} {ad['cards_picked']:>6} "
                f"{ad['cards_per_second']:>8.1f} "
                f"{ad['total_distance']:>10.2f} "
                f"{ad['idle_rounds']:>12}"
            )
        print(f"  Conflicts: {m['total_conflicts']} (rate: {m['conflict_rate']:.1%})")


# ---------------------------------------------------------------------------
# Anomaly Detection
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """Post-run checks that flag unexpected behaviour."""

    def __init__(self, event_log: EventLog, num_agents: int) -> None:
        self._log = event_log
        self._num_agents = num_agents

    def detect(self) -> List[str]:
        warnings: List[str] = []
        pickups = self._log.by_type("pickup")
        conflicts = self._log.by_type("conflict")

        if not pickups:
            return ["No pickup events recorded"]

        # Unbalanced workload
        if self._num_agents >= 2:
            agent_counts: Dict[str, int] = {}
            for evt in pickups:
                aid = evt.agent_id or "unknown"
                agent_counts[aid] = agent_counts.get(aid, 0) + 1
            max_count = max(agent_counts.values())
            total = sum(agent_counts.values())
            if max_count / total > 0.60:
                top = max(agent_counts, key=agent_counts.get)  # type: ignore[arg-type]
                warnings.append(
                    f"Unbalanced workload: {top} picked up {max_count}/{total} cards "
                    f"({max_count/total:.0%})"
                )

        # Excessive conflicts
        conflict_rate = len(conflicts) / len(pickups) if pickups else 0
        if conflict_rate > 0.30:
            warnings.append(
                f"Excessive conflicts: {len(conflicts)} conflicts in {len(pickups)} "
                f"pickups ({conflict_rate:.0%})"
            )

        # Stalled agent: 5+ consecutive rounds without a pickup
        round_events = self._log.by_type("round")
        if round_events:
            stall_counts: Dict[str, int] = {}
            max_stalls: Dict[str, int] = {}
            for evt in round_events:
                active = set(evt.data.get("active_agents", []))
                idle = set(evt.data.get("idle_agents", []))
                for aid in active:
                    stall_counts[aid] = 0
                for aid in idle:
                    stall_counts.setdefault(aid, 0)
                    stall_counts[aid] += 1
                    max_stalls[aid] = max(max_stalls.get(aid, 0), stall_counts[aid])
            for aid, streak in max_stalls.items():
                if streak >= 5:
                    warnings.append(f"Stalled agent: {aid} went {streak} consecutive rounds idle")

        return warnings

    def print_warnings(self) -> None:
        warnings = self.detect()
        if warnings:
            print("\n  Anomaly Warnings:")
            for w in warnings:
                print(f"    ! {w}")


# ---------------------------------------------------------------------------
# Terminal TUI Dashboard
# ---------------------------------------------------------------------------

class Dashboard:
    """Curses-based live terminal dashboard for the simulation.

    Usage::

        dash = Dashboard()
        dash.start()
        try:
            # ... simulation loop ...
            dash.update(cards, positions, picked_count, total, events, stats)
        finally:
            dash.stop()
    """

    def __init__(self) -> None:
        self._stdscr: Optional[Any] = None
        self._active = False

    def start(self) -> None:
        self._stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        curses.curs_set(0)
        if curses.has_colors():
            curses.start_color()
            curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)   # cards
            curses.init_pair(2, curses.COLOR_CYAN, curses.COLOR_BLACK)    # agents
            curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # header
            curses.init_pair(4, curses.COLOR_RED, curses.COLOR_BLACK)     # conflicts
        self._stdscr.nodelay(True)
        self._active = True

    def stop(self) -> None:
        if self._active and self._stdscr is not None:
            curses.nocbreak()
            curses.echo()
            curses.curs_set(1)
            curses.endwin()
            self._active = False

    def update(
        self,
        cards: List[dict],
        agent_positions: Dict[str, List[float]],
        picked_count: int,
        total_cards: int,
        recent_events: List[Event],
        stats: Dict[str, Any],
    ) -> None:
        if not self._active or self._stdscr is None:
            return
        try:
            self._draw(cards, agent_positions, picked_count, total_cards,
                       recent_events, stats)
        except curses.error:
            pass  # terminal too small, skip update

    def _draw(
        self,
        cards: List[dict],
        agent_positions: Dict[str, List[float]],
        picked_count: int,
        total_cards: int,
        recent_events: List[Event],
        stats: Dict[str, Any],
    ) -> None:
        scr = self._stdscr
        assert scr is not None
        scr.erase()
        max_y, max_x = scr.getmaxyx()

        # Header
        title = " 52 Card Pickup - Live Dashboard "
        if max_x > len(title):
            scr.addstr(0, 0, title.center(max_x), curses.color_pair(3) | curses.A_BOLD)

        # Build 10x10 grid (row 0 = y=9..10, row 9 = y=0..1)
        grid = [["." for _ in range(10)] for _ in range(10)]
        for c in cards:
            if not c["picked_up"]:
                gx = min(9, int(c["x"]))
                gy = min(9, int(c["y"]))
                grid[9 - gy][gx] = "C"

        # Place agents on grid
        agent_cells: Dict[tuple, str] = {}
        for aid, pos in agent_positions.items():
            gx = min(9, max(0, int(pos[0])))
            gy = min(9, max(0, int(pos[1])))
            idx = aid.split("_")[-1] if "_" in aid else "A"
            row = 9 - gy
            agent_cells[(row, gx)] = idx

        # Draw grid (rows 2-12)
        grid_start_row = 2
        grid_start_col = 1
        for r in range(10):
            if grid_start_row + r >= max_y - 1:
                break
            for col in range(10):
                ch_col = grid_start_col + col * 3
                if ch_col + 1 >= max_x:
                    break
                if (r, col) in agent_cells:
                    scr.addstr(grid_start_row + r, ch_col,
                               agent_cells[(r, col)],
                               curses.color_pair(2) | curses.A_BOLD)
                elif grid[r][col] == "C":
                    scr.addstr(grid_start_row + r, ch_col, "C",
                               curses.color_pair(1))
                else:
                    scr.addstr(grid_start_row + r, ch_col, ".")

        # Agent status panel (right side)
        panel_col = 35
        if panel_col < max_x - 10:
            scr.addstr(2, panel_col, "Agent Status", curses.A_BOLD)
            row = 3
            for aid, pos in sorted(agent_positions.items()):
                if row >= max_y - 1:
                    break
                cards_by = sum(
                    1 for c in cards
                    if c["picked_up"] and c.get("picked_up_by") == aid
                )
                line = f"{aid}: ({pos[0]:.1f},{pos[1]:.1f}) cards:{cards_by}"
                scr.addstr(row, panel_col, line[:max_x - panel_col - 1])
                row += 1

        # Progress bar
        progress_row = 13
        if progress_row < max_y - 1:
            pct = picked_count / total_cards if total_cards > 0 else 0
            bar_width = min(20, max_x - 15)
            filled = int(bar_width * pct)
            bar = "#" * filled + "-" * (bar_width - filled)
            scr.addstr(progress_row, 1,
                       f"Progress: [{bar}] {picked_count}/{total_cards}")

        # Stats row
        stats_row = 14
        if stats_row < max_y - 1:
            elapsed = stats.get("elapsed", 0.0)
            rnd = stats.get("round", 0)
            conflicts = stats.get("conflicts", 0)
            line = f"Elapsed: {elapsed:.3f}s  Round: {rnd}  Conflicts: {conflicts}"
            scr.addstr(stats_row, 1, line[:max_x - 2])

        # Recent events
        events_row = 16
        if events_row < max_y - 1:
            scr.addstr(events_row, 1, "Recent events:", curses.A_BOLD)
            for i, evt in enumerate(recent_events[-6:]):
                row = events_row + 1 + i
                if row >= max_y - 1:
                    break
                ts = f"[{evt.timestamp:.3f}]"
                if evt.event_type == "pickup":
                    card_key = evt.data.get("card", "?")
                    line = f"{ts} {evt.agent_id} picked up {card_key}"
                elif evt.event_type == "conflict":
                    winner = evt.data.get("winner", "?")
                    losers = evt.data.get("losers", [])
                    card = evt.data.get("card", "?")
                    line = f"{ts} CONFLICT over {card}: {winner} wins vs {', '.join(losers)}"
                    if len(line) < max_x - 2:
                        scr.addstr(row, 1, line[:max_x - 2], curses.color_pair(4))
                        continue
                elif evt.event_type == "governance":
                    line = f"{ts} GOVERNANCE: {evt.data.get('violation', '?')}"
                    if len(line) < max_x - 2:
                        scr.addstr(row, 1, line[:max_x - 2], curses.color_pair(4) | curses.A_BOLD)
                        continue
                else:
                    desc = evt.data.get("description", evt.event_type)
                    line = f"{ts} {evt.agent_id or ''} {desc}"
                scr.addstr(row, 1, line[:max_x - 2])

        scr.refresh()


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def replay_event_log(filepath: str, speed: float = 1.0) -> None:
    """Replay a saved event log through the TUI dashboard.

    Reconstructs the simulation state from events and displays them at
    *speed* multiplier (1.0 = real-time, 2.0 = double speed).
    """
    log = EventLog.load(filepath)
    events = log.events
    if not events:
        print("Empty event log.")
        return

    # Reconstruct initial card state from scatter event
    scatter_events = log.by_type("scatter")
    if not scatter_events:
        print("No scatter event found in log.")
        return

    cards = scatter_events[0].data.get("cards", [])
    if not cards:
        print("No card data in scatter event.")
        return

    # Reset all cards to unpicked
    for c in cards:
        c["picked_up"] = False
        c["picked_up_by"] = None

    # Find all agent IDs
    agent_ids = set()
    for evt in events:
        if evt.agent_id:
            agent_ids.add(evt.agent_id)

    positions: Dict[str, List[float]] = {aid: [0.0, 0.0] for aid in agent_ids}
    picked_count = 0
    conflicts = 0
    round_num = 0
    recent: List[Event] = []

    dash = Dashboard()
    dash.start()
    try:
        prev_ts = 0.0
        for evt in events:
            # Sleep proportional to time gap
            gap = (evt.timestamp - prev_ts) / speed if speed > 0 else 0
            if gap > 0:
                time.sleep(min(gap, 0.5))  # cap at 0.5s for usability
            prev_ts = evt.timestamp

            # Update state based on event
            if evt.event_type == "pickup":
                card_key = evt.data.get("card", "")
                for c in cards:
                    if f"{c['rank']} of {c['suit']}" == card_key and not c["picked_up"]:
                        c["picked_up"] = True
                        c["picked_up_by"] = evt.agent_id
                        break
                picked_count += 1
                if evt.agent_id and "position" in evt.data:
                    positions[evt.agent_id] = evt.data["position"]
            elif evt.event_type == "conflict":
                conflicts += 1
            elif evt.event_type == "round":
                round_num = evt.data.get("round", round_num + 1)
            elif evt.event_type == "move":
                if evt.agent_id and "position" in evt.data:
                    positions[evt.agent_id] = evt.data["position"]

            recent.append(evt)
            if len(recent) > 20:
                recent = recent[-20:]

            dash.update(
                cards, positions, picked_count, 52, recent,
                {"elapsed": evt.timestamp, "round": round_num, "conflicts": conflicts},
            )

        # Hold final state briefly
        time.sleep(2.0)
    finally:
        dash.stop()

    print(f"Replay complete. {len(events)} events replayed.")
