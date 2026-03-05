"""Unit tests for observability: event log, governance, metrics."""

import pytest

from card_pickup import SUITS, RANKS
from observability import Event, EventLog, GovernanceChecker, GovernanceViolation


def _make_card(suit="hearts", rank="A", x=0.0, y=0.0, picked_up=False, picked_up_by=None):
    return {
        "suit": suit, "rank": rank, "x": x, "y": y,
        "picked_up": picked_up, "picked_up_by": picked_up_by,
    }


def _full_deck(picked_up=False, picked_up_by=None):
    cards = []
    for suit in SUITS:
        for rank in RANKS:
            cards.append(_make_card(suit, rank, picked_up=picked_up, picked_up_by=picked_up_by))
    return cards


# ---------------------------------------------------------------------------
# EventLog
# ---------------------------------------------------------------------------

class TestEventLog:
    def test_emit_and_query(self):
        log = EventLog()
        log.emit("scatter", data={"count": 52})
        log.emit("pickup", agent_id="agent_0", data={"card": "A of hearts"})
        log.emit("pickup", agent_id="agent_1", data={"card": "K of spades"})
        assert len(log.events) == 3
        assert len(log.by_type("pickup")) == 2
        assert len(log.by_agent("agent_0")) == 1

    def test_serialize_roundtrip(self):
        log = EventLog()
        log.emit("scatter")
        log.emit("pickup", agent_id="agent_0")
        serialized = log.serialize()
        restored = EventLog.deserialize(serialized)
        assert len(restored.events) == 2
        assert restored.events[0].event_type == "scatter"
        assert restored.events[1].agent_id == "agent_0"

    def test_save_load(self, tmp_path):
        log = EventLog()
        log.emit("test", data={"value": 42})
        filepath = str(tmp_path / "test_log.json")
        log.save(filepath)
        loaded = EventLog.load(filepath)
        assert len(loaded.events) == 1
        assert loaded.events[0].data["value"] == 42


# ---------------------------------------------------------------------------
# GovernanceChecker
# ---------------------------------------------------------------------------

class TestGovernanceChecker:
    def test_passes_valid_state(self):
        log = EventLog()
        checker = GovernanceChecker(log, ["agent_0"])
        cards = _full_deck(picked_up=False)
        checker.check(cards)  # should not raise

    def test_fails_wrong_card_count(self):
        log = EventLog()
        checker = GovernanceChecker(log, ["agent_0"])
        cards = [_make_card()]  # only 1 card
        with pytest.raises(GovernanceViolation, match="Card count"):
            checker.check(cards)

    def test_fails_phantom_agent(self):
        log = EventLog()
        checker = GovernanceChecker(log, ["agent_0"])
        cards = _full_deck(picked_up=False)
        cards[0]["picked_up"] = True
        cards[0]["picked_up_by"] = "ghost_agent"
        with pytest.raises(GovernanceViolation, match="Phantom agent"):
            checker.check(cards)

    def test_fails_backwards_progress(self):
        log = EventLog()
        checker = GovernanceChecker(log, ["agent_0"])
        cards = _full_deck(picked_up=False)
        # First check with 5 picked up
        for i in range(5):
            cards[i]["picked_up"] = True
            cards[i]["picked_up_by"] = "agent_0"
        checker.check(cards)
        # Now regress to 3 picked up
        cards[3]["picked_up"] = False
        cards[3]["picked_up_by"] = None
        cards[4]["picked_up"] = False
        cards[4]["picked_up_by"] = None
        with pytest.raises(GovernanceViolation, match="backwards"):
            checker.check(cards)

    def test_monotonic_progress_ok(self):
        log = EventLog()
        checker = GovernanceChecker(log, ["agent_0"])
        cards = _full_deck(picked_up=False)
        checker.check(cards)  # 0 picked
        cards[0]["picked_up"] = True
        cards[0]["picked_up_by"] = "agent_0"
        checker.check(cards)  # 1 picked
        cards[1]["picked_up"] = True
        cards[1]["picked_up_by"] = "agent_0"
        checker.check(cards)  # 2 picked — no error
