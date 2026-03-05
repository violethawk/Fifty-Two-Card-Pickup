"""Unit tests for core simulation logic — no API key needed."""

import random

from card_pickup import (
    AppState,
    Card,
    SUITS,
    RANKS,
    VERIFIER_X,
    VERIFIER_Y,
    _card_key,
    _find_card_by_key,
    _greedy_nearest_card,
    _resolve_conflicts,
    _analyze_scatter,
    _extract_timing,
    scatter_node,
    delivery_node,
    verify_node,
    _make_initial_state,
    build_graph,
    _extract_elapsed,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_card(suit="hearts", rank="A", x=0.0, y=0.0, picked_up=False, picked_up_by=None):
    return {
        "suit": suit, "rank": rank, "x": x, "y": y,
        "picked_up": picked_up, "picked_up_by": picked_up_by,
    }


def _make_deck_at_origin():
    """Full 52-card deck all at (0, 0), unpicked."""
    cards = []
    for suit in SUITS:
        for rank in RANKS:
            cards.append(_make_card(suit, rank, 0.0, 0.0))
    return cards


# ---------------------------------------------------------------------------
# scatter_node
# ---------------------------------------------------------------------------

class TestScatterNode:
    def test_produces_52_cards(self):
        state = _make_initial_state(1)
        state = scatter_node(state)
        assert len(state["cards"]) == 52

    def test_all_unpicked(self):
        state = _make_initial_state(1)
        state = scatter_node(state)
        assert all(not c["picked_up"] for c in state["cards"])

    def test_unique_suit_rank_pairs(self):
        state = _make_initial_state(1)
        state = scatter_node(state)
        keys = [(c["suit"], c["rank"]) for c in state["cards"]]
        assert len(keys) == len(set(keys))

    def test_positions_in_range(self):
        random.seed(99)
        state = _make_initial_state(1)
        state = scatter_node(state)
        for c in state["cards"]:
            assert 0.0 <= c["x"] <= 10.0
            assert 0.0 <= c["y"] <= 10.0


# ---------------------------------------------------------------------------
# _card_key / _find_card_by_key
# ---------------------------------------------------------------------------

class TestCardKey:
    def test_card_key_format(self):
        c = _make_card("spades", "K")
        assert _card_key(c) == "K of spades"

    def test_find_card_by_key_found(self):
        cards = [_make_card("hearts", "A"), _make_card("spades", "K")]
        assert _find_card_by_key(cards, "K of spades") == 1

    def test_find_card_by_key_not_found(self):
        cards = [_make_card("hearts", "A")]
        assert _find_card_by_key(cards, "K of spades") is None


# ---------------------------------------------------------------------------
# _greedy_nearest_card
# ---------------------------------------------------------------------------

class TestGreedyNearest:
    def test_picks_closest(self):
        cards = [
            _make_card(x=5.0, y=5.0),
            _make_card(x=1.0, y=1.0, suit="spades"),
        ]
        idx = _greedy_nearest_card(cards, 0.0, 0.0)
        assert idx == 1  # closer card is at index 1

    def test_skips_picked(self):
        cards = [
            _make_card(x=1.0, y=1.0, picked_up=True),
            _make_card(x=5.0, y=5.0, suit="spades"),
        ]
        idx = _greedy_nearest_card(cards, 0.0, 0.0)
        assert idx == 1  # index 0 is picked, so picks index 1

    def test_returns_none_when_all_picked(self):
        cards = [_make_card(x=1.0, y=1.0, picked_up=True)]
        assert _greedy_nearest_card(cards, 0.0, 0.0) is None


# ---------------------------------------------------------------------------
# _resolve_conflicts
# ---------------------------------------------------------------------------

class TestResolveConflicts:
    def test_no_conflict(self):
        cards = [
            _make_card("hearts", "A", x=1.0, y=1.0),
            _make_card("spades", "K", x=9.0, y=9.0),
        ]
        targets = {"agent_0": "A of hearts", "agent_1": "K of spades"}
        positions = {"agent_0": [0.0, 0.0], "agent_1": [10.0, 10.0]}
        resolved = _resolve_conflicts(targets, positions, cards)
        assert resolved["agent_0"] == 0
        assert resolved["agent_1"] == 1

    def test_closest_wins(self):
        cards = [_make_card("hearts", "A", x=2.0, y=0.0)]
        targets = {"agent_0": "A of hearts", "agent_1": "A of hearts"}
        positions = {"agent_0": [1.0, 0.0], "agent_1": [5.0, 0.0]}
        resolved = _resolve_conflicts(targets, positions, cards)
        assert resolved["agent_0"] == 0
        assert resolved["agent_1"] is None

    def test_tiebreak_by_id(self):
        cards = [_make_card("hearts", "A", x=5.0, y=5.0)]
        targets = {"agent_0": "A of hearts", "agent_1": "A of hearts"}
        # Both equidistant
        positions = {"agent_0": [5.0, 0.0], "agent_1": [5.0, 10.0]}
        resolved = _resolve_conflicts(targets, positions, cards)
        assert resolved["agent_0"] == 0
        assert resolved["agent_1"] is None

    def test_missing_card_returns_none(self):
        cards = [_make_card("hearts", "A")]
        targets = {"agent_0": "K of spades"}
        positions = {"agent_0": [0.0, 0.0]}
        resolved = _resolve_conflicts(targets, positions, cards)
        assert resolved["agent_0"] is None


# ---------------------------------------------------------------------------
# _analyze_scatter
# ---------------------------------------------------------------------------

class TestAnalyzeScatter:
    def test_returns_expected_keys(self):
        random.seed(42)
        state = _make_initial_state(1)
        state = scatter_node(state)
        metrics = _analyze_scatter(state["cards"])
        assert "quadrant_counts" in metrics
        assert "left_right_split" in metrics
        assert "spatial_spread" in metrics
        assert "avg_nearest_neighbor_distance" in metrics
        assert "quadrant_balance_ratio" in metrics

    def test_quadrants_sum_to_52(self):
        random.seed(42)
        state = _make_initial_state(1)
        state = scatter_node(state)
        metrics = _analyze_scatter(state["cards"])
        assert sum(metrics["quadrant_counts"].values()) == 52


# ---------------------------------------------------------------------------
# verify_node
# ---------------------------------------------------------------------------

class TestVerifyNode:
    def test_pass_all_picked(self):
        cards = []
        for suit in SUITS:
            for rank in RANKS:
                cards.append(_make_card(suit, rank, picked_up=True, picked_up_by="agent_0"))
        state = _make_initial_state(1)
        state["cards"] = cards
        state = verify_node(state)
        assert state["result"].startswith("PASS")

    def test_fail_unpicked_card(self):
        cards = []
        for suit in SUITS:
            for rank in RANKS:
                cards.append(_make_card(suit, rank, picked_up=True, picked_up_by="agent_0"))
        cards[0]["picked_up"] = False
        state = _make_initial_state(1)
        state["cards"] = cards
        state = verify_node(state)
        assert state["result"].startswith("FAIL")

    def test_fail_wrong_count(self):
        state = _make_initial_state(1)
        state["cards"] = [_make_card(picked_up=True, picked_up_by="agent_0")]
        state = verify_node(state)
        assert state["result"].startswith("FAIL")

    def test_fail_duplicate(self):
        cards = []
        for suit in SUITS:
            for rank in RANKS:
                cards.append(_make_card(suit, rank, picked_up=True, picked_up_by="agent_0"))
        # Make a duplicate
        cards[1]["suit"] = cards[0]["suit"]
        cards[1]["rank"] = cards[0]["rank"]
        state = _make_initial_state(1)
        state["cards"] = cards
        state = verify_node(state)
        assert state["result"].startswith("FAIL")


# ---------------------------------------------------------------------------
# End-to-end Phase 1 (deterministic)
# ---------------------------------------------------------------------------

class TestDelivery:
    def test_delivery_all_cards_delivered(self):
        random.seed(42)
        graph = build_graph(with_supervisor=False, llm_pickup=False)
        state = _make_initial_state(4)
        final = graph.invoke(state)
        assert final["cards_delivered"] == 52
        assert len(final["deliveries"]) == 4

    def test_delivery_agents_at_verifier(self):
        random.seed(42)
        graph = build_graph(with_supervisor=False, llm_pickup=False)
        state = _make_initial_state(2)
        final = graph.invoke(state)
        for aid, pos in final["agent_positions"].items():
            assert abs(pos[0] - VERIFIER_X) < 0.01
            assert abs(pos[1] - VERIFIER_Y) < 0.01

    def test_timing_breakdown(self):
        random.seed(42)
        graph = build_graph(with_supervisor=False, llm_pickup=False)
        state = _make_initial_state(2)
        final = graph.invoke(state)
        timing = _extract_timing(final)
        assert timing["pickup_duration"] > 0
        assert timing["delivery_duration"] > 0
        assert abs(timing["total_duration"] - (timing["pickup_duration"] + timing["delivery_duration"])) < 0.01


class TestEndToEnd:
    def test_phase1_single_agent(self):
        random.seed(42)
        graph = build_graph(with_supervisor=False, llm_pickup=False)
        state = _make_initial_state(1)
        final = graph.invoke(state)
        assert final["result"].startswith("PASS")
        assert len(final["cards"]) == 52
        assert all(c["picked_up"] for c in final["cards"])
        assert final["cards_delivered"] == 52

    def test_phase1_four_agents(self):
        random.seed(42)
        graph = build_graph(with_supervisor=False, llm_pickup=False)
        state = _make_initial_state(4)
        final = graph.invoke(state)
        assert final["result"].startswith("PASS")
        assert final["cards_delivered"] == 52
