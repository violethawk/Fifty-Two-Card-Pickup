"""Unit tests for benchmark patterns."""

from benchmarks import PATTERNS, _make_deck


class TestPatterns:
    def test_all_patterns_produce_52_cards(self):
        for name, fn in PATTERNS.items():
            cards = fn()
            assert len(cards) == 52, f"{name} produced {len(cards)} cards"

    def test_all_patterns_unique_cards(self):
        for name, fn in PATTERNS.items():
            cards = fn()
            keys = [(c["suit"], c["rank"]) for c in cards]
            assert len(keys) == len(set(keys)), f"{name} has duplicates"

    def test_all_patterns_positions_in_range(self):
        for name, fn in PATTERNS.items():
            cards = fn()
            for c in cards:
                assert 0.0 <= c["x"] <= 10.0, f"{name}: x={c['x']} out of range"
                assert 0.0 <= c["y"] <= 10.0, f"{name}: y={c['y']} out of range"

    def test_all_patterns_unpicked(self):
        for name, fn in PATTERNS.items():
            cards = fn()
            assert all(not c["picked_up"] for c in cards), f"{name} has picked cards"

    def test_patterns_are_deterministic(self):
        for name, fn in PATTERNS.items():
            cards1 = fn()
            cards2 = fn()
            for c1, c2 in zip(cards1, cards2):
                assert c1["x"] == c2["x"] and c1["y"] == c2["y"], \
                    f"{name} is not deterministic"
