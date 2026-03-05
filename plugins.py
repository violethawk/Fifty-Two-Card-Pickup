"""
plugins.py
==========

Plugin interfaces and built-in implementations for pickup strategies
and LLM providers.  Allows swapping components without modifying core code.

Usage::

    python card_pickup.py --strategy greedy --provider mock
"""

from __future__ import annotations

import json
import math
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# LLM Provider interface
# ---------------------------------------------------------------------------

class LLMProvider:
    """Base class for LLM providers.

    Subclasses must implement :meth:`complete` and set :attr:`name`.

    Example::

        class MyProvider(LLMProvider):
            name = "my_provider"
            def complete(self, system, user, max_tokens=300):
                # call your API
                return response_text, input_tokens, output_tokens
    """
    name: str = "base"

    def complete(
        self, system: str, user: str, max_tokens: int = 300
    ) -> Tuple[str, int, int]:
        """Send a prompt and return (response_text, input_tokens, output_tokens)."""
        raise NotImplementedError


class AnthropicProvider(LLMProvider):
    """Wraps the Anthropic Python SDK."""
    name = "anthropic"

    def __init__(self, model: str = "claude-haiku-4-5-20251001") -> None:
        import anthropic
        self._client = anthropic.Anthropic()
        self._model = model

    def complete(
        self, system: str, user: str, max_tokens: int = 300
    ) -> Tuple[str, int, int]:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = response.content[0].text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3].strip()
        return text, response.usage.input_tokens, response.usage.output_tokens


class MockProvider(LLMProvider):
    """Returns deterministic responses for testing without an API key.

    The mock provider parses the user message to find card positions and
    returns a greedy nearest-neighbor plan as if the LLM had chosen it.
    """
    name = "mock"

    def complete(
        self, system: str, user: str, max_tokens: int = 300
    ) -> Tuple[str, int, int]:
        # Parse position from system prompt
        agent_x, agent_y = 0.0, 0.0
        if "position" in system.lower():
            import re
            match = re.search(r"position \(([0-9.]+), ([0-9.]+)\)", system)
            if match:
                agent_x = float(match.group(1))
                agent_y = float(match.group(2))

        # Parse remaining cards from user message
        cards = []
        for line in user.split("\n"):
            line = line.strip()
            if " at (" in line:
                parts = line.split(" at (")
                if len(parts) == 2:
                    name = parts[0].strip()
                    coord = parts[1].rstrip(")")
                    try:
                        cx, cy = coord.split(",")
                        cards.append((name, float(cx), float(cy.strip())))
                    except ValueError:
                        continue

        # Greedy nearest-neighbor ordering
        targets = []
        cx, cy = agent_x, agent_y
        remaining = list(cards)
        for _ in range(min(10, len(remaining))):
            best = None
            best_dist = float("inf")
            for card in remaining:
                d = math.hypot(card[1] - cx, card[2] - cy)
                if d < best_dist:
                    best_dist = d
                    best = card
            if best is None:
                break
            targets.append(best[0])
            cx, cy = best[1], best[2]
            remaining.remove(best)

        response = json.dumps({
            "targets": targets,
            "strategy": "Mock: greedy nearest-neighbor",
        })
        # Approximate token counts
        input_tokens = len(system.split()) + len(user.split())
        output_tokens = len(response.split())
        return response, input_tokens, output_tokens


# ---------------------------------------------------------------------------
# Pickup Strategy interface
# ---------------------------------------------------------------------------

class PickupStrategy:
    """Base class for pickup strategies.

    Subclasses must implement :meth:`pick_next` and set :attr:`name`.

    Example::

        class MyStrategy(PickupStrategy):
            name = "my_strategy"
            def pick_next(self, agent_id, position, cards, other_agents):
                # your logic here
                return card_index  # or None to wait
    """
    name: str = "base"

    def pick_next(
        self,
        agent_id: str,
        position: Tuple[float, float],
        cards: List[dict],
        other_agents: Dict[str, Tuple[float, float]],
    ) -> Optional[int]:
        """Return the index of the next card to pick up, or None to wait."""
        raise NotImplementedError


class GreedyNearestStrategy(PickupStrategy):
    """Phase 1 greedy nearest-neighbor: always pick the closest unpicked card."""
    name = "greedy"

    def pick_next(
        self,
        agent_id: str,
        position: Tuple[float, float],
        cards: List[dict],
        other_agents: Dict[str, Tuple[float, float]],
    ) -> Optional[int]:
        best_idx = None
        best_dist = float("inf")
        for idx, card in enumerate(cards):
            if card["picked_up"]:
                continue
            d = math.hypot(card["x"] - position[0], card["y"] - position[1])
            if d < best_dist:
                best_dist = d
                best_idx = idx
        return best_idx


class LLMStrategy(PickupStrategy):
    """Phase 3 LLM-powered strategy: uses an LLM provider to plan moves."""
    name = "llm"

    SYSTEM_TEMPLATE = """\
You are a pickup agent in a 52 Card Pickup simulation on a 10x10 grid.
You are at position ({x:.1f}, {y:.1f}).

Plan your next moves to efficiently collect cards. Each move costs time \
proportional to the distance traveled (0.005s per unit).

Prioritize:
- Cards close to your current position (minimize travel distance)
- Clusters of nearby cards (plan a path through them)
- Cards that other agents are NOT heading toward (check their intentions)

Respond with ONLY a JSON object (no markdown, no extra text):
{{"targets": ["<rank> of <suit>", ...], "strategy": "<brief explanation>"}}

List 5-10 cards in the order you want to collect them.
"""

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider
        self._last_plan: List[str] = []
        self._last_strategy: str = ""
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0

    @property
    def token_usage(self) -> Tuple[int, int]:
        return self._total_input_tokens, self._total_output_tokens

    def plan(
        self,
        agent_id: str,
        position: Tuple[float, float],
        cards: List[dict],
        other_agents: Dict[str, Tuple[float, float]],
        intentions: Dict[str, str],
    ) -> Tuple[List[str], str]:
        """Call the LLM to get a batch plan. Returns (targets, strategy)."""
        remaining = [c for c in cards if not c["picked_up"]]
        remaining_desc = [
            f"  {c['rank']} of {c['suit']} at ({c['x']:.1f}, {c['y']:.1f})"
            for c in remaining
        ]

        other_info = []
        for aid, pos in other_agents.items():
            if aid == agent_id:
                continue
            intention = intentions.get(aid, "none")
            other_info.append(
                f"  {aid} at ({pos[0]:.1f}, {pos[1]:.1f}), heading toward: {intention}"
            )

        user_msg = (
            f"Remaining cards ({len(remaining)}):\n"
            + "\n".join(remaining_desc)
            + "\n\nOther agents:\n"
            + ("\n".join(other_info) if other_info else "  (none)")
        )
        system = self.SYSTEM_TEMPLATE.format(x=position[0], y=position[1])

        try:
            text, in_tok, out_tok = self._provider.complete(system, user_msg)
            self._total_input_tokens += in_tok
            self._total_output_tokens += out_tok
            decision = json.loads(text)
            targets = decision.get("targets", [])
            strategy = decision.get("strategy", "No strategy provided.")
            if not targets:
                raise ValueError("Empty targets")
            self._last_plan = targets
            self._last_strategy = strategy
            return targets, strategy
        except Exception:
            # Greedy fallback
            greedy = GreedyNearestStrategy()
            targets = []
            pos = position
            temp_cards = [dict(c) for c in cards]
            for _ in range(min(10, len(remaining))):
                idx = greedy.pick_next("", pos, temp_cards, {})
                if idx is None:
                    break
                c = temp_cards[idx]
                targets.append(f"{c['rank']} of {c['suit']}")
                pos = (c["x"], c["y"])
                temp_cards[idx] = dict(c, picked_up=True)
            self._last_plan = targets
            self._last_strategy = "Fallback: greedy nearest-neighbor"
            return targets, self._last_strategy

    def pick_next(
        self,
        agent_id: str,
        position: Tuple[float, float],
        cards: List[dict],
        other_agents: Dict[str, Tuple[float, float]],
    ) -> Optional[int]:
        """Pick the first valid card from the last plan, or greedy fallback."""
        for target in self._last_plan:
            for idx, card in enumerate(cards):
                if not card["picked_up"] and f"{card['rank']} of {card['suit']}" == target:
                    return idx
        # Fallback
        greedy = GreedyNearestStrategy()
        return greedy.pick_next(agent_id, position, cards, other_agents)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PROVIDERS: Dict[str, type] = {
    "anthropic": AnthropicProvider,
    "mock": MockProvider,
}

STRATEGIES: Dict[str, type] = {
    "greedy": GreedyNearestStrategy,
    "llm": LLMStrategy,
}


def get_provider(name: str, **kwargs) -> LLMProvider:
    """Instantiate an LLM provider by name."""
    cls = PROVIDERS.get(name)
    if cls is None:
        raise ValueError(f"Unknown provider: {name}. Available: {list(PROVIDERS.keys())}")
    return cls(**kwargs)


def get_strategy(name: str, **kwargs) -> PickupStrategy:
    """Instantiate a pickup strategy by name."""
    cls = STRATEGIES.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGIES.keys())}")
    return cls(**kwargs)
