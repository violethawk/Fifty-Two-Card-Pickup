# Phase 5 Tutorial: Extensibility and Teaching

## What You'll Learn

- How to use the benchmark suite for reproducible comparisons
- How the plugin architecture lets you swap strategies and providers
- How to extend the system for your own experiments

## Prerequisites

- Completed Phases 1-4 tutorials

## The Benchmark Suite

The benchmark suite (`benchmarks.py`) provides six standardized scatter patterns:

| Pattern | Description | Expected Best Config |
|---------|-------------|---------------------|
| `uniform` | Golden-ratio spiral across grid | 4 agents |
| `clustered` | All cards in bottom-left corner | 1 agent |
| `two_clusters` | Cards near (1,1) and (9,9) | 2 agents |
| `four_clusters` | 13 cards per quadrant corner | 4 agents |
| `diagonal` | Cards along (0,0) to (10,10) diagonal | 4 agents |
| `edge` | Cards along grid perimeter | 4 agents |

Run it:

```bash
python card_pickup.py --benchmark
```

Each pattern is deterministic — same positions every time. This lets you compare strategies fairly: any difference in timing is due to the strategy, not the scatter.

### Adding Your Own Pattern

In `benchmarks.py`, add a function that returns `List[Card]`:

```python
def pattern_center() -> List[Card]:
    """All cards clustered at the center of the grid."""
    positions = []
    for i in range(52):
        angle = (i / 52.0) * 2 * math.pi
        r = (i % 5) * 0.3
        positions.append((5.0 + r * math.cos(angle), 5.0 + r * math.sin(angle)))
    return _make_deck(positions)
```

Register it:

```python
PATTERNS["center"] = pattern_center
```

The benchmark runner picks it up automatically.

## The Plugin Architecture

`plugins.py` defines two interfaces: **LLM providers** and **pickup strategies**.

### LLM Providers

```python
class LLMProvider:
    name: str
    def complete(self, system, user, max_tokens=300) -> Tuple[str, int, int]:
        """Return (response_text, input_tokens, output_tokens)."""
```

Built-in:
- `AnthropicProvider` — calls Claude via the Anthropic SDK
- `MockProvider` — returns greedy nearest-neighbor plans (no API key needed)

### Pickup Strategies

```python
class PickupStrategy:
    name: str
    def pick_next(self, agent_id, position, cards, other_agents) -> Optional[int]:
        """Return the index of the next card to pick up, or None to wait."""
```

Built-in:
- `GreedyNearestStrategy` — always pick the closest unpicked card
- `LLMStrategy` — use an LLM provider to plan batches of moves

### Adding Your Own Provider

Want to use OpenAI instead of Anthropic?

```python
class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, model="gpt-4o-mini"):
        from openai import OpenAI
        self._client = OpenAI()
        self._model = model

    def complete(self, system, user, max_tokens=300):
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = response.choices[0].message.content.strip()
        usage = response.usage
        return text, usage.prompt_tokens, usage.completion_tokens
```

Register it:

```python
from plugins import PROVIDERS
PROVIDERS["openai"] = OpenAIProvider
```

### Adding Your Own Strategy

Want a "furthest first" strategy that picks the card furthest from all other agents?

```python
class FurthestFromAgentsStrategy(PickupStrategy):
    name = "furthest"

    def pick_next(self, agent_id, position, cards, other_agents):
        best_idx = None
        best_min_dist = -1
        for idx, card in enumerate(cards):
            if card["picked_up"]:
                continue
            # Minimum distance from any other agent
            min_d = float("inf")
            for aid, pos in other_agents.items():
                if aid == agent_id:
                    continue
                d = math.hypot(card["x"] - pos[0], card["y"] - pos[1])
                min_d = min(min_d, d)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_idx = idx
        return best_idx
```

## CLI Flags

```bash
python card_pickup.py --strategy greedy    # Phase 1 style
python card_pickup.py --strategy llm       # Phase 3 style
python card_pickup.py --provider anthropic  # Real API calls
python card_pickup.py --provider mock       # No API key needed
python card_pickup.py --benchmark           # Run all patterns
```

## Project Structure

```
card_pickup.py      # Core simulation: agents, state, graph
observability.py    # Event logging, governance, metrics, dashboard
benchmarks.py       # Scatter patterns and benchmark runner
plugins.py          # Strategy and provider interfaces
prompts/            # Tier 2 implementation prompts for each phase
docs/               # Tutorials and guides
```

## Exercises

1. **Write a new strategy**: Implement a "random walk" strategy that picks a random unpicked card. How does it compare to greedy on the benchmarks?
2. **Write a new provider**: Wrap a local LLM (Ollama, llama.cpp) as a provider. How does response quality compare to Claude?
3. **Cross-compare**: Run all 6 benchmark patterns with greedy, LLM, and your custom strategy. Build a comparison matrix.
4. **Teach someone**: Walk a friend through the Phase 1 tutorial. Time how long it takes. Can they complete it in under 2 hours?
