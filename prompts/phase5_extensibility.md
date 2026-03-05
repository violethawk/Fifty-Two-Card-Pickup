# Phase 5 Implementation Prompt — Extensibility and Teaching

## Tier 2 (Semi-Formal)

---

## Context

Phases 1-4 are complete. The project has deterministic agents, an LLM supervisor, LLM pickup agents with conflict resolution, and full observability infrastructure. All code is in `card_pickup.py` and `observability.py`.

Phase 5 packages the project for teaching and extensibility.

## Deliverables (in implementation order)

### 1. Benchmark Suite (`benchmarks.py`)

Standardized scatter patterns for reproducible comparisons across strategies and configurations.

**Patterns:**
- `uniform` — cards randomly scattered with a fixed seed (baseline)
- `clustered` — cards concentrated in one quadrant (tests whether 1 agent is optimal)
- `two_clusters` — cards split between two corners (tests 2-agent split)
- `four_clusters` — cards in four distinct clusters, one per quadrant (tests 4-agent split)
- `diagonal` — cards along the diagonal from (0,0) to (10,10)
- `edge` — cards along the perimeter of the grid

Each pattern is a function that returns a list of 52 Card dicts with predetermined positions (deterministic, no randomness needed for the pattern itself).

**Benchmark runner:**
- Runs all patterns against all configurations (1/2/4 agents, deterministic)
- Optionally runs supervisor + LLM agents against each pattern
- Reports timing, verifier results, and which config was fastest per pattern
- Output as a formatted table

**CLI:** `python card_pickup.py --benchmark`

### 2. Plugin Architecture

Allow swapping LLM providers, pickup strategies, and coordination protocols without modifying core code.

**Strategy plugin interface:**
```python
class PickupStrategy:
    """Base class for pickup strategies."""
    name: str
    def pick_next(self, agent_id, position, cards, other_agents) -> Optional[int]:
        """Return the index of the next card to pick up, or None to wait."""
        ...
```

**Built-in strategies:**
- `GreedyNearestStrategy` — the Phase 1 greedy nearest-neighbor
- `LLMStrategy` — the Phase 3 Haiku-powered planner

**LLM provider plugin interface:**
```python
class LLMProvider:
    """Base class for LLM providers."""
    name: str
    def complete(self, system: str, user: str, max_tokens: int) -> Tuple[str, int, int]:
        """Return (response_text, input_tokens, output_tokens)."""
        ...
```

**Built-in providers:**
- `AnthropicProvider` — wraps the current Anthropic SDK calls
- `MockProvider` — returns deterministic responses for testing (no API key needed)

**Registration and selection:**
- Strategies and providers registered in a simple dict registry
- Selected via CLI flags: `--strategy greedy|llm`, `--provider anthropic|mock`
- Default: greedy strategy, anthropic provider (when API key present)

**File:** `plugins.py` for interfaces and built-in implementations.

### 3. "Add Your Own Agent" Guide (`docs/add_your_own_agent.md`)

Step-by-step guide for adding a new agent to the graph:

1. Define your agent's role and what state it reads/writes
2. Write the node function (with example: a "Sorter" agent that reorders cards by suit after pickup)
3. Register the node in `build_graph`
4. Add edges to position it in the pipeline
5. Update `AppState` if your agent needs new fields
6. Test with the verifier
7. Add event logging for observability

Include complete working code for the example Sorter agent.

### 4. Tutorial Series (`docs/tutorial_phase_N.md`)

One document per phase. Each tutorial:
- States what you'll learn
- Lists prerequisites (previous phase)
- Walks through the code with explanations
- Highlights the key multi-agent concept introduced
- Ends with exercises

**Phase 1 tutorial** is the most important — a developer with no multi-agent experience should be able to complete it in under 2 hours.

### 5. Blog Post (`docs/blog_post.md`)

"52 Card Pickup: The Multi-Agent Hello World"

Narrative structure:
- The problem: multi-agent systems are hard to learn because examples are too complex
- The metaphor: 52 Card Pickup maps perfectly to agent roles, coordination, and constraints
- The journey: Phase 1 through Phase 5, what each teaches
- Key insights: hybrid architecture, intelligence vs overhead tradeoff, observability matters
- Call to action: try it yourself, extend it, teach with it

## Implementation Constraints

- Benchmark suite must work without an API key (deterministic patterns + deterministic agents)
- Plugin architecture must not break existing code — all current behavior accessible via defaults
- Tutorials reference actual code in the repository (file paths, line numbers)
- All docs use plain Markdown
- No new pip dependencies

## Success Criteria

- `python card_pickup.py --benchmark` runs all patterns and prints comparison table
- `--strategy` and `--provider` flags work for swapping implementations
- "Add Your Own Agent" guide produces a working agent when followed step by step
- Phase 1 tutorial is completable by a newcomer in under 2 hours
- Blog post tells a coherent story from start to finish
