# Future Directions

Ideas for expanding 52 Card Pickup beyond the current Phase 1-5 implementation. These are roughly ordered from most natural next steps to more ambitious explorations.

---

## Smarter Pickup Algorithms

The current pickup strategy is greedy nearest-neighbor — one of the weakest heuristics for what is essentially a Traveling Salesman Problem variant. Room to explore:

- **2-opt / k-opt local search** — iteratively improve a tour by swapping edges. Simple to implement, meaningful improvement over greedy.
- **Cluster-then-sweep** — identify card clusters, clear each cluster before crossing the grid. Mirrors how a human would actually pick up cards.
- **Simulated annealing / genetic algorithms** — classical metaheuristics applied to tour optimization.
- **Christofides algorithm** — guaranteed 1.5x optimal for metric TSP. More complex but a well-known benchmark.

The greedy baseline is deliberately beatable, which makes this a good axis for measuring how much value better algorithms (or LLM planning) actually add.

---

## Smarter Region Assignment

Regions are currently hardcoded geometry — left/right split at x=5 for 2 agents, quadrants for 4. This ignores actual card distribution:

- **Voronoi partitioning** — assign cards to the nearest agent's deployment position, creating organic regions.
- **Load-balanced partitioning** — ensure each agent gets roughly equal work (card count or total travel distance), not just equal area.
- **Auction-based allocation** — agents bid on cards or clusters based on proximity. Introduces a market mechanism.
- **Dynamic re-partitioning** — adjust regions mid-run as agents finish at different rates.

---

## Active Supervisor

The supervisor currently makes one decision at the start (agent count + deployment positions) and then disappears. A more realistic supervisor pattern would stay in the loop:

- **Runtime monitoring** — observe agent progress and detect imbalances.
- **Re-tasking** — if one agent finishes early, reassign idle capacity to help others.
- **Work-stealing** — agents that clear their region pull unclaimed cards from overloaded neighbors.
- **Escalation** — supervisor intervenes when conflict rates spike or agents appear stuck.
- **Conflict arbitration** — currently conflicts are resolved by a headless distance function. The supervisor could arbitrate disputes with broader strategic context.

This would demonstrate the difference between a dispatcher (current) and a true supervisor pattern.

---

## Flexible Agent Count

Currently limited to {1, 2, 4} agents with fixed region geometry. Generalizing:

- **Arbitrary agent counts** — 3, 5, 6, etc. Requires moving away from hardcoded quadrant splits.
- **Dynamic spawning** — supervisor starts with 2 agents, observes the workload, spawns a 3rd mid-run if needed.
- **Agent cost model** — each agent has a spawn cost (time overhead). Supervisor must weigh parallelism gains against coordination overhead.

---

## Deployment Position Optimization

The current deployment strategy (region centroid) is a reasonable heuristic but not optimal:

- **Density-weighted positioning** — deploy near the densest cluster within a region, not the centroid.
- **Tour-aware deployment** — the best starting position depends on pickup *order*, not just average card position. Optimize for "minimize distance to first card + expected total tour."
- **Competitive deployment** — in adversarial scenarios, agents might want to deploy to "claim" high-value areas before others.

---

## Richer Coordination Protocols

The current Phase 3 protocol is: plan, broadcast intention, resolve conflicts by distance, execute. Alternatives:

- **Negotiation** — agents propose trades ("I'll skip this card if you skip that one").
- **Contract Net Protocol** — task announcements, bids, and awards. A classic multi-agent pattern.
- **Shared planning** — agents jointly construct a global plan rather than independently planning and deconflicting.
- **Stigmergy** — indirect coordination through environment modification (marking areas as "claimed" on the grid).

---

## Benchmark Formalization

The project already has benchmark patterns (uniform, clustered, diagonal, etc.) but could become a more rigorous benchmark:

- **Standardized scoring** — total distance traveled, wall-clock time, number of LLM calls, API cost. A composite score.
- **Leaderboard patterns** — fixed seed + pattern combinations that define the benchmark suite.
- **Strategy plugins** — a clean interface for submitting new pickup/coordination strategies and comparing them against baselines.
- **Scaling dimensions** — vary grid size (10x10 to 100x100), card count (52 to 520), agent count. Test how strategies degrade.

---

## Timer Agent Consolidation

The current timer start/stop are separate LangGraph nodes that don't make decisions — they're bookkeeping, not agents. Options:

- **Fold into pickup/delivery nodes** — inline the `time.perf_counter()` calls where they're needed.
- **Single timer node** — one node that contextually records start vs. stop based on state.
- **Remove entirely** — let the observability layer (Phase 4 event log) handle timing, since it already timestamps everything.

This is a minor cleanup but would make the agent count more honest.

---

## Teaching Extensions

- **"Build a Better Supervisor" challenge** — given the same scatter patterns, can you write a supervisor prompt that beats the default?
- **"Beat Greedy" challenge** — implement a pickup strategy plugin that outperforms nearest-neighbor.
- **Visualization of strategy differences** — side-by-side animation of greedy vs. optimized tours on the same scatter.
- **Cost-performance frontier** — plot LLM cost vs. pickup time across strategies. When does intelligence pay for itself?
