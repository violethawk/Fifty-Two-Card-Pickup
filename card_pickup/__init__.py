"""
card_pickup — Multi-agent 52 Card Pickup simulation.

All public symbols are re-exported from the internal _core module
so that existing imports like ``from card_pickup import AppState``
continue to work unchanged.
"""

from card_pickup._core import (  # noqa: F401
    # Types
    Card,
    Delivery,
    AppState,
    # Constants
    SUITS,
    RANKS,
    TRAVEL_COST_PER_UNIT,
    VERIFIER_X,
    VERIFIER_Y,
    SUPERVISOR_SYSTEM_PROMPT,
    PICKUP_AGENT_SYSTEM_PROMPT,
    HAIKU_INPUT_COST_PER_M,
    HAIKU_OUTPUT_COST_PER_M,
    # Module-level state
    _active_dashboard,
    _active_event_log,
    # Node functions (agents)
    scatter_node,
    supervisor_node,
    timer_start_node,
    pickup_node,
    llm_pickup_node,
    delivery_node,
    timer_stop_node,
    verify_node,
    # Graph
    build_graph,
    _make_initial_state,
    # Utilities
    _card_key,
    _find_card_by_key,
    _greedy_nearest_card,
    _plan_agent_moves,
    _resolve_conflicts,
    _analyze_scatter,
    _compute_deployment_positions,
    _determine_region,
    _pickup_region,
    _extract_elapsed,
    _extract_timing,
    _estimate_cost,
    # CLI / runners
    run_trials,
    run_supervisor_comparison,
    run_llm_comparison,
    print_summary,
    main,
)
