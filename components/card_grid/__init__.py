"""Interactive card grid component — canvas-based, clicks handled in JS."""

import os
from typing import Dict, List, Optional

import streamlit.components.v1 as components

_COMPONENT_DIR = os.path.join(os.path.dirname(__file__), "frontend")
_component_func = components.declare_component("card_grid", path=_COMPONENT_DIR)


def card_grid(
    cards: List[dict],
    agent_positions: Optional[Dict[str, list]] = None,
    config: Optional[dict] = None,
    key: Optional[str] = None,
) -> Optional[dict]:
    """Render an interactive card grid on an HTML5 canvas.

    Returns a dict like ``{"type": "card_click", "index": 5, "seq": 1}``
    or ``{"type": "verifier_click", "seq": 2}`` when the user clicks,
    or ``None`` when there is no new interaction.
    """
    return _component_func(
        cards=cards,
        agent_positions=agent_positions or {},
        config=config or {},
        key=key,
        default=None,
    )
