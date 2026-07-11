"""Shared helper for settings mouseover tooltips.

Every settings control shows, on hover, the setting's variable path (in bold) and
a short plain-language description — so the label can stay human/intuitive while
the exact TOML key stays discoverable. Qt renders the returned HTML in tooltips.
"""
from __future__ import annotations

import html


def tooltip(var_path: str, description: str) -> str:
    """Rich tooltip: the bold settings variable path over a one-line description."""
    return f"<b>{html.escape(var_path)}</b><br>{html.escape(description)}"
