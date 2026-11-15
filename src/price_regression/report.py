"""Plain fixed-width terminal tables (no dependencies). Korean characters count as 2 columns."""
from __future__ import annotations

import unicodedata
from typing import Sequence


def display_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def pad(text: str, width: int, right: bool = False) -> str:
    gap = " " * max(0, width - display_width(text))
    return gap + text if right else text + gap


def fmt(value: float, digits: int = 1) -> str:
    """Format any number-like value (numpy included) as a plain decimal string."""
    return f"{float(value):,.{digits}f}"


def table(headers: Sequence[str], rows: Sequence[Sequence[object]], right_from: int = 1) -> str:
    """Left-align the first ``right_from`` columns, right-align the rest."""
    cells = [[str(c) for c in row] for row in rows]
    widths = [display_width(h) for h in headers]
    for row in cells:
        for i, c in enumerate(row):
            widths[i] = max(widths[i], display_width(c))

    def line(row: Sequence[str]) -> str:
        return "  ".join(pad(c, widths[i], right=i >= right_from) for i, c in enumerate(row)).rstrip()

    out = [line(list(headers)), "  ".join("-" * w for w in widths)]
    out += [line(row) for row in cells]
    return "\n".join(out)
