from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List


def comparison_markdown(rows: Iterable[Dict[str, object]]) -> str:
    rows = list(rows)
    if not rows:
        return ""
    headers = list(rows[0].keys())
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        table.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(table)


def save_comparison_markdown(path: str | Path, rows: Iterable[Dict[str, object]]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(comparison_markdown(rows))
    return destination
