from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"runs": [], "current_model_path": None}
    return json.loads(path.read_text())


def update_registry(path: Path, entry: dict[str, Any]) -> None:
    data = load_registry(path)
    data["runs"].append(entry)
    data["current_model_path"] = entry.get("model_path")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def current_model_path(path: Path) -> str | None:
    data = load_registry(path)
    return data.get("current_model_path")
