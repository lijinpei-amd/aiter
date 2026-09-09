"""Resolve exact benchmark cases inherited from preserved evidence harnesses."""

from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def source_files() -> list[Path]:
    declarations = json.loads((ROOT / "cases.json").read_text())
    paths = {ROOT / "cases.json"}
    paths.update(REPO / declaration["source"] for declaration in declarations.values())
    return sorted(paths)


def _merge(base: dict, updates: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_cases(tokens: int | None = None) -> dict[str, dict]:
    declarations = json.loads((ROOT / "cases.json").read_text())
    loaded: dict[Path, dict] = {}
    result = {}
    for name, declaration in declarations.items():
        source = (REPO / declaration["source"]).resolve()
        if source not in loaded:
            loaded[source] = json.loads(source.read_text())
        source_case = declaration["source_case"]
        resolved = copy.deepcopy(loaded[source][source_case])
        resolved = _merge(resolved, declaration.get("overrides", {}))
        if tokens is not None:
            resolved = _merge(
                resolved, declaration.get("by_token", {}).get(str(tokens), {})
            )
        resolved["description"] = declaration["description"]
        resolved["config_origin"] = {
            "path": str(source.relative_to(REPO)),
            "case": source_case,
        }
        result[name] = resolved
    return result
