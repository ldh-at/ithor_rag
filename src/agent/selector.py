from typing import Dict, List

from .action_space import ACTIONS


def select_action(candidates: List[Dict], fallback: str) -> str:
    valid = [c for c in candidates if c.get("action") in ACTIONS]
    if not valid:
        return fallback
    valid.sort(key=lambda c: float(c.get("confidence", 0.0)), reverse=True)
    return valid[0]["action"]
