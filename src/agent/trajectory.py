from collections import deque
from typing import Deque, Dict, List


class Trajectory:
    def __init__(self, history_k: int = 6) -> None:
        self.history_k = history_k
        self.actions: Deque[str] = deque(maxlen=history_k)
        self.outcomes: Deque[Dict] = deque(maxlen=history_k)

    def add(self, action: str, success: bool, collision: bool, info: Dict) -> None:
        self.actions.append(action)
        payload = {"action": action, "success": success, "collision": collision}
        payload.update(info)
        self.outcomes.append(payload)

    def summary(self) -> str:
        items = list(self.outcomes)[-self.history_k :]
        if not items:
            return "(none)"
        parts = []
        for item in items:
            action = item["action"]
            if item.get("collision"):
                parts.append(f"{action} (blocked)")
            else:
                parts.append(action)
        return ". ".join(parts)

    def recent_actions(self) -> List[str]:
        return list(self.actions)
