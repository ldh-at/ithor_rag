import re
from typing import Optional


def parse_action_line(text: str) -> Optional[str]:
    matches = list(re.finditer(r"ACTION\s*=\s*([A-Za-z_]+)", text))
    if matches:
        return matches[-1].group(1)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    last_line = lines[-1]
    last_line = re.sub(r"^assistant\s*", "", last_line, flags=re.IGNORECASE).strip()
    match = re.match(r"^(?:ACTION\s*=\s*)?([A-Za-z_]+)$", last_line)
    if match:
        return match.group(1)
    return None


def safe_fallback(action: str) -> str:
    return action
