from typing import Dict, List


def summarize(episodes: List[Dict]) -> Dict:
    if not episodes:
        return {"success_rate": 0.0, "avg_steps": 0.0, "spl": 0.0}
    success = [e for e in episodes if e.get("success")]
    avg_steps = sum(e.get("steps", 0) for e in success) / max(1, len(success))
    spl_values = []
    for ep in episodes:
        if not ep.get("success"):
            spl_values.append(0.0)
            continue
        start_distance = ep.get("start_distance", 1.0)
        min_steps = max(1, int(round(start_distance / 0.25)))
        steps = max(1, int(ep.get("steps", 1)))
        spl_values.append(min_steps / max(steps, min_steps))
    spl = sum(spl_values) / len(episodes)
    return {"success_rate": len(success) / len(episodes), "avg_steps": avg_steps, "spl": spl}
