from typing import Dict, List


def target_relative_from_bbox(bbox: Dict, width: int) -> str:
    if not bbox or width <= 0:
        return "unknown"
    x = bbox.get("x", 0)
    w = bbox.get("width", 0)
    center = x + w / 2
    if center < width * 0.33:
        return "left"
    if center > width * 0.66:
        return "right"
    return "center"


def classify(
    vlm_output: Dict, env_meta: Dict, frame_width: int, target_seen_claim=None
) -> Dict:
    claims = vlm_output.get("state_claims", {}) if isinstance(vlm_output, dict) else {}
    if target_seen_claim is None:
        visible_claim = bool(claims.get("target_visible", False))
    else:
        visible_claim = bool(target_seen_claim)
    relative_claim = claims.get("target_relative", "unknown")
    target_visible = bool(env_meta.get("target_visible", False))
    bbox = env_meta.get("target_bbox")
    labels = {
        "PH_Existence": bool(visible_claim and not target_visible),
        "PH_Localization": False,
    }
    if target_visible and relative_claim in ("left", "center", "right"):
        actual = target_relative_from_bbox(bbox, frame_width)
        labels["PH_Localization"] = actual != "unknown" and actual != relative_claim
    return labels


def annotate_steps_for_eval(steps: List[Dict]) -> None:
    for step in steps:
        env_meta = step.get("env_meta_for_eval_only", {})
        frame_width = int(env_meta.get("frame_width", 0) or 0)
        step["hallucinations"] = classify(
            step.get("vlm_output", {}),
            env_meta,
            frame_width,
            step.get("target_seen_claim"),
        )
