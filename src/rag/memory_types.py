from typing import List


def build_place(landmarks: List[str]) -> str:
    if not landmarks:
        return "PLACE: none"
    return "PLACE: " + ", ".join(landmarks)


def build_dir(target_rel: str) -> str:
    return f"DIR: target_rel={target_rel}"


def build_loc(location: str) -> str:
    if not location:
        return "LOC: none"
    return f"LOC: {location}"
