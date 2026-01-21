ACTIONS = ["MoveAhead", "RotateLeft", "RotateRight", "LookUp", "LookDown", "Stop"]


def make_action(name: str) -> dict:
    if name == "MoveAhead":
        return {"action": "MoveAhead", "moveMagnitude": 0.25}
    if name == "RotateLeft":
        return {"action": "RotateLeft", "degrees": 90}
    if name == "RotateRight":
        return {"action": "RotateRight", "degrees": 90}
    if name == "LookUp":
        return {"action": "LookUp", "degrees": 30}
    if name == "LookDown":
        return {"action": "LookDown", "degrees": 30}
    return {"action": "Done"}
