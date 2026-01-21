import json
from typing import Dict, List, Optional, Tuple, Any
import gzip
import glob
import os


def _find_episodes(obj) -> Optional[List[Dict]]:
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        if any(key in obj[0] for key in ("scene", "scene_id", "scene_name")):
            return obj
    if hasattr(obj, "__dict__"):
        return _find_episodes(obj.__dict__)
    if isinstance(obj, dict):
        for value in obj.values():
            found = _find_episodes(value)
            if found is not None:
                return found
    return None


def _get_scene_and_target(ep: Dict) -> Tuple[Optional[str], Optional[str]]:
    scene = (
        ep.get("scene_id")
        or ep.get("scene")
        or ep.get("scene_name")
        or ep.get("sceneId")
    )
    target = (
        ep.get("object_type")
        or ep.get("objectType")
        or ep.get("target_object_type")
        or ep.get("targetObjectType")
        or ep.get("targetObjectType")
        or ep.get("target")
    )
    return scene, target


def _load_jsonl(path: str) -> List[Dict]:
    opener = gzip.open if path.endswith(".gz") else open
    episodes = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            episodes.append(json.loads(line))
    return episodes


def _find_cached_objectnav_eval(split: str) -> Optional[str]:
    base = os.path.expanduser("~/.prior/datasets/allenai/object-nav-eval")
    if not os.path.isdir(base):
        return None
    candidates = []
    for root in glob.glob(os.path.join(base, "*", "datasets")):
        fname = f"ithor-locobot-{split}.jsonl.gz"
        path = os.path.join(root, fname)
        if os.path.isfile(path):
            candidates.append(path)
    return candidates[0] if candidates else None


def load_episodes(cfg: Dict) -> Optional[List[Dict]]:
    run_cfg = cfg.get("run", {})
    dataset = run_cfg.get("dataset")
    if not dataset:
        episodes_file = run_cfg.get("episodes_file")
        if episodes_file:
            if episodes_file.endswith((".jsonl", ".jsonl.gz")):
                return _load_jsonl(episodes_file)
            with open(episodes_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return payload.get("episodes") if isinstance(payload, dict) else payload
        return None
    try:
        import prior  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("prior is required for dataset loading") from exc

    minival = bool(run_cfg.get("minival", True))
    split = run_cfg.get("split", "minival")
    try:
        data = prior.load_dataset(dataset, minival=minival)
    except Exception:
        if dataset == "object-nav-eval":
            split_key = "minival" if split == "minival" else split
            cached = _find_cached_objectnav_eval(split_key)
            if cached:
                return _load_jsonl(cached)
        raise
    if not isinstance(data, dict):
        if hasattr(data, split):
            split_data = getattr(data, split)
        elif split == "minival" and hasattr(data, "val"):
            split_data = getattr(data, "val")
        else:
            split_data = data
        ithor = getattr(split_data, "ithor", None) or split_data
        episodes = getattr(ithor, "episodes", None)
        if episodes is not None:
            return episodes
        episodes = _find_episodes(split_data)
        if episodes is not None:
            return episodes
        data = getattr(data, "__dict__", {})
    ithor = data.get("ithor", data) if isinstance(data, dict) else data
    split_data = ithor.get(split, ithor) if isinstance(ithor, dict) else ithor
    episodes = split_data.get("episodes") if isinstance(split_data, dict) else None
    if episodes is None:
        episodes = _find_episodes(split_data)
    if episodes is None and dataset == "object-nav-eval":
        split_key = "minival" if split == "minival" else split
        cached = _find_cached_objectnav_eval(split_key)
        if cached:
            return _load_jsonl(cached)
    return episodes


def _pick_pose_field(ep: Dict, keys: List[str]) -> Optional[Any]:
    for key in keys:
        if key in ep:
            return ep[key]
    return None


def extract_start_pose(ep: Dict) -> Optional[Dict[str, Any]]:
    agent_pose = ep.get("agentPose")
    if isinstance(agent_pose, dict):
        position = agent_pose.get("position")
        rotation = agent_pose.get("rotation")
        horizon = agent_pose.get("horizon") or agent_pose.get("cameraHorizon")
        standing = agent_pose.get("standing") or agent_pose.get("isStanding")
        if isinstance(rotation, (int, float)):
            rotation = {"x": 0.0, "y": float(rotation), "z": 0.0}
        return {
            "position": position,
            "rotation": rotation,
            "horizon": horizon,
            "standing": True if standing is None else bool(standing),
        }
    pose = _pick_pose_field(ep, ["agent_start_pose", "start_pose", "initial_pose"])
    if isinstance(pose, dict):
        position = pose.get("position") or pose.get("pos")
        rotation = pose.get("rotation") or pose.get("rot")
        horizon = pose.get("horizon") or pose.get("cameraHorizon")
        standing = pose.get("standing") or pose.get("isStanding")
        if isinstance(rotation, (int, float)):
            rotation = {"x": 0.0, "y": float(rotation), "z": 0.0}
        return {
            "position": position,
            "rotation": rotation,
            "horizon": horizon,
            "standing": True if standing is None else bool(standing),
        }
    position = _pick_pose_field(
        ep,
        [
            "agent_start_position",
            "start_position",
            "initial_position",
            "agentStartPosition",
        ],
    )
    rotation = _pick_pose_field(
        ep,
        [
            "agent_start_rotation",
            "start_rotation",
            "initial_rotation",
            "agentStartRotation",
        ],
    )
    if isinstance(rotation, (int, float)):
        rotation = {"x": 0.0, "y": float(rotation), "z": 0.0}
    horizon = _pick_pose_field(
        ep,
        [
            "agent_start_horizon",
            "start_horizon",
            "initial_horizon",
            "cameraHorizon",
        ],
    )
    standing = _pick_pose_field(ep, ["isStanding", "standing"])
    if position is None and rotation is None and horizon is None:
        return None
    return {
        "position": position,
        "rotation": rotation,
        "horizon": horizon,
        "standing": True if standing is None else bool(standing),
    }


def pick_episode(episodes: List[Dict], idx: int) -> Tuple[str, str, Dict]:
    ep = episodes[idx % len(episodes)]
    scene, target = _get_scene_and_target(ep)
    if scene is None or target is None:
        raise ValueError("Episode missing scene or target fields")
    return scene, target, ep
