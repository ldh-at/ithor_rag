import hashlib
import json
import os
import re
from collections import deque
from typing import Deque, Dict, List

from ..env.thor_objectnav_env import ThorObjectNavEnv
from ..rag.memory_types import build_dir, build_loc, build_place
from ..rag.retrieval import retrieve
from ..rag.store import RagStore
from ..vlm.parsing import parse_action_line, safe_fallback
from ..vlm.prompt_builder import build_prompt
from ..vlm.qwen_vl_hf import QwenVLHF
from .action_space import ACTIONS, make_action
from .trajectory import Trajectory
from ..utils.images import save_frame
from ..utils.logging import append_jsonl


def run_episode(
    cfg: Dict,
    env: ThorObjectNavEnv,
    model: QwenVLHF,
    prompt_tmpl: str,
    rag_store: RagStore,
    output_dir: str,
    episode_id: int = 0,
    scene: str = "",
    start_pose: Dict = None,
) -> Dict:
    traj = Trajectory(history_k=cfg["agent"]["history_k"])
    max_steps = cfg["run"]["max_steps"]
    action_space = cfg["agent"]["action_space"]
    safe_fallback_action = cfg["agent"]["safe_fallback"]
    rag_cfg = cfg["rag"]
    rag_types = rag_cfg.get("memory_types_enabled", [])
    rag_top_k = rag_cfg.get("top_k", 3)
    log_cfg = cfg.get("logging", {})
    save_frames = bool(log_cfg.get("save_frames", False))
    frame_stride = int(log_cfg.get("frame_stride", 1))
    save_video = bool(log_cfg.get("save_video", False))
    debug_save_vlm_raw = bool(log_cfg.get("debug_save_vlm_raw", False))
    debug_save_rag_hits = bool(log_cfg.get("debug_save_rag_hits", False))
    debug_save_env_meta_full = bool(log_cfg.get("debug_save_env_meta_full", False))
    frames_dir = os.path.join(output_dir, "frames", f"episode_{episode_id:03d}")
    video_dir = os.path.join(output_dir, "videos")
    debug_dir = os.path.join(output_dir, "debug")
    steps_path = os.path.join(output_dir, "steps.jsonl")
    vlm_raw_dir = os.path.join(debug_dir, "vlm_raw")
    lmk_raw_dir = os.path.join(debug_dir, "lmk_raw", f"episode_{episode_id:03d}")
    rag_hits_dir = os.path.join(debug_dir, "rag_hits")
    env_meta_dir = os.path.join(debug_dir, "env_meta")
    if save_frames:
        os.makedirs(frames_dir, exist_ok=True)
    if debug_save_vlm_raw:
        os.makedirs(vlm_raw_dir, exist_ok=True)
        os.makedirs(lmk_raw_dir, exist_ok=True)
    if debug_save_rag_hits:
        os.makedirs(rag_hits_dir, exist_ok=True)
    if debug_save_env_meta_full:
        os.makedirs(env_meta_dir, exist_ok=True)
    video_writer = None
    if save_video:
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, f"episode_{episode_id:03d}.mp4")
        import cv2

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_path, fourcc, 10, (env.width, env.height))

    if scene:
        event = env.reset(scene=scene, start_pose=start_pose)
    else:
        event = env.reset(env.scene, start_pose=start_pose)
    target_object_type = cfg.get("target_object_type", cfg["target"])
    target_prompt = cfg.get("target_prompt", cfg["target"])
    start_visible = env.list_visible(event, target_object_type)
    start_distance = start_visible.get("target_distance")
    if start_distance is None:
        start_distance = float(cfg["run"].get("success_distance", 1.0))
    steps: List[Dict] = []
    collisions = 0
    overconfident_stop = 0
    failed_move_ahead = 0
    last_actions: Deque[str] = deque(maxlen=8)
    last_collisions: Deque[bool] = deque(maxlen=8)
    turn_toggle = False

    def turn_bias(actions: List[str]) -> str:
        left = actions.count("RotateLeft")
        right = actions.count("RotateRight")
        if left == right:
            return "neutral"
        return "left" if left > right else "right"

    def truncate_snippet(text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        snippet = "\n".join(lines[:2])
        return snippet[:220]

    def extract_lmk_list(raw_text: str) -> List[str]:
        match = re.search(r"LMK\s*[:=]\s*(.*)", raw_text, re.IGNORECASE)
        if not match:
            return []
        payload = match.group(1).splitlines()[0].strip()
        if ";" in payload:
            payload = payload.split(";", 1)[0].strip()
        if not payload or payload.lower() == "none":
            return []
        items = [item.strip() for item in payload.split(",") if item.strip()]
        return items[:5]

    def extract_seen_flag(raw_text: str):
        match = re.search(r"SEEN\s*[:=]\s*(yes|no|true|false)", raw_text, re.IGNORECASE)
        if not match:
            return None
        val = match.group(1).lower()
        return val in ("yes", "true")

    def extract_loc(raw_text: str) -> str:
        match = re.search(r"LOC\s*[:=]\s*([^;\\n]+)", raw_text, re.IGNORECASE)
        if not match:
            return ""
        loc = match.group(1).strip()
        return loc if loc.lower() != "none" else ""

    stoplist = {
    }

    def normalize_lmk(text: str) -> str:
        cleaned = re.sub(r"\\s*\\(.*?\\)\\s*", "", text)
        cleaned = re.sub(r"[^a-zA-Z0-9\\s]", "", cleaned)
        return cleaned.strip().lower()

    def is_stop_lmk(text: str) -> bool:
        norm = normalize_lmk(text)
        if not norm:
            return True
        if norm in stoplist:
            return True
        if norm.endswith((" wall", " floor", " ceiling", " baseboard")):
            return True
        return False
    def format_rag_snippets_merged(hits: List[Dict]) -> List[str]:
        place_payload = None 
        loc_payload = None
        rest = []

        for h in hits:
            t = (h.get("text") or "").strip()
            if t.startswith("PLACE:") and place_payload is None:
                p = t.replace("PLACE:", "", 1).strip()
                if p.lower() != "none":
                    place_payload = p
                continue
            if t.startswith("LOC:") and loc_payload is None:
                l = t.replace("LOC:", "", 1).strip()
                if l.lower() != "none":
                    loc_payload = l
                continue
            rest.append(t)

        snippets = []

        if place_payload and loc_payload:
            snippets.append(
                f"Memory: In {loc_payload}, these landmarks are near each other: {place_payload}."
            )
        elif place_payload:
            snippets.append(
                f"Memory: these landmarks are near each other: {place_payload}."
            )
        elif loc_payload:
            snippets.append(
                f"Memory: location hint is {loc_payload}."
            )

        for t in rest:
            snippets.append(format_rag_snippet(t))

        return snippets


    def format_rag_snippet(text: str) -> str:
        if text.startswith("PLACE:"):
            payload = text.replace("PLACE:", "", 1).strip()
            return f"Memory: these landmarks are near each other: {payload}."
        if text.startswith("LOC:"):
            payload = text.replace("LOC:", "", 1).strip()
            return f"Memory: location hint is {payload}."
        if text.startswith("DIR:"):
            payload = text.replace("DIR:", "", 1).strip()
            return f"Memory: {payload}."
        return truncate_snippet(text)
    for step_idx in range(max_steps):
        frame = env.get_frame(event)
        if save_frames and step_idx % max(1, frame_stride) == 0:
            save_frame(os.path.join(frames_dir, f"step_{step_idx:05d}.png"), frame)
        if video_writer is not None:
            import cv2

            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        current_lmks = []
        lmk_seen = None
        lmk_raw = ""
        lmk_preview = ""
        query = f"target={target_prompt} lmk=none"
        rag_snippets = []
        rag_hit_ids = []
        rag_hits = []
        if rag_cfg.get("mode") == "retrieve":
            lmk_prompt = (
                f"Target: {target_prompt}. "
                "List up to 5 prominent objects you can see in the image, "
                "and a short location hint (e.g., kitchen, bedroom). "
                "and target object visibility."
                "Return exactly one line: "
                "LMK=<comma-separated objects or none>; SEEN=<yes/no>; LOC=<short location or none>"
            )
            lmk_raw, _ = model.generate_with_debug(frame, lmk_prompt, max_new_tokens=32)
            lmk_preview = lmk_raw[:120]
            lmk_seen = extract_seen_flag(lmk_raw)
            lmk_loc = extract_loc(lmk_raw)
            lmks = [item for item in extract_lmk_list(lmk_raw) if not is_stop_lmk(item)]
            seen = set()
            current_lmks = []
            for item in lmks:
                key = item.lower()
                if key in seen:
                    continue
                seen.add(key)
                current_lmks.append(item)
                if len(current_lmks) >= 5:
                    break
            if current_lmks:
                query = f"target={target_prompt} lmk={', '.join(current_lmks)}"
            if debug_save_vlm_raw:
                lmk_path = os.path.join(lmk_raw_dir, f"step_{step_idx:05d}.txt")
                with open(lmk_path, "w", encoding="utf-8") as f:
                    f.write(lmk_raw)
            hits = retrieve(rag_store.all(), query, rag_top_k, rag_types)
            rag_snippets = format_rag_snippets_merged(hits)
            rag_hit_ids = [h.get("id") for h in hits]
            rag_hits = hits

        loop_break_action = None
        if len(last_actions) >= 4 and all(a == "RotateLeft" for a in list(last_actions)[-4:]):
            loop_break_action = "RotateRight"
        elif len(last_actions) >= 4 and all(a == "RotateRight" for a in list(last_actions)[-4:]):
            loop_break_action = "RotateLeft"
        elif len(last_actions) >= 6:
            tail = list(last_actions)[-6:]
            if all(a in ("LookUp", "LookDown") for a in tail) and {
                "LookUp",
                "LookDown",
            }.issubset(set(tail)):
                loop_break_action = "RotateLeft" if turn_toggle else "RotateRight"
                turn_toggle = not turn_toggle
        moveahead_recent = [
            c
            for a, c in zip(list(last_actions), list(last_collisions))
            if a == "MoveAhead"
        ]
        if len(moveahead_recent) >= 3 and all(moveahead_recent[-3:]):
            loop_break_action = "RotateLeft" if turn_toggle else "RotateRight"
            turn_toggle = not turn_toggle

        if loop_break_action:
            action = loop_break_action
            raw = ""
            vlm_output = {"action": action, "source": "loop_breaker"}
            loop_break_triggered = True
            planner_input_token_estimate = 0
        else:
            prompt = build_prompt(
                prompt_tmpl, target_prompt, action_space, traj.summary(), rag_snippets
            )
            planner_input_token_estimate = len(prompt.split())
            raw, vlm_debug = model.generate_with_debug(frame, prompt)
            parsed_action = parse_action_line(raw)
            if parsed_action is None:
                parsed_action = safe_fallback(safe_fallback_action)
            action = parsed_action if parsed_action in ACTIONS else safe_fallback_action
            vlm_output = {"action": action, "source": "planner"}
            loop_break_triggered = False

        if step_idx < 30:
            print(
                f"[sanity] tokens={planner_input_token_estimate} action={action} loop_break={loop_break_triggered}"
            )

        action_dict = make_action(action)
        if action == "Stop":
            done = True
        else:
            event = env.step(action_dict)
            done = False

        visible_info = env.list_visible(event, target_object_type)

        last_success = bool(event.metadata.get("lastActionSuccess", True)) if not done else True
        collision = not last_success
        if collision:
            collisions += 1
        if action == "MoveAhead" and not last_success:
            failed_move_ahead += 1
        traj.add(action, last_success, collision, {"t": step_idx})
        last_actions.append(action)
        last_collisions.append(collision)

        memory_updates = []
        if "PLACE" in rag_types:
            text = build_place(current_lmks)
            rag_store.upsert(text, {"type": "PLACE"})
            memory_updates.append({"type": "PLACE", "text": text})
        if "LOC" in rag_types:
            text = build_loc(lmk_loc)
            rag_store.upsert(text, {"type": "LOC"})
            memory_updates.append({"type": "LOC", "text": text})
        if "DIR" in rag_types:
            text = build_dir("unknown")
            rag_store.upsert(text, {"type": "DIR"})
            memory_updates.append({"type": "DIR", "text": text})

        raw_preview = raw[:200]
        raw_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        raw_full = None
        if debug_save_vlm_raw:
            raw_full = vlm_debug.get("full_text", raw)
            raw_path = os.path.join(vlm_raw_dir, f"step_{step_idx:05d}.txt")
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(raw_full)
        if debug_save_rag_hits:
            hits_path = os.path.join(rag_hits_dir, f"step_{step_idx:05d}.json")
            with open(hits_path, "w", encoding="utf-8") as f:
                json.dump(rag_hits, f, indent=2)
        if debug_save_env_meta_full:
            meta_path = os.path.join(env_meta_dir, f"step_{step_idx:05d}.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(visible_info, f, indent=2)

        step_record = {
            "step_idx": step_idx,
            "episode_id": episode_id,
            "scene": env.scene,
            "target_object_type": target_object_type,
            "target_prompt": target_prompt,
            "action": action,
            "collision": collision,
            "vlm_output": vlm_output,
            "vlm_raw_preview": raw_preview,
            "vlm_raw_hash": raw_hash,
            "lmk_preview": lmk_preview,
            "lmk_list": current_lmks,
            "target_seen_claim": lmk_seen,
            "target_loc_claim": lmk_loc,
            "rag_hit_ids": rag_hit_ids,
            "env_meta_for_eval_only": {
                "target_visible": visible_info.get("target_visible"),
                "target_bbox": visible_info.get("target_bbox"),
                "target_distance": visible_info.get("target_distance"),
                "frame_width": frame.shape[1],
            },
            "memory_updates": memory_updates,
            "loop_break_triggered": loop_break_triggered,
            "planner_input_token_estimate": planner_input_token_estimate,
        }
        if raw_full is not None:
            step_record["vlm_raw"] = raw_full
        steps.append(step_record)
        append_jsonl(steps_path, [step_record])

        if done:
            break

    if video_writer is not None:
        video_writer.release()

    success = False
    success_distance = float(cfg["run"].get("success_distance", 1.0))
    if action == "Stop" and visible_info.get("target_visible"):
        distance = visible_info.get("target_distance")
        if distance is None:
            success = True
        else:
            success = distance <= success_distance
    if action == "Stop" and not success:
        overconfident_stop += 1
    episode_summary = {
        "success": success,
        "steps": len(steps),
        "episode_id": episode_id,
        "scene": env.scene,
        "target_object_type": target_object_type,
        "target_prompt": target_prompt,
        "collisions": collisions,
        "failed_move_ahead": failed_move_ahead,
        "overconfident_stop": overconfident_stop,
        "start_distance": start_distance,
    }
    return {"steps": steps, "summary": episode_summary}
