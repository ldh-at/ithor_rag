import argparse
import json
import os
import time
import yaml

from .agent.loop import run_episode
from .env.thor_objectnav_env import ThorObjectNavEnv
from .rag.store import RagStore
from .metrics.nav_metrics import summarize
from .metrics.hallucinations import annotate_steps_for_eval
from .utils.logging import append_jsonl, ensure_dir, write_json
from .utils.episodes import load_episodes, pick_episode, extract_start_pose
from .vlm.qwen_vl_hf import QwenVLHF


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sweep", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    run_id = f"run_{int(time.time())}"
    output_dir = os.path.join(cfg["run"]["output_dir"], run_id)
    ensure_dir(output_dir)
    write_json(os.path.join(output_dir, "config.yaml"), cfg)

    model = QwenVLHF(cfg["model"]["local_path"], cfg["model"]["device"])
    unity_log = os.path.join(output_dir, "unity_player.log")

    episodes = load_episodes(cfg)
    if episodes is None and not cfg["run"].get("allow_fallback", False):
        raise RuntimeError("No episodes found. Set run.dataset or run.episodes_file, or allow_fallback=true.")
    if episodes:
        initial_scene, _, _ = pick_episode(episodes, 0)
    else:
        initial_scene = cfg["run"]["scenes"][0]

    env = ThorObjectNavEnv(
        scene=initial_scene,
        headless=cfg["headless"]["enabled"],
        use_cloud=cfg["headless"].get("use_cloud", False),
        server_timeout=cfg["headless"].get("server_timeout", 300.0),
        server_start_timeout=cfg["headless"].get("server_start_timeout", 600.0),
        unity_log_file=unity_log,
        use_xvfb=cfg["headless"].get("use_xvfb", False),
        xvfb_display=cfg["headless"].get("xvfb_display", ":99"),
        graphics_cfg=cfg.get("graphics", {}),
    )
    rag_store = RagStore(os.path.join(output_dir, "rag_store.jsonl"))

    prompt_cfg = load_yaml(os.path.join(os.path.dirname(args.config), "prompt.yaml"))
    prompt_tmpl = prompt_cfg["planner"]["template"]

    episode_summaries = []
    all_steps = []
    for idx in range(cfg["run"]["num_episodes"]):
        if episodes:
            scene, target, ep = pick_episode(episodes, idx)
            cfg["target"] = target
            cfg["target_object_type"] = target
            cfg["target_prompt"] = str(target).lower()
            cfg["episode_meta"] = ep
            start_pose = extract_start_pose(ep)
        else:
            target = cfg["run"]["target_objects"][idx % len(cfg["run"]["target_objects"])]
            scene = cfg["run"]["scenes"][idx % len(cfg["run"]["scenes"])]
            cfg["target"] = target
            cfg["target_object_type"] = target
            cfg["target_prompt"] = str(target).lower()
            cfg.pop("episode_meta", None)
            start_pose = None
        episode_rec = {
            "episode_id": idx,
            "scene": scene,
            "target": target,
            "start_pose": start_pose,
            "seed": cfg["run"].get("seed"),
            "split": cfg["run"].get("split"),
            "dataset": cfg["run"].get("dataset"),
        }
        append_jsonl(os.path.join(output_dir, "episode_meta.jsonl"), episode_rec)
        result = run_episode(
            cfg,
            env,
            model,
            prompt_tmpl,
            rag_store,
            output_dir,
            episode_id=idx,
            scene=scene,
            start_pose=start_pose,
        )
        annotate_steps_for_eval(result["steps"])
        episode_summaries.append(result["summary"])
        all_steps.extend(result["steps"])
        steps_eval_path = os.path.join(output_dir, "steps_eval.jsonl")
        with open(steps_eval_path, "a", encoding="utf-8") as f:
            for item in result["steps"]:
                f.write(json.dumps(item) + "\n")
        write_json(
            os.path.join(output_dir, "episode_summary.json"),
            {"episodes": episode_summaries},
        )

    metrics = summarize(episode_summaries)
    halluc_counts = {"PH_Existence": 0, "PH_Localization": 0, "overconfident_stop": 0}
    for step in all_steps:
        hall = step.get("hallucinations", {})
        if hall.get("PH_Existence"):
            halluc_counts["PH_Existence"] += 1
        if hall.get("PH_Localization"):
            halluc_counts["PH_Localization"] += 1
    overconf = sum(ep.get("overconfident_stop", 0) for ep in episode_summaries)
    halluc_counts["overconfident_stop"] = overconf
    write_json(os.path.join(output_dir, "metrics.json"), {"nav": metrics, "hallucinations": halluc_counts})

    env.close()


if __name__ == "__main__":
    main()
