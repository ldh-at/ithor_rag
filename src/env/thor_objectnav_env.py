from typing import Any, Dict, List, Optional

from ai2thor.controller import Controller

from .headless import apply_graphics_env, configure_headless, start_xvfb


class ThorObjectNavEnv:
    def __init__(
        self,
        scene: str,
        headless: bool = True,
        width: int = 300,
        height: int = 300,
        seed: int = 0,
        use_cloud: bool = False,
        server_timeout: float = 300.0,
        server_start_timeout: float = 600.0,
        unity_log_file: Optional[str] = None,
        use_xvfb: bool = False,
        xvfb_display: str = ":99",
        graphics_cfg: Optional[Dict] = None,
    ) -> None:
        self.scene = scene
        self.width = width
        self.height = height
        self.seed = seed
        self.use_cloud = use_cloud
        self.server_timeout = server_timeout
        self.server_start_timeout = server_start_timeout
        self.unity_log_file = unity_log_file
        apply_graphics_env(graphics_cfg or {})
        import os

        os.environ.setdefault("AI2THOR_DISABLE_READ_CONFIG", "1")
        if headless:
            configure_headless(offscreen=True)
        self._xvfb_proc = None
        if use_xvfb:
            self._xvfb_proc = start_xvfb(xvfb_display)
            headless = False
        if self.use_cloud:
            os.environ.setdefault("AI2THOR_USE_CLOUD", "1")
        if self.unity_log_file:
            os.environ.setdefault("UNITY_LOG_FILE", self.unity_log_file)
        platform = None
        local_executable_path = None
        if self.use_cloud:
            try:
                from ai2thor import platform as thor_platform

                platform = thor_platform.CloudRendering
            except Exception:
                platform = "CloudRendering"
            # Prefer CloudRendering local build if available.
            import glob
            import os

            candidates = sorted(
                glob.glob(os.path.expanduser("~/.ai2thor/releases/thor-CloudRendering-*")),
                reverse=True,
            )
            for candidate in candidates:
                exe_name = os.path.basename(candidate)
                exe_path = os.path.join(candidate, exe_name)
                if os.path.isfile(exe_path) and os.access(exe_path, os.X_OK):
                    local_executable_path = exe_path
                    break
        self.controller = Controller(
            scene=scene,
            width=width,
            height=height,
            agentMode="default",
            renderInstanceSegmentation=False,
            renderDepthImage=False,
            renderClassImage=False,
            renderObjectImage=False,
            headless=headless,
            platform=platform,
            local_executable_path=local_executable_path,
            x_display=xvfb_display if use_xvfb else None,
            gpu_device=None,
            server_timeout=server_timeout,
            server_start_timeout=server_start_timeout,
            unity_log_file=self.unity_log_file,
        )
        self.reset(scene)

    def reset(self, scene: Optional[str] = None, start_pose: Optional[Dict[str, Any]] = None) -> Any:
        if scene:
            self.scene = scene
        event = self.controller.reset(self.scene)
        self.controller.step(action="Initialize", gridSize=0.25, agentMode="default")
        if start_pose:
            position = start_pose.get("position")
            rotation = start_pose.get("rotation")
            horizon = start_pose.get("horizon")
            standing = start_pose.get("standing")
            teleport_args = {
                "action": "TeleportFull",
                "forceAction": True,
            }
            if position is not None:
                teleport_args["position"] = position
            if rotation is not None:
                teleport_args["rotation"] = rotation
            if horizon is not None:
                teleport_args["horizon"] = horizon
            if standing is not None:
                teleport_args["standing"] = standing
            event = self.controller.step(**teleport_args)
        return event

    def step(self, action: Dict[str, Any]) -> Any:
        return self.controller.step(**action)

    def get_frame(self, event: Any):
        return event.frame

    def get_metadata(self, event: Any) -> Dict[str, Any]:
        return event.metadata

    def close(self) -> None:
        self.controller.stop()
        if self._xvfb_proc is not None:
            self._xvfb_proc.terminate()

    def list_visible(self, event: Any, target: str) -> Dict[str, Any]:
        objects = event.metadata.get("objects", [])
        visible = [o for o in objects if o.get("visible")]
        target_visible = False
        target_bbox = None
        target_distance = None
        for obj in visible:
            if obj.get("objectType", "").lower() == target.lower():
                target_visible = True
                target_bbox = obj.get("boundingBox")
                target_distance = obj.get("distance")
                break
        return {
            "visible": visible,
            "target_visible": target_visible,
            "target_bbox": target_bbox,
            "target_distance": target_distance,
        }
