import os
import subprocess
import time
from typing import Dict, Optional


def configure_headless(offscreen: bool = True) -> None:
    """Best-effort headless config; AI2-THOR may need Vulkan/OpenGL on server."""
    os.environ.setdefault("AI2THOR_DISABLE_READ_CONFIG", "1")
    os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp")
    if offscreen:
        os.environ.setdefault("DISPLAY", "")
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


def start_xvfb(display: str = ":99") -> Optional[subprocess.Popen]:
    if os.environ.get("DISPLAY"):
        return None
    lock_path = f"/tmp/.X{display.lstrip(':')}-lock"
    if os.path.exists(lock_path):
        # Assume Xvfb is already running on this display.
        os.environ["DISPLAY"] = display
        return None
    proc = subprocess.Popen(
        [
            "Xvfb",
            display,
            "-screen",
            "0",
            "1024x768x24",
            "+extension",
            "GLX",
            "+render",
            "-noreset",
        ]
    )
    os.environ["DISPLAY"] = display
    time.sleep(1.0)
    return proc


def apply_graphics_env(graphics_cfg: Dict) -> None:
    if not graphics_cfg:
        return
    if graphics_cfg.get("force_software_gl"):
        os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
        os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "llvmpipe")
        os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "mesa")
    gl_version = graphics_cfg.get("gl_version")
    glsl_version = graphics_cfg.get("glsl_version")
    if gl_version:
        os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", str(gl_version))
    if glsl_version:
        os.environ.setdefault("MESA_GLSL_VERSION_OVERRIDE", str(glsl_version))
    if graphics_cfg.get("use_opengl"):
        os.environ.setdefault("UNITY_GFX_DEVICE", "OpenGLCore")
