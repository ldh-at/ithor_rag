import cv2
import numpy as np


def save_frame(path: str, frame: np.ndarray) -> None:
    cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
