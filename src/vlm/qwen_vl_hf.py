import re
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


class QwenVLHF:
    def __init__(self, model_path: str, device: str = "cuda") -> None:
        self.model_path = model_path
        self.device = device
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=None,
            device_map="auto" if device.startswith("cuda") else None,
            trust_remote_code=True,
            local_files_only=True,
        )
        if device == "cpu":
            self.model.to("cpu")
        self.model.eval()

    def generate(self, frame: np.ndarray, prompt: str, max_new_tokens: int = 256) -> str:
        image = Image.fromarray(frame)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        if self.device.startswith("cuda"):
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return self._extract_assistant(decoded)

    def generate_with_debug(
        self, frame: np.ndarray, prompt: str, max_new_tokens: int = 256
    ) -> Tuple[str, Dict]:
        image = Image.fromarray(frame)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        if self.device.startswith("cuda"):
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        assistant_text = self._extract_assistant(decoded)
        debug = {
            "full_text": decoded,
            "input_text_preview": text[:400],
            "image_size": image.size,
            "image_mode": image.mode,
        }
        return assistant_text, debug

    @staticmethod
    def _extract_assistant(text: str) -> str:
        parts = re.split(r"\bassistant\b", text, flags=re.IGNORECASE)
        if parts:
            return parts[-1].strip()
        return text.strip()
