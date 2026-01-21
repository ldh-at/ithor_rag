import json
import os
from typing import Union, List, Dict


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def append_jsonl(path: str, items: Union[List[Dict], Dict]) -> None:
    if isinstance(items, dict):
        items = [items]
    with open(path, "a", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
