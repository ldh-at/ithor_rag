import json
import os
from typing import Dict, List


class RagStore:
    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8"):
                pass
        self.entries: List[Dict] = self._load()

    def _load(self) -> List[Dict]:
        entries = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if "id" not in entry:
                    entry["id"] = len(entries)
                entries.append(entry)
        return entries

    def upsert(self, text: str, metadata: Dict) -> None:
        entry = {"id": len(self.entries), "text": text, "metadata": metadata}
        self.entries.append(entry)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def all(self) -> List[Dict]:
        return self.entries
