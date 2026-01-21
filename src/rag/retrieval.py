from collections import Counter
from typing import Dict, List, Tuple


def tokenize(text: str) -> List[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return [tok for tok in cleaned.split() if tok]


def similarity(a: str, b: str) -> float:
    tokens_a = tokenize(a)
    tokens_b = tokenize(b)
    if not tokens_a or not tokens_b:
        return 0.0
    counter_a = Counter(tokens_a)
    counter_b = Counter(tokens_b)
    common = sum((counter_a & counter_b).values())
    denom = (sum(counter_a.values()) * sum(counter_b.values())) ** 0.5
    if denom == 0:
        return 0.0
    return common / denom


def retrieve(entries: List[Dict], query: str, top_k: int, types: List[str]) -> List[Dict]:
    scored: List[Tuple[float, Dict]] = []
    for entry in entries:
        meta = entry.get("metadata", {})
        if types and meta.get("type") not in types:
            continue
        score = similarity(query, entry.get("text", ""))
        scored.append((score, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:top_k]]
