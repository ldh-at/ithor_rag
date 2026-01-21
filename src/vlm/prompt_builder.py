from typing import List


def build_prompt(template: str, target: str, action_space: List[str], trajectory: str, rag_snippets: List[str]) -> str:
    rag_text = " | ".join(rag_snippets) if rag_snippets else "(none)"
    return template.format(
        target=target,
        action_space=action_space,
        trajectory=trajectory,
        rag_snippets=rag_text,
    )
