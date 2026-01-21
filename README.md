ObjectNav VLM RAG Pipeline
==========================

Runtime vs Eval Separation
--------------------------
- Runtime action selection uses only RGB + prompt text (recent actions + RAG snippets).
- Environment metadata is logged under `env_meta_for_eval_only` and is never passed to the planner.
- Hallucination labels are computed after each episode using `env_meta_for_eval_only`.

Loop-Break Heuristics
---------------------
- If `RotateLeft` repeats 4+ times consecutively, force one `RotateRight`.
- If `RotateRight` repeats 4+ times consecutively, force one `RotateLeft`.
- If the last 3 `MoveAhead` attempts collide, alternate turns using a `turn_toggle`.

iTHOR Episodes
--------------
- You can drive episodes from a dataset via `prior` or from a local `episodes_file`.
- Config keys under `run`:
  - `episodes_file`: path to a JSON file with `episodes`.
  - `dataset`: name passed to `prior.load_dataset(...)`.
  - `split`: split name, e.g. `minival`, `train`, `val`, `test`.
  - `allow_fallback`: when `false`, requires dataset or episodes_file.
- If neither `episodes_file` nor `dataset` is set, the pipeline falls back to
  `scenes` + `target_objects`.
