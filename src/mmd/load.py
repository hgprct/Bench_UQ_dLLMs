"""Load generated run data for MMD computation."""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TEXT_FIELDS = ("x0_string", "x_string", "answer", "parsed_answer",
               "response", "content_response", "normalized_answer")


@dataclass(frozen=True)
class TrajectorySample:
    """One generated trajectory for one prompt."""
    sample_id: int
    response_sample_id: int
    example_id: str
    final_answer: str
    step_answers: tuple[str, ...]


@dataclass(frozen=True)
class PromptTrajectories:
    """All sampled trajectories for one original prompt."""
    prompt_id: str
    qa_index: int | None
    question: str
    samples: tuple[TrajectorySample, ...]
    num_steps: int


@dataclass(frozen=True)
class GenerationRun:
    """A generated run with all data needed for MMD computation."""
    run_dir: Path
    prompts: tuple[PromptTrajectories, ...]
    config: dict[str, Any]
    metadata: dict[str, Any]
    diagnostics: dict[str, Any]


def is_valid_text(value: Any) -> bool:
    """Check if a value represents meaningful text."""
    text = str(value).strip() if value is not None else ""
    return bool(text) and text.lower() not in {"none", "nan", "null"}


def load_generated_run(
    run_dir: str | Path,
    *,
    num_prompts: int | None = None,
    step_view: str = "x0",
) -> GenerationRun:
    """Load generated trajectories from pipeline outputs.

    Uses examples.jsonl + optional config.json/metadata.json.
    Does not require traces.npz.
    """
    if step_view not in {"x0", "x", "step"}:
        raise ValueError(f"Unsupported step_view '{step_view}'. Use x0, x, or step.")

    examples_path, config_path, metadata_path, resolved_dir = _resolve_paths(run_dir)
    records = _read_jsonl(examples_path)
    config = _read_json(config_path) if config_path else {}
    metadata = _read_json(metadata_path) if metadata_path else {}
    if isinstance(metadata.get("run_config"), dict):
        config = {**metadata["run_config"], **config}

    grouped: OrderedDict[str, list[tuple[int, dict]]] = OrderedDict()
    for idx, record in enumerate(records):
        key = _prompt_id(record, idx, config)
        grouped.setdefault(key, []).append((idx, record))

    prompts: list[PromptTrajectories] = []
    step_mismatch_count = 0
    limit = None if num_prompts is None else int(num_prompts)
    for prompt_id, rows in grouped.items():
        if limit is not None and len(prompts) >= limit:
            break
        rows = sorted(rows, key=lambda item: _response_sample_id(item[1], item[0], config))
        first_record = rows[0][1]
        samples: list[TrajectorySample] = []
        step_counts: list[int] = []
        for idx, record in rows:
            steps = _step_answers(record, step_view)
            step_counts.append(len(steps))
            samples.append(TrajectorySample(
                sample_id=_to_int(record.get("sample_id"), idx),
                response_sample_id=_response_sample_id(record, idx, config),
                example_id=str(record.get("example_id", record.get("sample_id", idx))),
                final_answer=_text_from_view(record.get("final")),
                step_answers=steps,
            ))
        if len(set(step_counts)) > 1:
            step_mismatch_count += 1
        prompts.append(PromptTrajectories(
            prompt_id=str(prompt_id),
            qa_index=_to_int(_first_present(first_record, ("qa_index",))),
            question=str(first_record.get("question", "")),
            samples=tuple(samples),
            num_steps=min(step_counts) if step_counts else 0,
        ))

    return GenerationRun(
        run_dir=resolved_dir, prompts=tuple(prompts),
        config=config, metadata=metadata,
        diagnostics={
            "examples_path": str(examples_path),
            "num_generated_rows_loaded": len(records),
            "num_prompt_groups": len(grouped),
            "num_prompts_returned": len(prompts),
            "num_step_count_mismatches": step_mismatch_count,
            "step_view": step_view,
        },
    )


def _resolve_paths(run_dir: str | Path) -> tuple[Path, Path | None, Path | None, Path]:
    run_dir = Path(run_dir).expanduser()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    if not run_dir.is_dir():
        raise ValueError(f"Expected a directory, got file: {run_dir}")

    metadata_path = run_dir / "metadata.json"
    metadata = _read_json(metadata_path) if metadata_path.exists() else {}
    files = metadata.get("files", {}) if isinstance(metadata.get("files"), dict) else {}
    examples_path = run_dir / str(files.get("examples", "examples.jsonl"))
    config_path = run_dir / str(files.get("config", "config.json"))

    if not examples_path.exists():
        raise FileNotFoundError(f"Examples file not found: {examples_path}")
    return (
        examples_path,
        config_path if config_path.exists() else None,
        metadata_path if metadata_path.exists() else None,
        run_dir,
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object on line {line_number} of {path}")
            rows.append(row)
    return rows


def _metadata(record: dict) -> dict:
    m = record.get("metadata")
    return m if isinstance(m, dict) else {}


def _first_present(record: dict, keys: tuple[str, ...]) -> Any:
    meta = _metadata(record)
    for key in keys:
        if record.get(key) is not None:
            return record[key]
        if meta.get(key) is not None:
            return meta[key]
    return None


def _to_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _text_from_view(view: Any) -> str:
    if not isinstance(view, dict):
        return ""
    for field in TEXT_FIELDS:
        value = view.get(field)
        if is_valid_text(value):
            return str(value).strip()
    return ""


def _step_view(step: dict, step_view: str) -> dict | None:
    if step_view == "x0":
        if isinstance(step.get("x0"), dict):
            return step["x0"]
        if "x0_string" in step:
            return {"x0_string": step["x0_string"]}
        return step
    if step_view == "x":
        if isinstance(step.get("x"), dict):
            return step["x"]
        if "x_string" in step:
            return {"x_string": step["x_string"]}
        return step
    return step


def _step_answers(record: dict, step_view: str) -> tuple[str, ...]:
    indexed: list[tuple[int, dict]] = []
    for fallback, step in enumerate(record.get("steps") or []):
        if not isinstance(step, dict):
            continue
        idx = _to_int(step.get("step"), fallback)
        indexed.append((int(idx), step))
    indexed.sort(key=lambda item: item[0])
    return tuple(_text_from_view(_step_view(step, step_view)) for _, step in indexed)


def _prompt_id(record: dict, fallback_index: int, config: dict) -> str:
    value = _first_present(record, ("qa_example_id", "qa_index"))
    if value is not None and str(value).strip():
        return str(value)
    num_samples = int(config.get("num_response_samples", 1) or 1)
    if num_samples > 1:
        return str(fallback_index // num_samples)
    return str(record.get("example_id", record.get("sample_id", fallback_index)))


def _response_sample_id(record: dict, fallback_index: int, config: dict) -> int:
    value = _first_present(record, ("response_sample_id",))
    result = _to_int(value)
    if result is not None:
        return result
    num_samples = int(config.get("num_response_samples", 1) or 1)
    return int(fallback_index % num_samples) if num_samples > 1 else 0
