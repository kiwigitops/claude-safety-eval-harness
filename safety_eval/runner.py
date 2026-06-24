"""Dataset loading and evaluation orchestration."""

from __future__ import annotations

import csv
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from safety_eval import database
from safety_eval.evaluator import evaluate_response, summarize_scores
from safety_eval.models import PromptCase, create_model_client


@dataclass(frozen=True)
class RunResult:
    row_id: int
    run_id: str
    prompt_case: PromptCase
    model_name: str
    response: str
    timestamp: str
    scores: dict[str, Any]
    notes: str


def load_dataset(dataset_path: str) -> list[PromptCase]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_json_dataset(path)
    if suffix == ".csv":
        return _load_csv_dataset(path)
    raise ValueError("Dataset must be a .json or .csv file.")


def _load_json_dataset(path: Path) -> list[PromptCase]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("JSON dataset must be a list of prompt objects.")
    return [_prompt_case_from_mapping(item, index) for index, item in enumerate(data, start=1)]


def _load_csv_dataset(path: Path) -> list[PromptCase]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return [
            _prompt_case_from_mapping(row, index)
            for index, row in enumerate(reader, start=1)
        ]


def _prompt_case_from_mapping(item: Any, index: int) -> PromptCase:
    if not isinstance(item, dict):
        raise ValueError(f"Prompt row {index} must be an object or CSV row.")
    prompt = str(item.get("prompt", "")).strip()
    if not prompt:
        raise ValueError(f"Prompt row {index} is missing a prompt.")
    return PromptCase(
        id=_clean_field(item, "id", f"prompt-{index:03d}"),
        category=_clean_field(item, "category", "uncategorized"),
        prompt=prompt,
        expected_behavior=_clean_field(item, "expected_behavior"),
        notes=_clean_field(item, "notes"),
    )


def _clean_field(item: dict[str, Any], key: str, default: str = "") -> str:
    value = item.get(key)
    if value is None:
        return default
    cleaned = str(value).strip()
    return cleaned or default


def run_evaluation(
    *,
    dataset_path: str,
    model_name: str,
    db_path: str = database.DEFAULT_DB_PATH,
    notes: str = "",
) -> list[RunResult]:
    prompt_cases = load_dataset(dataset_path)
    model_client = create_model_client(model_name)
    run_id = str(uuid.uuid4())

    results: list[RunResult] = []
    with database.connect(db_path) as conn:
        database.init_db(conn)
        for prompt_case in prompt_cases:
            response = model_client.generate(prompt_case)
            scores = evaluate_response(prompt_case.prompt, response)
            timestamp = datetime.now(timezone.utc).isoformat()
            combined_notes = " | ".join(
                part
                for part in [prompt_case.notes, notes, summarize_scores(scores)]
                if part
            )
            row_id = database.insert_run(
                conn,
                run_id=run_id,
                prompt_case=prompt_case,
                model_name=model_client.name,
                response=response,
                timestamp=timestamp,
                scores=scores,
                notes=combined_notes,
            )
            results.append(
                RunResult(
                    row_id=row_id,
                    run_id=run_id,
                    prompt_case=prompt_case,
                    model_name=model_client.name,
                    response=response,
                    timestamp=timestamp,
                    scores=scores.to_dict(),
                    notes=combined_notes,
                )
            )
    return results


def summarize_results(results: list[RunResult] | list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        return {
            "total": 0,
            "refusal_count": 0,
            "refusal_rate": 0.0,
            "harmful_compliance_count": 0,
            "jailbreak_count": 0,
            "hallucination_risk_count": 0,
            "uncertainty_count": 0,
            "average_risk_score": 0.0,
        }

    refusal_count = sum(_score_value(row, "refusal_detected") for row in results)
    harmful_count = sum(
        _score_value(row, "harmful_compliance_detected") for row in results
    )
    jailbreak_count = sum(
        _score_value(row, "jailbreak_success_indicator") for row in results
    )
    hallucination_count = sum(
        _score_value(row, "hallucination_risk_language") for row in results
    )
    uncertainty_count = sum(_score_value(row, "uncertainty_language") for row in results)
    risk_sum = sum(_numeric_score(row, "overall_risk_score") for row in results)

    return {
        "total": total,
        "refusal_count": refusal_count,
        "refusal_rate": refusal_count / total,
        "harmful_compliance_count": harmful_count,
        "jailbreak_count": jailbreak_count,
        "hallucination_risk_count": hallucination_count,
        "uncertainty_count": uncertainty_count,
        "average_risk_score": risk_sum / total,
    }


def _score_value(row: RunResult | dict[str, Any], key: str) -> int:
    if isinstance(row, RunResult):
        return int(bool(row.scores.get(key)))
    return int(row.get(key, 0) or 0)


def _numeric_score(row: RunResult | dict[str, Any], key: str) -> float:
    if isinstance(row, RunResult):
        return float(row.scores.get(key, 0))
    return float(row.get(key, 0) or 0)
