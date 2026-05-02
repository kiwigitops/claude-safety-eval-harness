"""SQLite persistence for evaluation runs."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from safety_eval.evaluator import EvaluationScores
from safety_eval.models import PromptCase


DEFAULT_DB_PATH = "eval_runs.sqlite3"


SCHEMA = """
CREATE TABLE IF NOT EXISTS eval_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    prompt_id TEXT,
    category TEXT,
    prompt TEXT NOT NULL,
    model_name TEXT NOT NULL,
    response TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    refusal_detected INTEGER NOT NULL,
    harmful_compliance_detected INTEGER NOT NULL,
    jailbreak_success_indicator INTEGER NOT NULL,
    hallucination_risk_language INTEGER NOT NULL,
    uncertainty_language INTEGER NOT NULL,
    overall_risk_score INTEGER NOT NULL,
    matched_terms_json TEXT NOT NULL,
    scores_json TEXT NOT NULL,
    notes TEXT
);
"""


CSV_COLUMNS = [
    "id",
    "run_id",
    "prompt_id",
    "category",
    "prompt",
    "model_name",
    "response",
    "timestamp",
    "refusal_detected",
    "harmful_compliance_detected",
    "jailbreak_success_indicator",
    "hallucination_risk_language",
    "uncertainty_language",
    "overall_risk_score",
    "notes",
]


def connect(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(SCHEMA)
    conn.commit()


def insert_run(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    prompt_case: PromptCase,
    model_name: str,
    response: str,
    timestamp: str,
    scores: EvaluationScores,
    notes: str = "",
) -> int:
    scores_dict = scores.to_dict()
    cursor = conn.execute(
        """
        INSERT INTO eval_runs (
            run_id,
            prompt_id,
            category,
            prompt,
            model_name,
            response,
            timestamp,
            refusal_detected,
            harmful_compliance_detected,
            jailbreak_success_indicator,
            hallucination_risk_language,
            uncertainty_language,
            overall_risk_score,
            matched_terms_json,
            scores_json,
            notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            prompt_case.id,
            prompt_case.category,
            prompt_case.prompt,
            model_name,
            response,
            timestamp,
            int(scores.refusal_detected),
            int(scores.harmful_compliance_detected),
            int(scores.jailbreak_success_indicator),
            int(scores.hallucination_risk_language),
            int(scores.uncertainty_language),
            scores.overall_risk_score,
            json.dumps(scores.matched_terms, sort_keys=True),
            json.dumps(scores_dict, sort_keys=True),
            notes,
        ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def fetch_runs(db_path: str = DEFAULT_DB_PATH, run_id: str | None = None) -> list[dict[str, Any]]:
    with connect(db_path) as conn:
        init_db(conn)
        if run_id:
            rows = conn.execute(
                "SELECT * FROM eval_runs WHERE run_id = ? ORDER BY id ASC", (run_id,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM eval_runs ORDER BY id ASC").fetchall()
    return [dict(row) for row in rows]


def latest_run_id(db_path: str = DEFAULT_DB_PATH) -> str | None:
    with connect(db_path) as conn:
        init_db(conn)
        row = conn.execute(
            "SELECT run_id FROM eval_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return str(row["run_id"]) if row else None


def export_runs_csv(rows: Iterable[dict[str, Any]], output_path: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})
    return path

