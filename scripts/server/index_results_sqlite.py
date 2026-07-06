#!/usr/bin/env python3
"""Index JSONL experiment outputs into SQLite for quick filtering and handoff."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))

from evaluate_results import dataset_accuracy, gold_for_em, prediction_for_em  # noqa: E402


def token_total(row: Dict[str, Any]) -> int:
    api = row.get("api_metrics") or {}
    if api:
        return int(api.get("total_tokens") or 0)
    llm = row.get("llm_metrics") or {}
    return int(llm.get("total_tokens_est") or 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--jsonl", required=True)
    args = parser.parse_args()

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        create table if not exists samples (
          run_name text not null,
          task text not null,
          sample_id text not null,
          correct integer not null,
          prediction text,
          gold text,
          risk_level text,
          route_type text,
          deterministic_shortcut integer,
          total_tokens integer,
          elapsed_seconds real,
          raw_json text not null,
          primary key (run_name, task, sample_id)
        )
        """
    )
    rows = []
    with Path(args.jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(
                (
                    args.run_name,
                    args.task,
                    str(row.get("id") or ""),
                    1 if dataset_accuracy(row) else 0,
                    json.dumps(prediction_for_em(row), ensure_ascii=False),
                    json.dumps(gold_for_em(row), ensure_ascii=False),
                    str(row.get("risk_level") or ""),
                    str(row.get("route_type") or ""),
                    1 if row.get("deterministic_shortcut_applied") else 0,
                    token_total(row),
                    float(row.get("elapsed_seconds_total") or 0.0),
                    json.dumps(row, ensure_ascii=False),
                )
            )
    conn.executemany(
        """
        insert or replace into samples (
          run_name, task, sample_id, correct, prediction, gold, risk_level, route_type,
          deterministic_shortcut, total_tokens, elapsed_seconds, raw_json
        ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    summary = conn.execute(
        """
        select task, count(*) as n, sum(correct) as correct, avg(total_tokens) as avg_tokens,
               avg(elapsed_seconds) as avg_seconds
        from samples
        where run_name = ?
        group by task
        order by task
        """,
        (args.run_name,),
    ).fetchall()
    conn.close()
    print(json.dumps({"indexed": len(rows), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
