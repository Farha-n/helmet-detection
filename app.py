from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request

from reporting import generate_report

REPORT_LOG_FILE = Path(os.getenv("REPORT_LOG_FILE", "reports/safety_log.jsonl"))

app = Flask(__name__)


def read_latest_report(file_path: Path) -> dict[str, Any] | None:
    if not file_path.exists():
        return None

    try:
        with file_path.open("r", encoding="utf-8") as fh:
            lines = [line.strip() for line in fh if line.strip()]
    except OSError:
        return None

    if not lines:
        return None

    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return None


@app.get("/")
def home() -> Any:
    return jsonify(
        {
            "service": "AI Helmet Detection System",
            "status": "running",
            "endpoints": ["/health", "/latest-report", "/generate-report"],
        }
    )


@app.get("/health")
def health() -> Any:
    return jsonify({"status": "ok"})


@app.get("/latest-report")
def latest_report() -> Any:
    report = read_latest_report(REPORT_LOG_FILE)
    if report is None:
        return jsonify({"message": "No reports found yet"}), 404
    return jsonify(report)


@app.post("/generate-report")
def create_report() -> Any:
    payload = request.get_json(silent=True) or {}

    status = str(payload.get("status", "SAFE"))
    person_count = int(payload.get("person_count", 0))
    violation_count = int(payload.get("violation_count", 0))
    extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else None

    report = generate_report(
        status=status,
        person_count=person_count,
        violation_count=violation_count,
        extra=extra,
    )

    REPORT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(report) + "\n")

    return jsonify(report), 201


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "1") == "1",
    )
