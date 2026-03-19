from __future__ import annotations

import datetime as dt
import os
from typing import Any


def generate_report(
    status: str,
    person_count: int,
    violation_count: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    summary = (
        f"Safety Report: status={status}, people={person_count}, "
        f"violations={violation_count}, timestamp={timestamp}"
    )

    report: dict[str, Any] = {
        "timestamp": timestamp,
        "status": status,
        "person_count": person_count,
        "violation_count": violation_count,
        "summary": summary,
    }

    if extra:
        report["context"] = extra

    ai_summary = maybe_generate_llm_summary(report)
    if ai_summary:
        report["ai_summary"] = ai_summary

    return report


def maybe_generate_llm_summary(report: dict[str, Any]) -> str | None:
    if os.getenv("ENABLE_LLM_REPORTS", "0") != "1":
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        prompt = (
            "Create one concise workplace safety note based on this report. "
            "Use plain English and include one recommended action.\n\n"
            f"Report: {report}"
        )
        response = client.responses.create(
            model=model_name,
            input=prompt,
            max_output_tokens=120,
        )
        text = (response.output_text or "").strip()
        return text or None
    except Exception:
        return None
