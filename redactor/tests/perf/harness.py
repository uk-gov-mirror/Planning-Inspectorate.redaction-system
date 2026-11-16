"""Perf test helpers that need neither Azure nor the e2e utilities."""

import os
import re
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum


def _run_id() -> str:
    return os.getenv("E2E_RUN_ID") or datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")


# ----------------------------
# Stats
# ----------------------------


def _percentiles(values: list[float]) -> dict[str, float]:
    xs = sorted(values)

    def p(q: float) -> float:
        if not xs:
            return float("nan")
        k = (len(xs) - 1) * q
        f = int(k)
        c = min(f + 1, len(xs) - 1)
        if f == c:
            return xs[f]
        return xs[f] + (xs[c] - xs[f]) * (k - f)

    return {
        "min": xs[0] if xs else float("nan"),
        "p50": p(0.50),
        "p95": p(0.95),
        "p99": p(0.99),
        "max": xs[-1] if xs else float("nan"),
        "mean": statistics.mean(xs) if xs else float("nan"),
    }


# ----------------------------
# Outcome + results
# ----------------------------


class Outcome(str, Enum):
    SUCCESS = "success"
    APP_FAIL = "app_fail"  # acceptable
    TEST_FAIL = "test_fail"  # not acceptable


@dataclass(frozen=True)
class PerfResult:
    outcome: Outcome
    runtime_status: str
    seconds: float
    checked_blob_exists: bool
    out_blob_exists: bool | None
    poll_url: str | None
    instance_id: str | None
    durable_status: dict | None
    diagnostics: str | None = None
    error: str | None = None
    downloaded_to: str | None = None

    @property
    def ok(self) -> bool:
        # OK means: not a test-side failure
        return self.outcome != Outcome.TEST_FAIL


def _summarise_durable_status(status: dict | None) -> dict:
    if not status:
        return {}
    keys = [
        "instanceId",
        "runtimeStatus",
        "createdTime",
        "lastUpdatedTime",
        "customStatus",
        "output",
        "errorMessage",
    ]
    return {k: status.get(k) for k in keys if k in status}


def _extract_server_metrics(durable_status: dict | None) -> dict[str, float | int]:
    """
    Timings and the failed chunk count that the app reported in the orchestration
    output, without the redacted content that the run metrics also hold.
    """
    output = (durable_status or {}).get("output")
    if not isinstance(output, dict):
        return {}
    run_metrics = output.get("run_metrics") or {}
    failed_chunks = sum(
        (rule.get("llm_metadata") or {}).get("failed_chunk_count", 0)
        for rule in (run_metrics.get("result_metrics") or {}).values()
        if isinstance(rule, dict)
    )
    return {
        "server_seconds": output.get("execution_time_seconds"),
        # Falls back to the serial total, which is the wall time when rules ran in turn
        "analysis_seconds": run_metrics.get(
            "analysis_wall_time", run_metrics.get("analysis_total_time")
        ),
        "failed_chunks": failed_chunks,
    }


def _failure_category(result: PerfResult) -> str:
    """A short label for why a job failed on the app side, so failures can be counted"""
    error = result.error or ""
    status = result.durable_status or {}
    if result.runtime_status == "StartHttpError":
        code = re.search(r"HTTP (\d{3})", error)
        return f"start_http_{code.group(1)}" if code else "start_http_error"
    if result.runtime_status == "Timeout":
        return "timeout"
    poll_error = re.search(
        r"Durable poll HTTP (\d{3})", status.get("errorMessage") or ""
    )
    if poll_error:
        return f"poll_http_{poll_error.group(1)}"
    if result.runtime_status in ("Failed", "Terminated"):
        return f"orchestration_{result.runtime_status.lower()}"
    output = status.get("output")
    if isinstance(output, dict) and output.get("status") == "FAIL":
        return "app_reported_fail"
    if result.checked_blob_exists and result.out_blob_exists is not True:
        return "output_blob_missing"
    return "other"
