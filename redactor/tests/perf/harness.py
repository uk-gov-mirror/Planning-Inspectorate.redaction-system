"""Perf test helpers that need neither Azure nor the e2e utilities."""

import os
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
        "min": xs[0],
        "p50": p(0.50),
        "p95": p(0.95),
        "p99": p(0.99),
        "max": xs[-1],
        "mean": statistics.mean(xs),
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
