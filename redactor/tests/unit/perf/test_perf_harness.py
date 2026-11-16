import math
import re

import pytest

from tests.perf.harness import (
    Outcome,
    PerfResult,
    _extract_server_metrics,
    _failure_category,
    _percentiles,
    _run_id,
)


def test_run_id_defaults_to_a_timestamp(monkeypatch):
    monkeypatch.delenv("E2E_RUN_ID", raising=False)

    assert re.fullmatch(r"\d{8}-\d{6}", _run_id())


def _durable_status(**run_metrics):
    return {
        "output": {"execution_time_seconds": 30.0, "run_metrics": run_metrics},
    }


def test_extract_server_metrics_reads_timings_and_failed_chunks():
    status = _durable_status(
        analysis_wall_time=20.0,
        analysis_total_time=50.0,
        result_metrics={
            "names": {"llm_metadata": {"failed_chunk_count": 1}},
            "contact": {"llm_metadata": {"failed_chunk_count": 2}},
            "images": {"total_images_to_analyse": 3},
        },
    )

    assert _extract_server_metrics(status) == {
        "server_seconds": 30.0,
        "analysis_seconds": 20.0,
        "failed_chunks": 3,
    }


def test_extract_server_metrics_falls_back_to_serial_analysis_time():
    status = _durable_status(analysis_total_time=50.0)

    assert _extract_server_metrics(status)["analysis_seconds"] == 50.0


def test_extract_server_metrics_without_output_is_empty():
    assert _extract_server_metrics(None) == {}
    assert _extract_server_metrics({"output": "not a dict"}) == {}


def _failed_result(runtime_status="Completed", error=None, **kwargs):
    return PerfResult(
        outcome=Outcome.APP_FAIL,
        runtime_status=runtime_status,
        seconds=1.0,
        checked_blob_exists=kwargs.pop("checked_blob_exists", False),
        out_blob_exists=kwargs.pop("out_blob_exists", None),
        poll_url=None,
        instance_id=None,
        durable_status=kwargs.pop("durable_status", None),
        error=error,
    )


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (
            _failed_result("StartHttpError", error="Start returned HTTP 429: busy"),
            "start_http_429",
        ),
        (_failed_result("Timeout", error="Timed out"), "timeout"),
        (
            _failed_result(
                "Failed", durable_status={"errorMessage": "Durable poll HTTP 503"}
            ),
            "poll_http_503",
        ),
        (_failed_result("Failed"), "orchestration_failed"),
        (
            _failed_result(durable_status={"output": {"status": "FAIL"}}),
            "app_reported_fail",
        ),
        (
            _failed_result(checked_blob_exists=True, out_blob_exists=False),
            "output_blob_missing",
        ),
        (_failed_result(), "other"),
    ],
)
def test_failure_category(result, expected):
    assert _failure_category(result) == expected


def test_percentiles_of_no_values_are_not_a_number():
    stats = _percentiles([])

    assert all(math.isnan(value) for value in stats.values())
