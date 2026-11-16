import re

from tests.perf.harness import _extract_server_metrics, _run_id


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
