import math
import re
from collections import Counter
from pathlib import Path

import pymupdf
import pytest

from tests.perf.harness import (
    Outcome,
    PerfResult,
    _extract_server_metrics,
    _failure_category,
    _percentiles,
    _run_id,
    has_highlights,
    load_expected_strings,
    proposed_redaction_terms,
    redaction_recall,
    select_fixtures,
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


def _pdf_with_highlight(path, content=None):
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Joan Brown")
    if content is not None:
        annotation = page.add_highlight_annot(pymupdf.Rect(72, 60, 130, 76))
        annotation.set_info({"content": content})
        annotation.update()
    pdf.save(path)
    pdf.close()
    return path


def test_select_fixtures_uses_each_fixture_about_equally_in_a_seeded_order():
    paths = [Path("a.pdf"), Path("b.pdf"), Path("c.pdf")]

    selected = select_fixtures(paths, total=7, seed=1)

    assert selected == select_fixtures(paths, total=7, seed=1)
    counts = Counter(selected)
    assert sorted(counts.values()) == [2, 2, 3]


def test_has_highlights(tmp_path):
    assert not has_highlights(_pdf_with_highlight(tmp_path / "plain.pdf"))
    assert has_highlights(_pdf_with_highlight(tmp_path / "marked.pdf", "Joan Brown"))


def test_proposed_redaction_terms_are_the_content_of_the_highlights(tmp_path):
    pdf_path = _pdf_with_highlight(tmp_path / "marked.pdf", "Joan Brown")

    assert proposed_redaction_terms(pdf_path) == ["Joan Brown"]


def test_load_expected_strings(tmp_path):
    fixture = tmp_path / "doc.pdf"
    assert load_expected_strings(fixture) == []

    (tmp_path / "doc.expected.json").write_text('{"redact": ["Joan Brown"]}')

    assert load_expected_strings(fixture) == ["Joan Brown"]


@pytest.mark.parametrize(
    ("expected", "proposed", "recall"),
    [
        (["Joan Brown", "4 High Street"], ["Joan Brown", "4 High Street"], 1.0),
        (["Joan Brown"], ["  joan   BROWN "], 1.0),
        (["Joan Brown"], ["Mrs Joan Brown"], 1.0),
        (["Mr John Smith"], ["John Smith"], 0.0),
        (["Joan Brown", "4 High Street"], ["Joan Brown"], 0.5),
        ([], ["Joan Brown"], None),
    ],
)
def test_redaction_recall(expected, proposed, recall):
    assert redaction_recall(expected, proposed) == recall
