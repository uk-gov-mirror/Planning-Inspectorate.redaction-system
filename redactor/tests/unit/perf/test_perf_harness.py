import re

from tests.perf.harness import _run_id


def test_run_id_defaults_to_a_timestamp(monkeypatch):
    monkeypatch.delenv("E2E_RUN_ID", raising=False)

    assert re.fullmatch(r"\d{8}-\d{6}", _run_id())
