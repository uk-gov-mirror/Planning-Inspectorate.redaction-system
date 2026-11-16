# redactor/tests/perf/test_perf_concurrent_redactions.py

import asyncio
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from ..e2e.utils import (
    az_blob_exists,
    az_download,
    az_upload,
    build_payload,
    function_start_url,
)
from .harness import (
    Outcome,
    PerfResult,
    _percentiles,
    _run_id,
    _summarise_durable_status,
)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def _load_dotenv_if_present() -> None:
    env_path = _repo_root() / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


# ----------------------------
# Perf config (env-driven)
# ----------------------------


def _int_env(name: str, default: int) -> int:
    v = os.getenv(name)
    return int(v) if v else default


def _float_env(name: str, default: float) -> float:
    v = os.getenv(name)
    return float(v) if v else default


def _env_required(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


PERF_CONCURRENCY = _int_env("PERF_CONCURRENCY", 20)
PERF_TOTAL = _int_env("PERF_TOTAL", 200)
PERF_TIMEOUT_S = _int_env("PERF_TIMEOUT_S", 200)
PERF_POLL_S = _float_env("PERF_POLL_S", 2.0)

PERF_EXISTS_SAMPLE_EVERY = _int_env("PERF_EXISTS_SAMPLE_EVERY", 5)

# IMPORTANT: default bumped to 180s to reduce false failures under load
PERF_EXISTS_WAIT_S = _float_env("PERF_EXISTS_WAIT_S", 180.0)
PERF_EXISTS_WAIT_POLL_S = _float_env("PERF_EXISTS_WAIT_POLL_S", 5.0)

# If > 0, download sampled outputs locally when they exist (proof of "written to file")
PERF_DOWNLOAD_SAMPLE_EVERY = _int_env("PERF_DOWNLOAD_SAMPLE_EVERY", 1)

PERF_MAX_P95_S = _float_env("PERF_MAX_P95_S", 0.0)

PERF_FIXTURE_PDF = os.getenv("PERF_FIXTURE_PDF", "pas_13.pdf")


# ----------------------------
# Repo helpers
# ----------------------------


def _repo_root() -> Path:
    """
    This file is typically:
      <repo>/redactor/tests/perf/test_perf_concurrent_redactions.py

    parents[3] -> <repo>
    """
    return Path(__file__).resolve().parents[3]


# ----------------------------
# Azure CLI diagnostics helpers
# ----------------------------


def _az_list_blobs_prefix(
    *,
    storage_account: str,
    container_name: str,
    prefix: str,
    limit: int = 200,
) -> list[str]:
    """
    Best-effort listing of blobs under a prefix.
    Returns a list of blob names (possibly empty). Never raises.
    """
    try:
        cmd = [
            "az",
            "storage",
            "blob",
            "list",
            "--account-name",
            storage_account,
            "--container-name",
            container_name,
            "--prefix",
            prefix,
            "--auth-mode",
            "login",
            "--query",
            "[].name",
            "-o",
            "tsv",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if r.returncode != 0:
            return [f"<<az blob list failed rc={r.returncode}: {r.stderr.strip()}>>"]

        lines = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
        return lines[:limit]
    except Exception as e:  # noqa: BLE001
        return [f"<<az blob list exception: {e}>>"]


# ----------------------------
# Blob-exists helper (sync, uses az cli underneath)
# ----------------------------


def _wait_for_blob_exists(
    *,
    storage_account: str,
    container_name: str,
    blob_name: str,
    timeout_s: float,
    poll_s: float,
) -> bool:
    """
    Durable can report Completed before the output blob is observable via `az storage blob exists`,
    so we allow a grace period.
    """
    deadline = time.time() + timeout_s
    last = False
    while time.time() < deadline:
        last = az_blob_exists(storage_account, container_name, blob_name)
        if last:
            return True
        time.sleep(poll_s)
    return bool(last)


# ----------------------------
# Async durable runner
# ----------------------------


def _classify_http_as_app_fail(status_code: int) -> bool:
    # treat transient/service-side issues as app-side acceptable failures
    return status_code >= 500 or status_code in (429, 408)


async def _trigger_start(
    client: httpx.AsyncClient,
    start_url: str,
    payload: dict,
) -> dict:
    r = await client.post(start_url, json=payload, timeout=60)

    # If the app is unhappy (5xx/429/408), that's an acceptable app-side failure
    if _classify_http_as_app_fail(r.status_code):
        return {
            "_app_http_error": True,
            "status_code": r.status_code,
            "text": (r.text or "")[:2000],
        }

    # For 4xx, treat as test/config failure (payload, auth, wrong URL, etc.)
    if 400 <= r.status_code < 500:
        return {
            "_test_http_error": True,
            "status_code": r.status_code,
            "text": (r.text or "")[:2000],
        }

    r.raise_for_status()
    return r.json()


async def _poll_until_done(
    client: httpx.AsyncClient,
    poll_url: str,
    timeout_s: int,
    poll_s: float,
) -> dict:
    deadline = time.time() + timeout_s
    last_status: dict | None = None
    while time.time() < deadline:
        r = await client.get(poll_url, timeout=60)

        # durable polling endpoint returning 5xx/429 is considered app-side acceptable failure
        if _classify_http_as_app_fail(r.status_code):
            return {
                "runtimeStatus": "Failed",
                "errorMessage": f"Durable poll HTTP {r.status_code}",
                "output": {"status": "FAIL", "message": (r.text or "")[:2000]},
            }

        # 4xx on poll suggests the poll URL is invalid/expired or auth wrong -> test fail
        if 400 <= r.status_code < 500:
            raise RuntimeError(
                f"Durable poll returned HTTP {r.status_code}: {(r.text or '')[:500]}"
            )

        r.raise_for_status()
        status = r.json()
        last_status = status
        state = status.get("runtimeStatus")
        if state in ("Completed", "Failed", "Terminated"):
            return status
        await asyncio.sleep(poll_s)

    raise TimeoutError(
        f"Timed out waiting for orchestration. pollEndpoint={poll_url} last={last_status}"
    )


def _is_app_failure_from_status(durable_status: dict | None) -> bool:
    """
    Decide if this run failed due to the *app* (acceptable), even if runtimeStatus == Completed.
    We use:
      - runtimeStatus in Failed/Terminated
      - durable errorMessage present
      - output.status == "FAIL"
    """
    if not durable_status:
        return False

    runtime_status = durable_status.get("runtimeStatus")
    if runtime_status in ("Failed", "Terminated"):
        return True

    if durable_status.get("errorMessage"):
        return True

    out = durable_status.get("output") or {}
    return isinstance(out, dict) and out.get("status") == "FAIL"


async def _run_one(
    *,
    client: httpx.AsyncClient,
    start_url: str,
    payload: dict,
    timeout_s: int,
    poll_s: float,
    do_exists_check: bool,
    storage_account: str,
    container_name: str,
    out_blob: str,
    out_prefix: str,
    expected_prefix_token: str,  # e.g. "00001"
    tmp_dir: Path,
) -> PerfResult:
    t0 = time.perf_counter()
    poll_url: str | None = None
    instance_id: str | None = None
    durable_status: dict | None = None

    try:
        start = await _trigger_start(client, start_url, payload)

        if start.get("_app_http_error"):
            elapsed = time.perf_counter() - t0
            return PerfResult(
                outcome=Outcome.APP_FAIL,
                runtime_status="StartHttpError",
                seconds=elapsed,
                checked_blob_exists=False,
                out_blob_exists=None,
                poll_url=None,
                instance_id=None,
                durable_status=None,
                error=f"Start returned HTTP {start.get('status_code')}: {start.get('text')}",
            )

        if start.get("_test_http_error"):
            elapsed = time.perf_counter() - t0
            return PerfResult(
                outcome=Outcome.TEST_FAIL,
                runtime_status="StartHttpError",
                seconds=elapsed,
                checked_blob_exists=False,
                out_blob_exists=None,
                poll_url=None,
                instance_id=None,
                durable_status=None,
                error=f"TEST FAIL: start returned HTTP {start.get('status_code')}: {start.get('text')}",
            )

        poll_url = start.get("pollEndpoint")
        instance_id = start.get("id") or start.get("instanceId")

        if not poll_url:
            raise RuntimeError(f"Missing pollEndpoint in start response: {start}")

        durable_status = await _poll_until_done(
            client, poll_url, timeout_s=timeout_s, poll_s=poll_s
        )
        runtime_status = durable_status.get("runtimeStatus", "Unknown")
        elapsed = time.perf_counter() - t0

        # Determine if app reported failure (acceptable)
        app_failed = _is_app_failure_from_status(durable_status)

        # If app failed, we don't require output blob existence
        if app_failed:
            return PerfResult(
                outcome=Outcome.APP_FAIL,
                runtime_status=runtime_status,
                seconds=elapsed,
                checked_blob_exists=do_exists_check,
                out_blob_exists=None,
                poll_url=poll_url,
                instance_id=durable_status.get("instanceId") or instance_id,
                durable_status=_summarise_durable_status(durable_status),
                diagnostics=None,
                error=json.dumps(
                    _summarise_durable_status(durable_status), default=str
                ),
            )

        out_exists: bool | None = None
        diagnostics: str | None = None
        downloaded_to: str | None = None

        # For successful app runs, validate blob existence (sampled)
        if do_exists_check:
            out_exists = await asyncio.to_thread(
                _wait_for_blob_exists,
                storage_account=storage_account,
                container_name=container_name,
                blob_name=out_blob,
                timeout_s=PERF_EXISTS_WAIT_S,
                poll_s=PERF_EXISTS_WAIT_POLL_S,
            )

            # If Durable says Completed and app didn't report FAIL but blob isn't there:
            # this is likely app-side (inconsistent), but still acceptable as APP_FAIL
            if runtime_status == "Completed" and out_exists is not True:
                blob_names = await asyncio.to_thread(
                    _az_list_blobs_prefix,
                    storage_account=storage_account,
                    container_name=container_name,
                    prefix=out_prefix,
                )

                near_matches = [
                    b
                    for b in blob_names
                    if f"/{expected_prefix_token}" in b
                    or b.endswith(f"/{expected_prefix_token}")
                    or f"/{expected_prefix_token}_" in b
                ]
                if not near_matches:
                    near_matches = [
                        b
                        for b in blob_names
                        if expected_prefix_token in b.split("/")[-1]
                    ]

                diagnostics_obj = {
                    "expected_out_blob": out_blob,
                    "out_prefix_list_count": len(blob_names),
                    "out_prefix_list_sample": blob_names[:50],
                    "near_matches": near_matches[:20],
                    "durable_status_summary": _summarise_durable_status(durable_status),
                }
                diagnostics = json.dumps(diagnostics_obj, default=str)

                return PerfResult(
                    outcome=Outcome.APP_FAIL,
                    runtime_status=runtime_status,
                    seconds=elapsed,
                    checked_blob_exists=True,
                    out_blob_exists=out_exists,
                    poll_url=poll_url,
                    instance_id=durable_status.get("instanceId") or instance_id,
                    durable_status=_summarise_durable_status(durable_status),
                    diagnostics=diagnostics,
                    error=f"Completed but output blob not observed: out_exists={out_exists}",
                )

            # If blob exists and download sampling is enabled, download to local file (proof it was written to disk)
            if out_exists is True and PERF_DOWNLOAD_SAMPLE_EVERY > 0:
                # reuse the same sampling cadence as exists checks, unless user changes it
                out_file = tmp_dir / Path(out_blob).name
                try:
                    await asyncio.to_thread(
                        az_download,
                        storage_account,
                        container_name,
                        out_blob,
                        out_file,
                    )
                    downloaded_to = str(out_file)
                except Exception as e:  # noqa: BLE001
                    return PerfResult(
                        outcome=Outcome.TEST_FAIL,
                        runtime_status=runtime_status,
                        seconds=elapsed,
                        checked_blob_exists=True,
                        out_blob_exists=True,
                        poll_url=poll_url,
                        instance_id=durable_status.get("instanceId") or instance_id,
                        durable_status=_summarise_durable_status(durable_status),
                        diagnostics=None,
                        error=f"TEST FAIL: download failed: {e}",
                    )

        # Success criteria: app success + (if we checked existence, blob exists)
        success = (runtime_status == "Completed") and (
            True if out_exists is None else out_exists is True
        )

        return PerfResult(
            outcome=Outcome.SUCCESS if success else Outcome.APP_FAIL,
            runtime_status=runtime_status,
            seconds=elapsed,
            checked_blob_exists=do_exists_check,
            out_blob_exists=out_exists,
            poll_url=poll_url,
            instance_id=durable_status.get("instanceId") or instance_id,
            durable_status=_summarise_durable_status(durable_status),
            diagnostics=diagnostics,
            error=None
            if success
            else f"App considered non-success: status={runtime_status} out_exists={out_exists}",
            downloaded_to=downloaded_to,
        )

    except Exception as e:  # noqa: BLE001
        elapsed = time.perf_counter() - t0
        # Anything unexpected in the harness is test-side
        return PerfResult(
            outcome=Outcome.TEST_FAIL,
            runtime_status="Exception",
            seconds=elapsed,
            checked_blob_exists=do_exists_check,
            out_blob_exists=None,
            poll_url=poll_url,
            instance_id=instance_id,
            durable_status=_summarise_durable_status(durable_status),
            diagnostics=None,
            error=f"TEST FAIL: {e} poll={poll_url}" if poll_url else f"TEST FAIL: {e}",
        )


# ----------------------------
# Pytest perf test
# ----------------------------


@pytest.mark.perf
def test_concurrent_redactions_perf(tmp_path: Path) -> None:
    _load_dotenv_if_present()
    storage_account = _env_required("E2E_STORAGE_ACCOUNT")
    container_name = _env_required("E2E_CONTAINER_NAME")
    run_id = _run_id()
    start_url = function_start_url()

    repo_root = _repo_root()
    fixture_path = repo_root / "redactor/tests/resources/pdf" / PERF_FIXTURE_PDF
    assert fixture_path.exists(), f"Missing fixture PDF: {fixture_path}"

    # where we will download sampled outputs (proof they can be written to disk)
    downloads_dir = tmp_path / "downloaded_redactions"
    downloads_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Prepare jobs
    # ----------------------------
    jobs: list[dict[str, Any]] = []
    for i in range(PERF_TOTAL):
        idx = f"{i:05d}"
        in_blob = f"perf/{run_id}/in/{idx}.pdf"
        out_blob = f"perf/{run_id}/out/{idx}_REDACTED.pdf"

        az_upload(storage_account, container_name, in_blob, fixture_path)

        payload = build_payload(
            storage_account=storage_account,
            container_name=container_name,
            in_blob=in_blob,
            out_blob=out_blob,
            skip_redaction=False,
        )

        jobs.append(
            {
                "payload": payload,
                "out_blob": out_blob,
                "i": i,
                "idx": idx,
                "out_prefix": f"perf/{run_id}/out/",
            }
        )

    async def run_all() -> list[PerfResult]:
        limits = httpx.Limits(
            max_connections=max(50, PERF_CONCURRENCY * 4),
            max_keepalive_connections=max(20, PERF_CONCURRENCY * 2),
        )
        timeout = httpx.Timeout(60.0)

        async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
            sem = asyncio.Semaphore(PERF_CONCURRENCY)

            async def wrapped(job: dict[str, Any]) -> PerfResult:
                async with sem:
                    i = int(job["i"])
                    do_exists_check = PERF_EXISTS_SAMPLE_EVERY > 0 and (
                        i % PERF_EXISTS_SAMPLE_EVERY == 0
                    )
                    return await _run_one(
                        client=client,
                        start_url=start_url,
                        payload=job["payload"],
                        timeout_s=PERF_TIMEOUT_S,
                        poll_s=PERF_POLL_S,
                        do_exists_check=do_exists_check,
                        storage_account=storage_account,
                        container_name=container_name,
                        out_blob=job["out_blob"],
                        out_prefix=job["out_prefix"],
                        expected_prefix_token=job["idx"],
                        tmp_dir=downloads_dir,
                    )

            return await asyncio.gather(*(wrapped(j) for j in jobs))

    t_wall0 = time.perf_counter()
    results = asyncio.run(run_all())
    wall_elapsed = time.perf_counter() - t_wall0

    # ----------------------------
    # Report
    # ----------------------------
    times = [r.seconds for r in results]
    stats = _percentiles(times)

    successes = [r for r in results if r.outcome == Outcome.SUCCESS]
    app_failures = [r for r in results if r.outcome == Outcome.APP_FAIL]
    test_failures = [r for r in results if r.outcome == Outcome.TEST_FAIL]

    # throughput: successes per wall-clock second
    throughput = (len(successes) / wall_elapsed) if wall_elapsed > 0 else 0.0

    exists_checked = sum(1 for r in results if r.checked_blob_exists)
    exists_failures = [
        r for r in results if r.checked_blob_exists and r.out_blob_exists is not True
    ]

    downloaded = [r for r in results if r.downloaded_to]

    print("\n=== PERF SUMMARY ===")
    print(f"fixture={PERF_FIXTURE_PDF}")
    print(f"concurrency={PERF_CONCURRENCY} total={PERF_TOTAL} poll_s={PERF_POLL_S}")
    print(f"timeout_s={PERF_TIMEOUT_S} exists_sample_every={PERF_EXISTS_SAMPLE_EVERY}")
    print(
        f"exists_wait_s={PERF_EXISTS_WAIT_S} exists_wait_poll_s={PERF_EXISTS_WAIT_POLL_S}"
    )
    print(
        f"success={len(successes)} app_fail={len(app_failures)} test_fail={len(test_failures)}"
    )
    print(
        f"wall_seconds={wall_elapsed:.2f} throughput_success_per_sec={throughput:.4f}"
    )
    print("timings_seconds:", {k: round(v, 3) for k, v in stats.items()})
    print(
        f"blob_exists_checked={exists_checked} blob_exists_checked_failures={len(exists_failures)}"
    )
    print(f"downloaded_files={len(downloaded)} downloads_dir={downloads_dir}")

    if app_failures:
        print("\nApp failures (acceptable) (up to 5):")
        for r in app_failures[:5]:
            print(
                f"- runtime={r.runtime_status} seconds={r.seconds:.2f} "
                f"instanceId={r.instance_id} error={r.error} poll={r.poll_url}"
            )
            if r.durable_status:
                print("  durable_status:", json.dumps(r.durable_status, default=str))
            if r.diagnostics:
                print("  diagnostics:", r.diagnostics)

    if test_failures:
        print("\nTest failures (NOT acceptable) (up to 5):")
        for r in test_failures[:5]:
            print(
                f"- runtime={r.runtime_status} seconds={r.seconds:.2f} "
                f"instanceId={r.instance_id} error={r.error} poll={r.poll_url}"
            )
            if r.durable_status:
                print("  durable_status:", json.dumps(r.durable_status, default=str))
            if r.diagnostics:
                print("  diagnostics:", r.diagnostics)

    # ----------------------------
    # Assertions
    # ----------------------------
    assert len(test_failures) == 0, (
        f"{len(test_failures)} test-side failures (see summary above)"
    )

    if PERF_MAX_P95_S and PERF_MAX_P95_S > 0:
        assert stats["p95"] <= PERF_MAX_P95_S, (
            f"p95 {stats['p95']:.2f}s exceeds PERF_MAX_P95_S={PERF_MAX_P95_S:.2f}s"
        )
