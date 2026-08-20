import importlib
import inspect
import logging
import os
import sys
import time
from uuid import uuid4

import pytest
from filelock import FileLock

from test.util.test_case import TestCase


def quiet_azure_noise_early():
    """
    Must run BEFORE importing app/test modules that might initialise Azure Monitor / OTel.
    """
    want_http = os.getenv("E2E_AZURE_HTTP_LOGGING", "").lower() in (
        "1",
        "true",
        "yes",
        "y",
    )

    if not want_http:
        # Common noisy Azure loggers
        logging.getLogger("azure").setLevel(logging.WARNING)
        logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
            logging.WARNING
        )
        logging.getLogger("azure.identity").setLevel(logging.WARNING)
        logging.getLogger("azure.monitor").setLevel(logging.WARNING)
        logging.getLogger("opentelemetry").setLevel(logging.WARNING)

    if os.getenv("DISABLE_TELEMETRY", "").lower() in ("1", "true", "yes"):
        os.environ.setdefault("AZURE_MONITOR_OPENTELEMETRY_ENABLED", "false")
        os.environ.setdefault("OTEL_TRACES_EXPORTER", "none")
        os.environ.setdefault("OTEL_METRICS_EXPORTER", "none")
        os.environ.setdefault("OTEL_LOGS_EXPORTER", "none")


_CONFIGURED = False
logger = logging.getLogger(__name__)


def configure_session(eager_import: bool = True):
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True
    quiet_azure_noise_early()  # <-- IMPORTANT: before any imports
    if "RUN_ID" not in os.environ:
        run_id = str(uuid4())[:8]
        os.environ["RUN_ID"] = str(run_id)
    logger.info(f"Running with run_id='{os.environ['RUN_ID']}'")
    if eager_import:
        import_all_testing_modules()


def import_all_testing_modules():
    """
    Import all test modules
    """
    # Extract all testing modules under the `test/` directory
    module_content_to_exclude = {"__init__.py", "__pycache__", "conftest.py"}
    python_modules_to_load = []
    test_dir = os.path.dirname(__file__)
    files_to_explore = [
        x
        for x in os.listdir(test_dir)
        if "test" in x and os.path.isdir(os.path.join(test_dir, x))
    ]
    while files_to_explore:
        file = files_to_explore.pop(0)
        current_path = os.path.join(test_dir, file)
        if os.path.isfile(current_path):
            if file.endswith(".py") and all(
                x not in file for x in module_content_to_exclude
            ):
                python_modules_to_load.append(file)
        else:
            files_to_explore.extend(
                [os.path.join(current_path, x) for x in os.listdir(current_path)]
            )
    python_modules_to_load_cleaned = sorted(
        [x.replace(".py", "").replace(os.sep, ".") for x in python_modules_to_load]
    )
    python_modules_to_load_cleaned = [
        f"test.{x}" if not x.startswith("test") else x
        for x in python_modules_to_load_cleaned
    ]
    # Import all testing modules
    for module_to_import in python_modules_to_load_cleaned:
        importlib.import_module(module_to_import, package="test")
    return python_modules_to_load_cleaned


def extract_all_test_cases():
    """
    Iterate through the imported modules to select all TestCase instances
    """
    test_modules = [module for module in sys.modules if module.startswith("test.")]
    return {
        obj
        for module in test_modules
        for name, obj in inspect.getmembers(sys.modules[module])
        if inspect.isclass(obj) and issubclass(obj, TestCase) and obj != TestCase
    }


def process_arguments(session) -> list[type[TestCase]]:
    """
    Process the pytest invocation parameters to return a list of test cases whos
    module setup/teardown functions need to be called
    """
    pytest_args = session.config.invocation_params.args

    # If we're doing a marker-only run (e.g. -m e2e), do NOT run the unit/integration
    # TestCase session_setup/teardown for the whole suite.
    if "-m" in pytest_args:
        m_idx = pytest_args.index("-m")
        if m_idx + 1 < len(pytest_args) and "e2e" in pytest_args[m_idx + 1]:
            return []

    test_cases = extract_all_test_cases()
    directory_args = [x.replace(".py", "") for x in pytest_args if x.startswith("test")]
    # If no arguments were given or if no specific python files were given then
    # all tests are being executed
    if not directory_args:
        return test_cases
    test_case_module_map = {
        test_case.__module__.replace(".", "/"): test_case for test_case in test_cases
    }
    matched_modules = []
    for directory in directory_args:
        matches = [module for module in test_case_module_map if directory in module]
        matched_modules += matches
    return [test_case_module_map[directory] for directory in set(matched_modules)]


def _session_setup_task(session):
    logger.info("Setting up pytest session for tests")
    # Test-specific resources
    for test_case in process_arguments(session):
        logger.info("    Running setup for " + test_case.__module__)
        test_case().session_setup()


@pytest.fixture(scope="session", autouse=True)
def session_setup(tmp_path_factory, worker_id, request):
    # Code based on example from docs at
    # https://pytest-xdist.readthedocs.io/en/latest/how-to.html#making-session-scoped-fixtures-execute-only-once
    if worker_id == "master":
        return _session_setup_task(request.session)
    root_tmp_dir = tmp_path_factory.getbasetemp().parent
    fn = root_tmp_dir / "setupsession.json"
    with FileLock(str(fn) + ".lock"):
        if fn.is_file():
            master_worker_written_value = fn.read_text()
            if "Failed" in master_worker_written_value:
                pytest.fail("Master thread failed during setup")
        else:
            fn.write_text("Starting session setup")
            try:
                _session_setup_task(request.session)
                fn.write_text("Complete")
            except KeyboardInterrupt:
                sys.exit()
            except Exception:
                fn.write_text("Failed")
                raise


def _session_teardown_task(session):
    logger.info("Tearing down pytest session for tests")
    for test_case in process_arguments(session):
        logger.info("    Running teardown for " + test_case.__module__)
        t0 = time.perf_counter()
        test_case().session_teardown()
        logger.info(
            "    Teardown complete for %s (%.2fs)",
            test_case.__module__,
            time.perf_counter() - t0,
        )

    # Best-effort: stop OTel providers if they exist
    try:
        from opentelemetry import metrics, trace

        tp = trace.get_tracer_provider()
        mp = metrics.get_meter_provider()

        shutdown = getattr(tp, "shutdown", None)
        if callable(shutdown):
            shutdown()

        shutdown = getattr(mp, "shutdown", None)
        if callable(shutdown):
            shutdown()
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to shutdown OTel providers: {e}")


@pytest.fixture(scope="session", autouse=True)
def session_teardown(tmp_path_factory, worker_id, request):
    # Code based on example from docs at
    # https://pytest-xdist.readthedocs.io/en/latest/how-to.html#making-session-scoped-fixtures-execute-only-once
    yield
    if worker_id == "master":
        return _session_teardown_task(request.session)
    root_tmp_dir = tmp_path_factory.getbasetemp().parent
    fn = root_tmp_dir / "teardownsession.json"
    with FileLock(str(fn) + ".lock"):
        if fn.is_file():
            number_of_completed_workers = int(fn.read_text()) + 1
            fn.write_text(str(number_of_completed_workers))
        else:
            number_of_completed_workers = 1
            fn.write_text(str(number_of_completed_workers))
    # Call teardown if this call was initiated by the last worker that is still running
    num_tests = len(request.session.items)
    if num_tests < request.config.workerinput["workercount"]:
        last_worker = number_of_completed_workers >= num_tests
    else:
        last_worker = (
            number_of_completed_workers >= request.config.workerinput["workercount"]
        )
    logger.info("Num tests: " + str(num_tests))
    logger.info("completed workers count: " + str(number_of_completed_workers))
    logger.info("Last worker: " + str(last_worker))
    if last_worker:
        _session_teardown_task(request.session)
