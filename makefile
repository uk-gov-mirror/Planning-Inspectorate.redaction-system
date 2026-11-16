FUNC_RECEIVER_PORT ?= 7071
FUNC_RECEIVER_BASE_URL ?= http://localhost:$(FUNC_RECEIVER_PORT)
FUNC_PROCESSOR_PORT ?= 7072
FUNC_PROCESSOR_BASE_URL ?= http://localhost:$(FUNC_PROCESSOR_PORT)

PERF_TOTAL ?= 5
PERF_CONCURRENCY ?= 5
PERF_TIMEOUT_S ?= 7200

run:
	@echo "Starting Azurite..."
	azurite > /tmp/azurite.log 2>&1 &
	@echo "Starting Azure Functions receiver on port $(FUNC_RECEIVER_PORT)..."
	cd receiver && func start --port $(FUNC_RECEIVER_PORT) > /tmp/func_receiver.log 2>&1 &
	@echo "Starting Azure Functions processor on port $(FUNC_PROCESSOR_PORT)..."
	cd redactor && func start --port $(FUNC_PROCESSOR_PORT) > /tmp/func_processor.log 2>&1 &
	@echo "All services starting in the background. Logs: /tmp/azurite.log, /tmp/func_receiver.log, /tmp/func_processor.log"
	@echo "Run 'make wait-receiver-func wait-processor-func' or 'make trigger' once they're up."

trigger:
	python3 scripts/trigger_redaction.py

wait-receiver-func:
	@echo "Checking Functions host at $(FUNC_RECEIVER_BASE_URL)..."
	@for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do \
		if curl -sS $(FUNC_RECEIVER_BASE_URL) > /dev/null 2>&1; then \
			echo "Functions host is reachable at $(FUNC_RECEIVER_BASE_URL)"; \
			exit 0; \
		fi; \
		sleep 1; \
	done; \
	echo "Functions host is not reachable at $(FUNC_RECEIVER_BASE_URL)."; \
	echo "Did you run 'make run' in another terminal?"; \
	exit 1

wait-processor-func:
	@echo "Checking Functions host at $(FUNC_PROCESSOR_BASE_URL)..."
	@for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do \
		if curl -sS $(FUNC_PROCESSOR_BASE_URL) > /dev/null 2>&1; then \
			echo "Functions host is reachable at $(FUNC_PROCESSOR_BASE_URL)"; \
			exit 0; \
		fi; \
		sleep 1; \
	done; \
	echo "Functions host is not reachable at $(FUNC_PROCESSOR_BASE_URL)."; \
	echo "Did you run 'make run' in another terminal?"; \
	exit 1

e2e: wait-processor-func
	@echo "Running e2e tests (expects Functions already running)..."
	cd redactor && \
		export PYTHONPATH=$$(pwd) && \
		export E2E_FUNCTION_RECEIVER_BASE_URL=$(FUNC_RECEIVER_BASE_URL) && \
		export E2E_FUNCTION_PROCESSOR_BASE_URL=$(FUNC_PROCESSOR_BASE_URL) && \
		export E2E_SKIP_REDACTION=false && \
		pytest -m e2e -vv -rP

perf: wait-processor-func
	@echo "Running perf tests (expects Functions already running)..."
	cd redactor && \
		export PYTHONPATH=$$(pwd) && \
		PERF_TOTAL=$(PERF_TOTAL) PERF_CONCURRENCY=$(PERF_CONCURRENCY) PERF_TIMEOUT_S=$(PERF_TIMEOUT_S) \
		python3 -m pytest -q -s tests/perf/test_perf_concurrent_redactions.py


