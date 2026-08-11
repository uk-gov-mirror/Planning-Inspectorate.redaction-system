"""
The redaction function is implemented using the async-http pattern using Azure Durable Functions.
Information about this pattern can be found in the below documentation
https://learn.microsoft.com/en-us/azure/azure-functions/durable/durable-functions-overview?tabs=in-process%2Cnodejs-v3%2Cv2-model&pivots=python#async-http
https://learn.microsoft.com/en-us/azure/azure-functions/durable/quickstart-python-vscode
"""

import json
import logging
from datetime import timedelta
from typing import Any

import azure.durable_functions as df
import azure.functions as func

app = df.DFApp(http_auth_level=func.AuthLevel.FUNCTION)

logger = logging.getLogger(__name__)


# An HTTP-triggered function with a Durable Functions client binding
@app.service_bus_queue_trigger(
    arg_name="received_message",
    queue_name="redaction-internal-queue",
    connection="AZURE_SERVICE_BUS_NAMESPACE_CONNECTION_STRING",
)
@app.durable_client_input(client_name="client")
async def trigger(
    received_message: func.ServiceBusMessage, client: df.DurableOrchestrationClient
):
    """
    Service Bus trigger for redaction tasks
    """
    request_params: dict[str, Any] = json.loads(
        received_message.get_body().decode("utf-8")
    )
    logger.info("request params: %s", request_params)
    job_id = request_params.pop("job_id", None)
    if not job_id:
        message = "'job_id' property missing from service bus message"
        logger.error(message)
        raise ValueError(message)
    job_id = await client.start_new(
        "trigger_orchestrator", client_input=request_params, instance_id=job_id
    )
    logger.info(f"Started orchestration with ID = '{job_id}'")


# Orchestrator
@app.orchestration_trigger(context_name="context")
def trigger_orchestrator(context: df.DurableOrchestrationContext):
    """
    Orchestrator of the redaction process
    """
    input_params = context.get_input() | {"job_id": context.instance_id}
    timeout_mins = input_params.get("timeoutMinutes", 180)

    retry_options = df.RetryOptions(1, 1)

    # Durable timer
    # https://learn.microsoft.com/en-us/azure/durable-task/common/durable-task-timers
    deadline = context.current_utc_datetime + timedelta(minutes=timeout_mins)
    activity_task = context.call_activity_with_retry(
        "trigger_task", retry_options, input_params
    )
    timeout_task = context.create_timer(deadline)

    winner = yield context.task_any([activity_task, timeout_task])
    if winner == activity_task:
        timeout_task.cancel()
        logger.info(
            "Orchestrator %s: activity completed before timeout (%s mins)",
            context.instance_id,
            timeout_mins,
        )
        return activity_task.result
    else:
        # Timeout: send failure notification via a separate activity
        logger.info(
            "Orchestrator %s: timeout fired after %s mins, sending failure notification",
            context.instance_id,
            timeout_mins,
        )
        error_params = {
            "request_params": input_params,
            "error": f"Activity timed out after {timeout_mins} minutes",
            "job_id": context.instance_id,
        }
        yield context.call_activity("send_failure_notification", error_params)
        raise TimeoutError(f"trigger_task timed out after {timeout_mins} minutes")


# Activity
@app.activity_trigger(input_name="params")
def send_failure_notification(params):
    """
    Lightweight activity to send a service bus failure message when trigger_task times out or fails
    """
    from core.util.enum import PINSService
    from core.util.service_bus_util import ServiceBusUtil

    request_params: dict[str, Any] = params["request_params"]

    pins_service_raw = request_params.get("pinsService", None)
    if not pins_service_raw:
        return

    pins_service = PINSService(pins_service_raw.upper())
    failure_result = {
        "parameters": request_params,
        "stage": request_params.get("stage", "UNKNOWN"),
        "id": params["job_id"],
        "status": "FAIL",
        "message": f"Activity timed out or failed: {params['error']}",
    }
    ServiceBusUtil().send_redaction_process_complete_message(
        pins_service, failure_result
    )


# Activity
@app.activity_trigger(input_name="params")
def trigger_task(params):
    """
    Task which completes the redaction process
    """
    # Import inside this function so that the function app has a chance to start
    # Exceptions will instead be raised when this function is trigger
    from core.redaction_manager import RedactionManager
    from core.util.image_analysis import AzureVisionUtil
    from core.util.logging_util import LoggingUtil

    # Clear static state from any previous invocation sharing this process
    LoggingUtil().clear_logs()
    AzureVisionUtil.clear_cache()

    logger.info("Request params: %s", params)

    job_id = params.pop("job_id")
    stage = params["stage"]

    if stage in ["ANALYSE", "REDACT", "SANITISE"]:
        return RedactionManager(job_id, stage)._try_process(params)

    raise ValueError(f"Unknown stage extracted from service bus message {params}")


# Functions just for smoke testing connections
@app.route(route="testllm", methods=["GET"])
@app.durable_client_input(client_name="client")
async def test_llm_connection(
    req: func.HttpRequest, client: df.DurableOrchestrationClient
):
    """
    This function is called via HTTP get and confirms that the function app can
    connect to the LLM
    """
    from core.connectivity import send_llm_message

    # Return a response with a simplified body
    return func.HttpResponse(
        send_llm_message(),
        status_code=200,
    )


@app.route(route="testazurecomputervision", methods=["GET"])
@app.durable_client_input(client_name="client")
async def test_azure_vision_connection(
    req: func.HttpRequest, client: df.DurableOrchestrationClient
):
    """
    This function is called via HTTP get and confirms that the function app can
    connect to Azure Computer Vision
    """
    from core.connectivity import analyse_image

    # Return a response with a simplified body
    return func.HttpResponse(
        analyse_image(),
        status_code=200,
    )


@app.route(route="testservicebusconnection", methods=["GET"])
@app.durable_client_input(client_name="client")
async def test_service_bus_connection(
    req: func.HttpRequest, client: df.DurableOrchestrationClient
):
    """
    This function is called via HTTP get and confirms that the function app can
    connect to the Back Office Service Bus
    """
    from core.connectivity import send_service_bus_message

    # Return a response with a simplified body
    return func.HttpResponse(
        send_service_bus_message(),
        status_code=200,
    )
