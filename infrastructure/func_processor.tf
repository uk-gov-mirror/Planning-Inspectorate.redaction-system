resource "azurerm_service_plan" "processor" {
  #checkov:skip=CKV_AZURE_212: TODO: Limit reached in subscription
  #checkov:skip=CKV_AZURE_225: TODO: Limit reached in subscription
  name                = "${local.org}-asp-processor-${local.resource_suffix}"
  resource_group_name = azurerm_resource_group.primary.name
  location            = local.location
  os_type             = "Linux"
  sku_name            = "P2v3"
  #worker_count           = 2
  #zone_balancing_enabled = true
}

resource "azurerm_linux_function_app" "processor" {
  name                = "${local.org}-func-processor-${local.resource_suffix}"
  resource_group_name = azurerm_resource_group.primary.name
  location            = local.location

  storage_account_name          = azurerm_storage_account.redaction_storage.name
  storage_account_access_key    = azurerm_storage_account.redaction_storage.primary_access_key
  service_plan_id               = azurerm_service_plan.processor.id
  public_network_access_enabled = false
  virtual_network_subnet_id     = azurerm_subnet.function_app.id
  https_only                    = true

  site_config {
    application_stack {
      python_version = var.function_app_python_version
    }
    application_insights_key = azurerm_application_insights.redaction_system.instrumentation_key
    cors {
      allowed_origins = ["https://portal.azure.com"]
    }
    always_on = true
  }
  identity {
    type = "SystemAssigned"
  }
  app_settings = {
    "WEBSITE_CONTENTAZUREFILECONNECTIONSTRING" = "DefaultEndpointsProtocol=https;AccountName=${azurerm_storage_account.redaction_storage.name};AccountKey=${azurerm_storage_account.redaction_storage.primary_access_key};EndpointSuffix=core.windows.net"
    "WEBSITE_CONTENTSHARE" : "${local.org}-func-processor-${local.resource_suffix}"
    "SCM_DO_BUILD_DURING_DEPLOYMENT"                = "true"
    "OPENAI_ENDPOINT"                               = azurerm_cognitive_account.open_ai.endpoint
    "AZURE_VISION_ENDPOINT"                         = azurerm_cognitive_account.computer_vision.endpoint
    "STORAGE_NAME"                                  = azurerm_storage_account.redaction_storage.name
    "APP_INSIGHTS_CONNECTION_STRING"                = azurerm_application_insights.redaction_system.connection_string
    "WEBSITE_CONTENTOVERVNET"                       = 1
    "AZURE_SERVICE_BUS_NAMESPACE"                   = data.azurerm_servicebus_namespace.backoffice.name
    "AZURE_SERVICE_BUS_NAMESPACE_CONNECTION_STRING" = data.azurerm_servicebus_namespace.backoffice.default_primary_connection_string
    "SIGNATURE_DETECTOR_ENDPOINT"                   = azurerm_linux_web_app.signature_detector.default_hostname
    "LLM_MAX_WORKERS"                               = var.worker_caps.llm
    "VISION_MAX_WORKERS"                            = var.worker_caps.vision
    "SIGNATURE_MAX_WORKERS"                         = var.worker_caps.signature
    "VISION_REQUESTS_PER_MINUTE"                    = var.vision_requests_per_minute
    "LLM_TOKEN_RATE_LIMIT"                          = var.openai_quotas.gpt_56_luna * 1000
    "LLM_REQUEST_RATE_LIMIT"                        = var.openai_quotas.gpt_56_luna
    "LLM_QUOTA_ALLOCATION"                          = var.llm_quota_allocation
    "RULE_MAX_WORKERS"                              = var.rule_max_workers
  }
}

resource "azurerm_storage_share" "function_app_processor" {
  name               = azurerm_linux_function_app.processor.name
  storage_account_id = azurerm_storage_account.redaction_storage.id
  quota              = 5120
}

############################################################################
# RBAC
############################################################################
resource "azurerm_role_assignment" "function_app_processor_storage_contributor" {
  scope                = azurerm_storage_account.redaction_storage.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azurerm_linux_function_app.processor.identity[0].principal_id
}

resource "azurerm_role_assignment" "function_app_processor_openai_contributor" {
  scope                = azurerm_cognitive_account.open_ai.id
  role_definition_name = "Cognitive Services OpenAI User"
  principal_id         = azurerm_linux_function_app.processor.identity[0].principal_id
}

resource "azurerm_role_assignment" "function_app_processor_computervision_contributor" {
  scope                = azurerm_cognitive_account.computer_vision.id
  role_definition_name = "Cognitive Services User"
  principal_id         = azurerm_linux_function_app.processor.identity[0].principal_id
}

resource "azurerm_role_assignment" "function_app_processor_servicebus_datasender" {
  scope                = data.azurerm_servicebus_namespace.backoffice.id
  role_definition_name = "Azure Service Bus Data Sender"
  principal_id         = azurerm_linux_function_app.processor.identity[0].principal_id
}