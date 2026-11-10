variable "environment" {
  description = "The environment to deploy to"
  type        = string
}

variable "environment_short" {
  description = "The short name of environment to deploy to"
  type        = string
}

variable "budget_alert_config" {
  description = "Config for setting up budget alerts"
  type = object({
    expected_cost = number
  })
}

variable "storage_account_replication_type" {
  description = "The storage redundancy setting"
  type        = string
  default     = "LRS"
}

variable "storage_retention_days" {
  description = "The storage data retention period"
  type        = number
  default     = 7
}

variable "log_retention_days" {
  description = "The logging data retention period"
  type        = number
  default     = 30
}

variable "daily_log_cap" {
  description = "The max amount of logging data that can be logged in a day"
  type        = number
  default     = 1
}

variable "tooling_config" {
  description = "Config for the tooling subscription resources"
  type = object({
    container_registry_name = string
    container_registry_rg   = string
    network_name            = string
    network_rg              = string
    subscription_id         = string
  })
}

variable "vnet_cidr_block" {
  description = "The CIDR range for the vnet"
  type        = string
}

variable "subnet_cidr_block" {
  description = "The CIDR range for the subnet"
  type        = string
}

variable "functionapp_cidr_block" {
  description = "The CIDR range for the function app's subnet"
  type        = string
}

variable "function_app_python_version" {
  description = "The python version to use in the function app"
  type        = number
}

variable "storage_containers" {
  description = "The containers to create in the main storage account"
  type        = list(string)
}

variable "openai_quotas" {
  description = "The quota allocations for the Open AI deployments, which equates to x thousand tokens per minute"
  type = object({
    gpt_56_luna = number
  })
  default = {
    gpt_56_luna = 3000
  }
}

variable "worker_caps" {
  description = "The max worker threads per service in the processor function app"
  type = object({
    llm       = number
    vision    = number
    signature = number
  })
  default = {
    llm       = 16
    vision    = 2 # Sized for the Vision F0 tier
    signature = 4
  }
}

variable "vision_requests_per_minute" {
  description = "The max Azure Vision requests per minute, or 0 for no limit. Keep below the Vision tier's limit"
  type        = number
  default     = 15 # Vision F0 allows 20 per minute
}
