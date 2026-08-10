############################################################################
# Virtual networks
############################################################################
resource "azurerm_virtual_network" "redaction_system" {
  name                = "vnet-${local.service_name}-${var.environment}-${local.location_short}"
  location            = local.location
  resource_group_name = azurerm_resource_group.primary.name
  address_space       = [var.vnet_cidr_block]

  tags = local.tags
}

resource "azurerm_subnet" "redaction_system" {
  name                              = "RedactionSubnet"
  resource_group_name               = azurerm_resource_group.primary.name
  address_prefixes                  = [var.subnet_cidr_block]
  virtual_network_name              = azurerm_virtual_network.redaction_system.name
  service_endpoints                 = ["Microsoft.Storage"]
  private_endpoint_network_policies = "Enabled"
}

resource "azurerm_subnet" "function_app" {
  name                              = "FunctionAppSubnet"
  resource_group_name               = azurerm_resource_group.primary.name
  address_prefixes                  = [var.functionapp_cidr_block]
  virtual_network_name              = azurerm_virtual_network.redaction_system.name
  service_endpoints                 = ["Microsoft.Storage"]
  private_endpoint_network_policies = "Enabled"

  delegation {
    name = "functionAppDelegation"
    service_delegation {
      name    = "Microsoft.Web/serverFarms"
      actions = ["Microsoft.Network/virtualNetworks/subnets/action"]
    }
  }
}


############################################################################
# DNS Zone Vnet links
############################################################################
resource "azurerm_private_dns_zone_virtual_network_link" "storage" {
  for_each              = { for idx, val in local.storage_subresources : idx => val }
  name                  = "${local.org}-vnetlink-${each.value}-${local.service_name}-${var.environment}"
  resource_group_name   = var.tooling_config.network_rg
  private_dns_zone_name = data.azurerm_private_dns_zone.storage[each.key].name
  virtual_network_id    = azurerm_virtual_network.redaction_system.id
  provider              = azurerm.tooling
  resolution_policy     = "NxDomainRedirect"

  tags = local.tags
}

resource "azurerm_private_dns_zone_virtual_network_link" "function" {
  name                  = "${local.org}-vnetlink-functions-${local.service_name}-${var.environment}"
  resource_group_name   = var.tooling_config.network_rg
  private_dns_zone_name = data.azurerm_private_dns_zone.function.name
  virtual_network_id    = azurerm_virtual_network.redaction_system.id
  provider              = azurerm.tooling
  resolution_policy     = "NxDomainRedirect"

  tags = local.tags
}

resource "azurerm_private_dns_zone_virtual_network_link" "ai" {
  name                  = "${local.org}-vnetlink-ai-${local.service_name}-${var.environment}"
  resource_group_name   = var.tooling_config.network_rg
  private_dns_zone_name = data.azurerm_private_dns_zone.ai.name
  virtual_network_id    = azurerm_virtual_network.redaction_system.id
  provider              = azurerm.tooling
  resolution_policy     = "NxDomainRedirect"

  tags = local.tags
}

resource "azurerm_private_dns_zone_virtual_network_link" "open_ai" {
  name                  = "${local.org}-vnetlink-openai-${local.service_name}-${var.environment}"
  resource_group_name   = var.tooling_config.network_rg
  private_dns_zone_name = data.azurerm_private_dns_zone.openai.name
  virtual_network_id    = azurerm_virtual_network.redaction_system.id
  provider              = azurerm.tooling
  resolution_policy     = "NxDomainRedirect"

  tags = local.tags
}

resource "azurerm_private_dns_zone_virtual_network_link" "servicebus" {
  name                  = "${local.org}-vnetlink-servicebus-${local.service_name}-${var.environment}"
  resource_group_name   = var.tooling_config.network_rg
  private_dns_zone_name = data.azurerm_private_dns_zone.servicebus.name
  virtual_network_id    = azurerm_virtual_network.redaction_system.id
  provider              = azurerm.tooling

  tags = local.tags
}

############################################################################
# Private endpoints
############################################################################
resource "azurerm_private_endpoint" "redaction_storage" {
  for_each            = { for idx, val in local.storage_subresources : idx => val }
  name                = "${local.org}-pe-${azurerm_storage_account.redaction_storage.name}-${each.value}-${var.environment}"
  resource_group_name = azurerm_resource_group.primary.name
  location            = local.location
  subnet_id           = azurerm_subnet.redaction_system.id

  private_dns_zone_group {
    name                 = "${local.org}-pdns-${local.service_name}-storage-${each.value}-${var.environment}"
    private_dns_zone_ids = [data.azurerm_private_dns_zone.storage[each.key].id]
  }

  private_service_connection {
    name                           = "${local.org}-psc-${local.service_name}-storage-${each.value}-${var.environment}"
    is_manual_connection           = false
    private_connection_resource_id = azurerm_storage_account.redaction_storage.id
    subresource_names              = [each.value]
  }

  tags = local.tags
}

resource "azurerm_private_endpoint" "function_app_processor" {
  name                = "${local.org}-pe-${azurerm_linux_function_app.processor.name}-${var.environment}"
  resource_group_name = azurerm_resource_group.primary.name
  location            = local.location
  subnet_id           = azurerm_subnet.redaction_system.id

  private_dns_zone_group {
    name                 = "${local.org}-pdns-${local.service_name}-functionapp-processor-${var.environment}"
    private_dns_zone_ids = [data.azurerm_private_dns_zone.function.id]
  }

  private_service_connection {
    name                           = "${local.org}-psc-${local.service_name}-functionapp-processor-${var.environment}"
    is_manual_connection           = false
    private_connection_resource_id = azurerm_linux_function_app.processor.id
    subresource_names              = ["sites"]
  }

  tags = local.tags
}

resource "azurerm_private_endpoint" "function_app" {
  name                = "${local.org}-pe-${azurerm_linux_function_app.receiver.name}-${var.environment}"
  resource_group_name = azurerm_resource_group.primary.name
  location            = local.location
  subnet_id           = azurerm_subnet.redaction_system.id

  private_dns_zone_group {
    name                 = "${local.org}-pdns-${local.service_name}-functionapp-${var.environment}"
    private_dns_zone_ids = [data.azurerm_private_dns_zone.function.id]
  }

  private_service_connection {
    name                           = "${local.org}-psc-${local.service_name}-functionapp-${var.environment}"
    is_manual_connection           = false
    private_connection_resource_id = azurerm_linux_function_app.receiver.id
    subresource_names              = ["sites"]
  }

  tags = local.tags
}

resource "azurerm_private_endpoint" "open_ai_cognitiveservices" { # pins-pe-pins-openai-redaction-system-training-weu-cognitiveservices
  name                = "${local.org}-pe-${azurerm_cognitive_account.open_ai.name}-cognitiveservices"
  resource_group_name = azurerm_resource_group.primary.name
  location            = local.location
  subnet_id           = azurerm_subnet.redaction_system.id

  private_dns_zone_group {
    name                 = "${local.org}-pdns-${local.service_name}-openai-cognitiveservices-${var.environment}"
    private_dns_zone_ids = [data.azurerm_private_dns_zone.ai.id]
  }

  private_service_connection {
    name                           = "${local.org}-psc-${local.service_name}-openai-cognitiveservices-${var.environment}"
    is_manual_connection           = false
    private_connection_resource_id = azurerm_cognitive_account.open_ai.id
    subresource_names              = ["account"]
  }

  tags = local.tags
}

resource "azurerm_private_endpoint" "open_ai_openai" { # pins-pe-pins-openai-redaction-system-training-weu-openai
  name                = "${local.org}-pe-${azurerm_cognitive_account.open_ai.name}-openai"
  resource_group_name = azurerm_resource_group.primary.name
  location            = local.location
  subnet_id           = azurerm_subnet.redaction_system.id

  private_dns_zone_group {
    name                 = "${local.org}-pdns-${local.service_name}-openai-openai-${var.environment}"
    private_dns_zone_ids = [data.azurerm_private_dns_zone.openai.id]
  }

  private_service_connection {
    name                           = "${local.org}-psc-${local.service_name}-openai-openai-${var.environment}"
    is_manual_connection           = false
    private_connection_resource_id = azurerm_cognitive_account.open_ai.id
    subresource_names              = ["account"]
  }

  tags = local.tags
}

resource "azurerm_private_endpoint" "computer_vision_cognitiveservices" { # pins-pe-pins-cv-redaction-system-training-uks-cognitiveservices
  name                = "${local.org}-pe-${azurerm_cognitive_account.computer_vision.name}-cognitiveservices"
  resource_group_name = azurerm_resource_group.primary.name
  location            = local.location
  subnet_id           = azurerm_subnet.redaction_system.id

  private_dns_zone_group {
    name                 = "${local.org}-pdns-${local.service_name}-computervision-cognitiveservices-${var.environment}"
    private_dns_zone_ids = [data.azurerm_private_dns_zone.ai.id]
  }

  private_service_connection {
    name                           = "${local.org}-psc-${local.service_name}-computervision-cognitiveservices-${var.environment}"
    is_manual_connection           = false
    private_connection_resource_id = azurerm_cognitive_account.computer_vision.id
    subresource_names              = ["account"]
  }

  tags = local.tags
}

############################################################################
# Network peering
############################################################################
resource "azurerm_virtual_network_peering" "redaction_to_tooling" {
  name                      = "${local.org}-peer-${local.service_name}-to-tooling-${var.environment}"
  resource_group_name       = azurerm_resource_group.primary.name
  virtual_network_name      = azurerm_virtual_network.redaction_system.name
  remote_virtual_network_id = data.azurerm_virtual_network.tooling.id
}

resource "azurerm_virtual_network_peering" "tooling_to_redaction" {
  name                      = "${local.org}-peer-tooling-to-${local.service_name}-${var.environment}"
  resource_group_name       = var.tooling_config.network_rg
  virtual_network_name      = var.tooling_config.network_name
  remote_virtual_network_id = azurerm_virtual_network.redaction_system.id

  provider = azurerm.tooling
}

############################################################################
# Network security
############################################################################
resource "azurerm_network_security_group" "nsg" {
  for_each            = { for i, val in [azurerm_subnet.redaction_system, azurerm_subnet.function_app] : i => val }
  name                = "${local.org}-nsg-${local.service_name}-${each.value.name}"
  location            = local.location
  resource_group_name = azurerm_resource_group.primary.name

  tags = local.tags
}

resource "azurerm_subnet_network_security_group_association" "nsg" {
  for_each = { for i, val in [azurerm_subnet.redaction_system, azurerm_subnet.function_app] : i => val }

  network_security_group_id = azurerm_network_security_group.nsg[each.key].id
  subnet_id                 = each.value.id

  depends_on = [
    azurerm_network_security_group.nsg
  ]
}
