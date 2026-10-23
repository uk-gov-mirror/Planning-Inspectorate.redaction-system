"""Check that Azure OpenAI can be reached before starting an evaluation that calls it."""

from azure.core.exceptions import ClientAuthenticationError
from azure.identity import (
    AzureCliCredential,
    ChainedTokenCredential,
    ManagedIdentityCredential,
)

AZURE_TOKEN_SCOPE = "https://cognitiveservices.azure.com/.default"  # nosec: B105


def check_azure_auth():
    """Stop now if no Azure token can be obtained, as the LLM calls would otherwise fail
    one by one and the evaluation would finish with all-zero scores"""
    # Same credential chain as LLMTextAnalyser
    credential = ChainedTokenCredential(
        ManagedIdentityCredential(), AzureCliCredential()
    )
    try:
        credential.get_token(AZURE_TOKEN_SCOPE)
    except ClientAuthenticationError as e:
        raise SystemExit(
            f"Azure authentication failed, so no LLM calls can be made:\n{e}"
        )
