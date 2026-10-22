"""Command Code endpoint checks shared by generation and account probes."""
import urllib.parse

ALPHA_ORIGIN = "https://api.commandcode.ai"


class CommandCodeEndpointError(ValueError):
    """An endpoint belonging to another provider must never receive this credential."""
    endpoint_error = True
    status_code = 400
    retryable = False


def api_origin(base_url=None):
    raw = str(base_url or ALPHA_ORIGIN).strip().rstrip("/")
    parsed = urllib.parse.urlsplit(raw)
    host = (parsed.hostname or "").lower().rstrip(".")
    if host == "openrouter.ai" or host.endswith(".openrouter.ai"):
        raise CommandCodeEndpointError(
            "Command Code endpoint is set to OpenRouter. Select the Command Code provider "
            "with base_url=https://api.commandcode.ai before retrying.")
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return ALPHA_ORIGIN
