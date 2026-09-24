"""Command Code (commandcode.ai) provider profile for accounts that sign in with the Command Code CLI.

The provider is ``commandcode-oauth``: ``oauth`` names the credential path — the sign-in the
Command Code CLI already stores, or the studio hand-off, is what the transport authenticates with.
``commandcode-alpha`` remains as an alias for the endpoint it speaks.

Upstream ships Command Code as an *API-key* profile against the OpenAI-compatible Provider API
(``/provider/v1``). Accounts that sign in through the CLI are not covered by that profile: they
reach models through the same ``/alpha/generate`` endpoint the CLI uses.

The integration PR for this provider was asked to ship as an external model-provider plugin, so
everything lives outside the tree and touches no core file, using documented extension points:

* :meth:`ProviderProfile.create_client` — supplies the ``/alpha`` transport
  (:mod:`transport`) instead of an HTTP client.
* :meth:`ProviderProfile.fetch_models` — the account's live catalog.
"""

from __future__ import annotations

import datetime
import json
import logging
import urllib.parse
import urllib.request
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from providers import register_provider
from providers.base import ProviderProfile

from .auth import auth_handler, refresh_credential
from .errors import classify_api_error
from .transport import ALPHA_ORIGIN, CommandCodeAlphaClient, cli_token

logger = logging.getLogger("plugins.commandcode_oauth")

# The vendor's zero-cost entries, kept in front of the live catalog so the picker's defaults
# are the entries that cost nothing rather than whatever the endpoint lists last.
ZERO_COST_MODELS = ("meituan/LongCat-2.0:free", "poolside/laguna-s-2.1-free")

# Offline fallback only. The picker probes this provider live through ``fetch_models``; this
# short list is what resolves when there is no credential or the network is down. Refresh it by
# pasting a few ``fetch_models()`` ids here.
FALLBACK_MODELS = (
    "meituan/LongCat-2.0:free",
    "poolside/laguna-s-2.1-free",
    "deepseek/deepseek-v4.1-flash",
    "deepseek/deepseek-v4-pro",
    "Qwen/Qwen3.8-Omni-Flash",
    "meta/muse-spark-1.3-contributor",
    "meta/muse-spark-1.3",
    "moonshotai/Kimi-K3",
    "z-ai/glm-5.3-flash",
    "MiniMaxAI/MiniMax-M3",
    "Qwen/Qwen3.7-Max",
)

# The portal endpoints live at the API origin, not under the Provider API's /provider/v1.
_PROVIDER_PATH_MARKER = "/provider/v1"


def _api_origin(base_url: Optional[str]) -> str:
    raw = (base_url or "").strip().rstrip("/")
    if not raw or _PROVIDER_PATH_MARKER in raw:
        raw = ALPHA_ORIGIN
    parsed = urllib.parse.urlsplit(raw)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return ALPHA_ORIGIN


def _http_json(url: str, token: str, *, timeout: float) -> Optional[Any]:
    """Small fail-open GET helper: any problem yields ``None``, never an exception."""
    try:
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Bearer {token}")
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", "cli")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.debug("commandcode: GET %s failed: %s", url, exc)
        return None


def _as_float(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


# ``auth_type="api_key"`` is deliberate: the model picker only probes providers declared that way
# (hermes_cli/models.py: a non-api_key profile is answered from ``fallback_models`` alone). Our
# credential *is* a bearer token used as a key — the pool row is resolved through the api_key path,
# which reaches the same pool — while everything OAuth-shaped (the CLI grant, the studio hand-off,
# rotation) lives in ``auth_handler`` / ``refresh_credential`` below. The static list is therefore
# only an offline fallback, not the catalog.
class CommandCodeOAuthProfile(ProviderProfile):
    """Command Code, through the same ``/alpha/generate`` endpoint the CLI uses."""

    def create_client(self, **client_kwargs: Any) -> Any:
        """Supply the ``/alpha`` transport instead of an OpenAI-over-HTTP client."""
        return CommandCodeAlphaClient(**client_kwargs)

    def fetch_models(
        self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0
    ) -> list[str] | None:
        """The account's live catalog, zero-cost entries first.

        ``None`` (not ``[]``) when the account is unreachable: the caller then falls back to
        ``fallback_models``, whereas an empty list would look like a genuinely empty account.
        """
        token = (api_key or "").strip() or cli_token()
        if not token:
            return None
        payload = _http_json(f"{_api_origin(base_url)}{_PROVIDER_PATH_MARKER}/models", token, timeout=timeout)
        models: List[str] = []
        if isinstance(payload, dict):
            for item in payload.get("data") or payload.get("models") or []:
                model_id = item.get("id") if isinstance(item, dict) else item
                if isinstance(model_id, str) and model_id.strip():
                    models.append(model_id.strip())
        if not models:
            return None
        for name in reversed(ZERO_COST_MODELS):
            if name in models:
                models.remove(name)
            models.insert(0, name)
        return models

    def fetch_account_usage(self, *, base_url: str | None = None, api_key: str | None = None):
        """Credits plus the 5-hour/weekly windows, for ``hermes usage`` and ``/usage``.

        ``/alpha/billing/credits`` is what the official CLI's own ``/usage`` reads and it
        accepts the CLI/OAuth token — the Provider API under ``/provider/v1`` does not.
        ``orgId`` (from ``/alpha/whoami``) is passed when discoverable, as the CLI does.
        """
        token = (api_key or "").strip() or cli_token()
        if not token:
            return None
        origin = _api_origin(base_url)
        org_id = ""
        whoami = _http_json(f"{origin}/alpha/whoami", token, timeout=4.0)
        if isinstance(whoami, dict):
            for key in ("organizationId", "orgId", "organization_id"):
                value = whoami.get(key)
                if isinstance(value, str) and value.strip():
                    org_id = value.strip()
                    break
        suffix = f"?orgId={urllib.parse.quote(org_id)}" if org_id else ""
        credits_payload = _http_json(f"{origin}/alpha/billing/credits{suffix}", token, timeout=4.0)
        if not isinstance(credits_payload, dict):
            return None
        summary = _http_json(f"{origin}/alpha/usage/summary{suffix}", token, timeout=4.0)

        windows = []
        details = []
        limits = credits_payload.get("windowLimits") or {}
        for key, label in (("fiveHour", "5-hour"), ("weekly", "Weekly")):
            block = limits.get(key) or {}
            used, cap = _as_float(block.get("used")), _as_float(block.get("cap"))
            if used is None or not cap:
                continue
            reset_at = block.get("resetAt")
            reset_dt = (
                datetime.datetime.fromtimestamp(float(reset_at) / 1000.0, tz=datetime.timezone.utc)
                if isinstance(reset_at, (int, float)) and reset_at > 0 else None
            )
            windows.append(SimpleNamespace(
                label=label,
                used_percent=max(0.0, min(100.0, used / cap * 100.0)),
                reset_at=reset_dt,
                detail=f"${used:.2f} of ${cap:.2f} used",
            ))
        credits = credits_payload.get("credits") or {}
        monthly = _as_float(credits.get("monthlyCredits"))
        if monthly is not None:
            details.append(f"Monthly credits left: ${monthly:.2f}")
        purchased = _as_float(credits.get("purchasedCredits"))
        if purchased:
            details.append(f"Purchased credits: ${purchased:.2f}")
        if isinstance(summary, dict):
            total_cost = _as_float(summary.get("totalCost"))
            count = _as_float(summary.get("totalCount"))
            if total_cost is not None:
                calls = f" over {int(count)} calls" if count else ""
                details.append(f"This billing period: ${total_cost:.2f}{calls}")

        from agent.account_usage import AccountUsageSnapshot

        return AccountUsageSnapshot(
            provider=self.name,
            source="billing-api",
            fetched_at=datetime.datetime.now(datetime.timezone.utc),
            title="Command Code limits",
            windows=tuple(windows),
            details=tuple(details),
            raw=credits_payload,
        )


commandcode_oauth = CommandCodeOAuthProfile(
    # Named after the wire, like upstream's other Command Code profiles; ``commandcode-oauth``
    # is the alias because that is what the integration PR/issue call it and what existing
    # configs (``model.provider``) already say — so nothing has to be migrated.
    # Deliberately NOT aliased to "command-code": that token is the model-id vendor prefix
    # ("command-code/<model>") and would read as a provider name here.
    name="commandcode-oauth",
    aliases=("commandcode-alpha",),
    api_mode="chat_completions",
    # Declared so the registry mirror accepts an api_key row at all
    # (hermes_cli/auth_plugin_providers.register_plugin_provider drops api_key profiles with no
    # env_vars). The pool row is the primary credential — auth_handler fills it — and this env
    # var accepts the same bearer the Command Code CLI stores, which is what /alpha/generate wants.
    env_vars=("COMMANDCODE_CLI_TOKEN",),
    base_url=ALPHA_ORIGIN,
    auth_type="api_key",
    display_name="CommandCode (OAuth)",
    description="Command Code — CLI sign-in over the same /alpha/generate endpoint the CLI uses",
    fallback_models=FALLBACK_MODELS,
    # Provider-owned auth / classification. These three are dataclass *fields* on
    # ProviderProfile (default None), so they belong in the constructor — declaring them in
    # the class body would be shadowed by the instance default.
    auth_handler=auth_handler,
    refresh_credential=refresh_credential,
    classify_api_error=classify_api_error,
)

register_provider(commandcode_oauth)
