"""Command Code (commandcode.ai) provider profile for Go/free-tier accounts.

The provider is named ``commandcode-alpha`` after the wire it speaks — the private
``/alpha/generate`` protocol — following upstream's product+wire convention (``commandcode``
for the OpenAI-compatible Provider API, ``commandcode-anthropic`` for the Anthropic wire).
``commandcode-oauth`` stays as an alias: it is the name the integration PR and its issue
use, and what existing configs already say.

Upstream ships Command Code only as *API-key* profiles against the OpenAI-compatible
Provider API (``/provider/v1``). Accounts on the Go/free tiers have no Provider-API access
at all: they reach models through the CLI's private ``/alpha/generate`` protocol, using the
grant the Command Code CLI stores in ``~/.commandcode/auth.json``.

Per upstream's ruling on the integration PR — *"ship the OAuth/alpha adapter as an external
model-provider plugin so it can track the unversioned protocol independently"* — this lives
entirely outside the tree and touches no core file, using documented extension points:

* :meth:`ProviderProfile.create_client` — supplies the ``/alpha`` transport
  (:mod:`transport`) instead of an HTTP client.
* :meth:`ProviderProfile.fetch_models` — the account's live catalogue.
* :meth:`ProviderProfile.fetch_account_usage` — credits and 5-hour/weekly windows for
  ``hermes usage`` / ``/usage``.
* :meth:`ProviderProfile.auth_handler` / ``refresh_credential`` — ``hermes auth
  add|status|logout`` and pooled-row rotation (:mod:`auth`), so the plugin owns its own
  credential story instead of relying on core plumbing.
* :meth:`ProviderProfile.classify_api_error` — maps this endpoint's failures (notably a
  ``402 Insufficient Balance`` delivered *inside* a 200 stream) onto Hermes failover
  reasons (:mod:`errors`).

Deliberately absent: ``get_usage_cost``. Command Code prices live at
https://commandcode.ai/pricing and are not on the wire, so a hardcoded table would drift
and read as an invoice; the accurate spend view is ``hermes usage`` (credits, 5-hour and
weekly windows, billing-period total, straight from ``/alpha``) plus the per-request detail
on the account's usage page.

Install by dropping this directory into ``~/.hermes/plugins/model-providers/`` (or shipping
it as a distribution exposing the ``hermes_agent.plugins`` entry point).
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

logger = logging.getLogger("plugins.commandcode_alpha")

# Models the Go tier always has; kept in front of the live catalogue so the picker's
# defaults are the free ones rather than whatever the relay lists last.
FREE_FIRST_MODELS = ("meituan/LongCat-2.0:free", "poolside/laguna-s-2.1-free")

# Catalog snapshot for the picker: core never probes a non-api_key profile, so
# ``provider_model_ids`` (the /model picker, the Desktop picker and `hermes model`) reads
# this list for us — ``fetch_models`` still serves the live list to surfaces that probe
# (the setup wizard). Refresh by pasting ``fetch_models()`` output back here.
FALLBACK_MODELS = (
    "meituan/LongCat-2.0:free", "poolside/laguna-s-2.1-free", "claude-sonnet-5", "claude-sonnet-4-6",
    "claude-fable-5-1", "claude-fable-5", "claude-opus-5-5", "claude-opus-5",
    "claude-opus-4-8", "claude-opus-4-7", "claude-haiku-4-5-20251001", "gpt-6-astra",
    "gpt-6-sol", "gpt-6-luna", "gpt-5.6-sol", "gpt-5.6-terra",
    "gpt-5.6-luna", "gpt-5.5", "gpt-5.4", "gpt-5.3-codex",
    "gpt-5.4-mini", "deepseek/deepseek-v4-pro", "deepseek/deepseek-v4-flash", "deepseek/deepseek-v4-flash-vision-exp",
    "deepseek/deepseek-v4-flash-fast", "deepseek/deepseek-v4.1-flash", "moonshotai/Kimi-K3", "moonshotai/Kimi-K2.7-Code",
    "moonshotai/Kimi-K2.7-Code-Highspeed", "moonshotai/Kimi-K2.6", "moonshotai/Kimi-K2.5", "z-ai/glm-5.3-flash",
    "z-ai/glm-5.3-flashx", "zai-org/GLM-5.3", "zai-org/GLM-5.2", "zai-org/GLM-5.2-Fast",
    "zai-org/GLM-5.1", "zai-org/GLM-5", "MiniMaxAI/MiniMax-M3", "MiniMaxAI/MiniMax-M2.7",
    "MiniMaxAI/MiniMax-M2.5", "xiaomi/mimo-v2.6-pro", "xiaomi/mimo-v2.6-pro-ultraspeed", "xiaomi/mimo-v2.6-flash",
    "xiaomi/mimo-v2.5-pro", "xiaomi/mimo-v2.5", "Qwen/Qwen3.8-Omni-Flash", "Qwen/Qwen3.8-Max-0902",
    "Qwen/Qwen3.8-Max", "Qwen/Qwen3.8-27B", "Qwen/Qwen3.8-Flash", "Qwen/Qwen3.7-Max",
    "Qwen/Qwen3.7-Plus", "Qwen/Qwen3.7-Flash", "Qwen/Qwen3.6-Max-Preview", "Qwen/Qwen3.6-Plus",
    "meituan/LongCat-2.0", "stepfun/Step-5-Preview", "stepfun/Step-3.7-Flash", "stepfun/Step-3.5-Flash",
    "tencent/hy3-paid", "tencent/hy4-preview", "google/gemini-3.8-flash", "google/gemini-3.7-flash",
    "google/gemini-3.6-flash", "google/gemini-3.5-flash", "google/gemini-3.5-flash-lite", "google/gemini-3.1-flash-lite",
    "sakana/fugu-ultra", "nvidia/nemotron-3-ultra-550b-a55b", "thinkingmachines/inkling", "thinkingmachines/inkling-small",
    "stealth/space-bunny-alpha", "inclusionai/ling-3.0-flash-sante:free", "meta/muse-spark-1.1", "meta/muse-spark-1.2",
    "meta/muse-spark-1.2-contributor", "meta/muse-spark-1.3", "meta/muse-spark-1.3-contributor", "xai/grok-4.5",
    "xai/grok-4.6", "xai/grok-4.7",
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


class CommandCodeAlphaProfile(ProviderProfile):
    """Command Code Go/free tiers through the CLI's ``/alpha/generate`` protocol."""

    def create_client(self, **client_kwargs: Any) -> Any:
        """Supply the ``/alpha`` transport instead of an OpenAI-over-HTTP client."""
        return CommandCodeAlphaClient(**client_kwargs)

    def fetch_models(
        self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0
    ) -> list[str] | None:
        """The account's live catalogue, free models first.

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
        for name in reversed(FREE_FIRST_MODELS):
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


commandcode_alpha = CommandCodeAlphaProfile(
    # Named after the wire, like upstream's other Command Code profiles; ``commandcode-oauth``
    # is the alias because that is what the integration PR/issue call it and what existing
    # configs (``model.provider``) already say — so nothing has to be migrated.
    # Deliberately NOT aliased to "command-code": that token is the model-id vendor prefix
    # ("command-code/<model>") and would read as a provider name here.
    name="commandcode-alpha",
    aliases=("commandcode-oauth",),
    api_mode="chat_completions",
    env_vars=(),
    base_url=ALPHA_ORIGIN,
    auth_type="oauth_external",
    display_name="CommandCode (OAuth)",
    description="Command Code Go/free accounts — CLI credentials over the private /alpha/generate protocol",
    fallback_models=FALLBACK_MODELS,
    # Provider-owned auth / classification. These three are dataclass *fields* on
    # ProviderProfile (default None), so they belong in the constructor — declaring them in
    # the class body would be shadowed by the instance default.
    auth_handler=auth_handler,
    refresh_credential=refresh_credential,
    classify_api_error=classify_api_error,
)

register_provider(commandcode_alpha)
