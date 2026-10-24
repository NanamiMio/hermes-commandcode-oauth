"""Command Code (commandcode.ai) provider profile for accounts that sign in with the Command Code CLI.

The provider is ``commandcode-oauth`` (a historical name): a static key from the CLI's
credential file or studio hand-off authenticates requests; there is no OAuth exchange.
``commandcode-alpha`` remains as an alias for the endpoint it speaks.

Upstream ships Command Code as an *API-key* profile against the OpenAI-compatible Provider API
(``/provider/v1``). Accounts that sign in through the CLI are not covered by that profile: they
reach models through the same ``/alpha/generate`` endpoint the CLI uses.

The integration PR for this provider was asked to ship as an external model-provider plugin, so
everything lives outside the tree and touches no core file, using provider hooks and a temporary
compatibility property:

* :meth:`ProviderProfile.create_client` — supplies the ``/alpha`` transport
  (:mod:`transport`) instead of an HTTP client.
* :meth:`ProviderProfile.fetch_models` — client catalog with confirmed account locks filtered.
* ``fallback_models`` property — live catalogs for OAuth pickers until upstream #122203 ships.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import time
import urllib.parse
import urllib.request
from typing import Any, Optional

from providers import register_provider
from providers.base import ProviderProfile

from .auth import CommandCodeAuthError, auth_handler, read_cli_token, refresh_credential, validate
from .model_metadata import capability_overrides, model_info
from .transport import DEFAULT_MAX_TOKENS, canonical_model_id
from .errors import classify_api_error
from .transport import ALPHA_ORIGIN, CommandCodeAlphaClient
from .credentials import env_token, allow_cli_fallback
from .http_support import open_credentialed
from .catalog import CatalogCache
from .endpoints import api_origin as _api_origin
from .model_access import fetch_access_context, model_rows, organization_id

logger = logging.getLogger("plugins.commandcode_oauth")

# The vendor's zero-cost entries, kept in front of the live catalog so the picker's defaults
# are the entries that cost nothing rather than whatever the endpoint lists last.
ZERO_COST_MODELS = ("poolside/laguna-s-2.1-free",)

# The offline list stays separate from the temporary live fallback_models property.
FALLBACK_MODELS = (
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


def _http_json(url: str, token: str, *, timeout: float) -> Optional[Any]:
    """Small fail-open GET helper: any problem yields ``None``, never an exception."""
    try:
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Bearer {token}")
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", "cli")
        with open_credentialed(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.debug("commandcode: GET %s failed: %s", url, exc)
        return None


def _as_float(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


class CommandCodeOAuthProfile(ProviderProfile):
    """Command Code, through the same ``/alpha/generate`` endpoint the CLI uses."""

    @property
    def fallback_models(self) -> tuple[str, ...]:
        """Bridge old OAuth pickers by fetching when core reads this dataclass field.

        Inspired by anpicasso/hermes-kiro-provider's property bridge. No core patch is needed;
        the setter keeps dataclass construction offline. Cold probes use a three-second HTTP timeout.
        """
        try:
            # Metadata reads before auth add must not sign in using the host CLI implicitly.
            token = self._credential(cli_fallback=False)
            if not token:
                return self._offline_models
            origin = self._catalog_origin()
            models = self._catalog_cache.read(self._catalog_key(token, origin), lambda: self.fetch_models(
                api_key=token, base_url=origin, timeout=3.0))
            # An authenticated probe failure is not permission to advertise the
            # static catalog: it includes models this account may not access.
            return models if models is not None else ()
        except Exception:
            logger.debug("commandcode: compatibility catalog unavailable", exc_info=True)
            return ()

    @fallback_models.setter
    def fallback_models(self, models: tuple[str, ...]) -> None:
        self._offline_models = tuple(models)
        self._catalog_cache = CatalogCache()

    def _catalog_origin(self) -> str:
        from hermes_cli.config import load_config

        config = load_config()
        model = config.get("model") or {}
        names = {self.name, *self.aliases}
        if isinstance(model, dict) and str(model.get("provider") or "").strip().lower() in names:
            configured = str(model.get("base_url") or "").strip()
            if configured:
                return _api_origin(configured)
        return _api_origin(self.base_url)

    def _catalog_key(self, token: str, origin: str) -> tuple[str, str, str]:
        from hermes_constants import hermes_home_key

        return hermes_home_key(), origin, hashlib.sha256(token.encode()).hexdigest()

    def create_client(self, **client_kwargs: Any) -> Any:
        """Supply the ``/alpha`` transport instead of an OpenAI-over-HTTP client."""
        return CommandCodeAlphaClient(**client_kwargs)

    def build_extra_body(self, *, session_id=None, **context):
        return {"commandcode_session_id": session_id} if session_id else {}


    def supported_reasoning_efforts(self, model):
        info = model_info(canonical_model_id(model or ""))
        return tuple("none" if e == "off" else e for e in info["efforts"]) if info else None


    def build_api_kwargs_extras(self, *, reasoning_config=None, model=None, **context):
        from agent.reasoning_effort import clamp_effort
        levels = self.supported_reasoning_efforts(model)
        if not isinstance(reasoning_config, dict) or not levels:
            return {}, {}
        requested = ("none" if reasoning_config.get("enabled") is False
                     else reasoning_config.get("effort"))
        if not requested:
            return {}, {}
        effort = clamp_effort(requested, levels)
        if effort not in levels:
            raise ValueError(f"Unsupported Command Code reasoning effort: {effort}")
        return {}, {"reasoning_effort": "off" if effort == "none" else effort}


    def _credential(self, api_key: str | None = None, *, cli_fallback: bool = True) -> str:
        """Resolve an explicit key, env override, Hermes pool, then the read-only CLI file."""
        token = (api_key or "").strip() or env_token()
        if token:
            return token
        from hermes_cli.auth import read_credential_pool

        for entry in read_credential_pool(self.name):
            if entry.get("last_status") == "dead":
                continue
            token = str(entry.get("access_token") or "").strip()
            if token:
                return token
        if not cli_fallback or not allow_cli_fallback():
            return ""
        try:
            return read_cli_token()
        except CommandCodeAuthError:
            return ""

    def setup_status(self, **kwargs: Any) -> dict[str, Any]:
        token = self._credential(kwargs.get("api_key"))
        identity = validate(token) if token else None
        return {
            "available": True,
            "logged_in": identity is not None,
            "detail": ("Command Code credentials verified." if identity is not None else
                       "Run `hermes auth add commandcode-oauth` to import the CLI key or sign in with your browser."),
            "login_command": ["hermes", "auth", "add", "commandcode-oauth"],
        }

    def discover_models(self, **kwargs: Any) -> list[dict[str, Any]] | None:
        token = self._credential(kwargs.get("api_key"))
        if not token:
            return None
        origin = _api_origin(kwargs.get("base_url")) if kwargs.get("base_url") else self._catalog_origin()
        key = self._catalog_key(token, origin)
        models = self._catalog_cache.read(key, lambda: self.fetch_models(
            api_key=token, base_url=origin, timeout=kwargs.get("timeout", 8.0)))
        return self._catalog_cache.rows(key) if models is not None else None

    def fetch_models(
        self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0
    ) -> list[str] | None:
        """Client catalog minus hidden models and confirmed account locks; unknown access stays open."""
        token = self._credential(api_key)
        if not token:
            return None
        origin = _api_origin(base_url) if base_url else self._catalog_origin()
        key = self._catalog_key(token, origin)
        deadline = time.monotonic() + timeout
        payload = _http_json(f"{origin}{_PROVIDER_PATH_MARKER}/models?view=client", token, timeout=timeout)
        rows = model_rows(payload)
        if rows is None:
            self._catalog_cache.remember(key, None)
            return None
        raw = payload.get("data") or payload.get("models") or []
        if any(isinstance(row, dict) and "category" in row for row in raw):
            context = fetch_access_context(origin, token, _http_json, deadline=deadline)
            if context is None:
                # Keep the last verified list and its lock notes. In particular,
                # never cache a public catalog as an account catalog on timeout.
                logger.warning("commandcode: account access lookup incomplete; retaining verified catalog")
                self._catalog_cache.remember(key, None)
                return None
            rows = model_rows(payload, context)
        models = [row["id"] for row in rows if row["available"]]
        for name in reversed(ZERO_COST_MODELS):
            if name in models:
                models.remove(name)
                models.insert(0, name)
        self._catalog_cache.remember(key, models, rows=rows)
        return models

    def fetch_account_usage(self, *, base_url: str | None = None, api_key: str | None = None):
        """Credits plus the 5-hour/weekly windows, for ``hermes usage`` and ``/usage``.

        ``/alpha/billing/credits`` is what the official CLI's own ``/usage`` reads and it
        accepts the CLI/OAuth token — the Provider API under ``/provider/v1`` does not.
        ``orgId`` (from ``/alpha/whoami``) is passed when discoverable, as the CLI does.
        """
        token = self._credential(api_key)
        if not token:
            return None
        origin = _api_origin(base_url)
        deadline = time.monotonic() + 10.0

        def get(path: str):
            remaining = deadline - time.monotonic()
            return _http_json(f"{origin}{path}", token, timeout=min(4.0, remaining)) if remaining > 0 else None

        org_id = organization_id(get("/alpha/whoami"))
        suffix = f"?orgId={urllib.parse.quote(org_id)}" if org_id else ""
        credits_payload = get(f"/alpha/billing/credits{suffix}")
        if not isinstance(credits_payload, dict):
            return None
        summary = get(f"/alpha/usage/summary{suffix}")

        from agent.account_usage import AccountUsageWindow

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
            windows.append(AccountUsageWindow(
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
    # Keep the canonical provider name used by auth commands and existing configs.
    # Deliberately NOT aliased to "command-code": that token is the model-id vendor prefix
    # ("command-code/<model>") and would read as a provider name here.
    name="commandcode-oauth",
    aliases=("commandcode-alpha",),
    api_mode="chat_completions",
    # The property bridge supplies live OAuth picker rows until hermes-agent#122203 ships.
    env_vars=("COMMANDCODE_CLI_TOKEN",),
    base_url=ALPHA_ORIGIN,
    auth_type="oauth_external",
    display_name="CommandCode (OAuth)",
    description="Command Code — CLI sign-in over the same /alpha/generate endpoint the CLI uses",
    fallback_models=FALLBACK_MODELS,
    default_max_tokens=DEFAULT_MAX_TOKENS,
    model_capabilities=capability_overrides(),
    supports_vision=True,
    supports_vision_tool_messages=True,
    unsupported_response_formats=("json_schema", "json_object"),
    # Provider-owned auth / classification. These three are dataclass *fields* on
    # ProviderProfile (default None), so they belong in the constructor — declaring them in
    # the class body would be shadowed by the instance default.
    auth_handler=auth_handler,
    refresh_credential=refresh_credential,
    classify_api_error=classify_api_error,
)

register_provider(commandcode_oauth)
