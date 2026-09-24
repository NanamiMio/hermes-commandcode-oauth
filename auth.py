"""Command Code credential handling for the ``commandcode-alpha`` provider plugin.

Two ways in, both ending as a pooled OAuth row:

1. **Import** the grant the official Command Code CLI already stored in
   ``~/.commandcode/auth.json`` — the common case, since the CLI is the thing that
   logged in. This is the same file the official CLI and every community bridge read.
2. **Loopback hand-off**: open
   ``https://commandcode.ai/studio/auth/cli?callback=…&state=…`` and catch the JSON
   ``POST`` the studio page makes to ``127.0.0.1:5959/callback``. The grant is written
   back to ``~/.commandcode/auth.json`` so the CLI and Hermes share one login.

Per ``providers/base.py`` a non-api-key plugin owns its own auth: ``auth_handler``
serves ``hermes auth add|status|logout <name>`` and ``refresh_credential`` rotates a
pooled row. Both are self-contained — nothing here imports Hermes internals beyond the
credential-pool types, so the plugin keeps working when the core moves.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
from urllib.parse import urlparse

logger = logging.getLogger("plugins.commandcode_oauth.auth")

CLI_AUTH_PATH = Path.home() / ".commandcode" / "auth.json"
ALPHA_ORIGIN = "https://api.commandcode.ai"
WHOAMI_URL = f"{ALPHA_ORIGIN}/alpha/whoami"
STUDIO_URL = "https://commandcode.ai"
CALLBACK_PORT = 5959
# ``manual:`` prefix = the pool never prunes the row when re-seeding from env/config.
POOL_SOURCE = "manual:commandcode_cli"
CANONICAL_PROVIDER = "commandcode-alpha"


class CommandCodeAuthError(RuntimeError):
    """A credential problem worth showing verbatim. Never carries the token itself."""


def pool_provider(args: Any) -> str:
    """Canonical profile name for the credential pool — ``args.provider`` may be an alias."""
    raw = str(getattr(args, "provider", "") or "").strip().lower()
    try:
        from providers import get_provider_profile

        profile = get_provider_profile(raw)
        if profile is not None:
            return profile.name
    except Exception:  # pragma: no cover - registry always present in a real process
        logger.debug("pool_provider: registry lookup failed for %r", raw)
    return raw or CANONICAL_PROVIDER


# ── the CLI's credential file ────────────────────────────────────────────────

def read_cli_token() -> str:
    """The access token from ``~/.commandcode/auth.json``; raises when absent/invalid."""
    if not CLI_AUTH_PATH.exists():
        raise CommandCodeAuthError(
            f"Command Code CLI credentials not found at {CLI_AUTH_PATH}. "
            "Run `hermes auth add commandcode-alpha` to sign in."
        )
    try:
        data = json.loads(CLI_AUTH_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CommandCodeAuthError(f"Could not read {CLI_AUTH_PATH}: {exc}") from exc
    if not isinstance(data, dict):
        raise CommandCodeAuthError(f"{CLI_AUTH_PATH} is not a JSON object.")
    token = str(data.get("apiKey") or data.get("api_key") or "").strip()
    if not token:
        raise CommandCodeAuthError(f"{CLI_AUTH_PATH} has no apiKey.")
    return token


def write_cli_token(token: str, *, user_id: str = "", user_name: str = "", key_name: str = "hermes-plugin") -> None:
    """Atomically write the grant back so the CLI and Hermes share one login (0600)."""
    payload = {
        "apiKey": token,
        "userId": user_id,
        "userName": user_name,
        "keyName": key_name,
        "authenticatedAt": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
    }
    CLI_AUTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CLI_AUTH_PATH.with_suffix(f".tmp-{os.getpid()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp, CLI_AUTH_PATH)
    finally:
        tmp.unlink(missing_ok=True)


def validate(token: str, *, timeout: float = 10.0) -> Optional[Dict[str, str]]:
    """Confirm the grant with ``/alpha/whoami``; None when the token is not accepted."""
    key = (token or "").strip()
    if not key:
        return None
    try:
        import urllib.request

        req = urllib.request.Request(WHOAMI_URL)
        req.add_header("Authorization", f"Bearer {key}")
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", "cli")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.debug("validate: %s", exc)
        return None
    user = data.get("user") if isinstance(data, dict) else None
    if isinstance(user, dict):
        return {"user_id": str(user.get("id") or ""), "user_name": str(user.get("userName") or "")}
    return {"user_id": "", "user_name": ""}


# ── browser hand-off (loopback) ──────────────────────────────────────────────

def _callback_handler(expected_state: str):
    result: Dict[str, Any] = {"payload": None, "error": None}

    class _Handler(BaseHTTPRequestHandler):
        def _cors(self, origin: Optional[str] = None) -> None:
            self.send_header("Access-Control-Allow-Origin", origin or STUDIO_URL)
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def _json(self, status: int, payload: Dict[str, Any], origin: Optional[str] = None) -> None:
            self.send_response(status)
            self._cors(origin)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode("utf-8"))

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self._cors(self.headers.get("Origin"))
            self.end_headers()

        def do_POST(self) -> None:  # noqa: N802
            origin = self.headers.get("Origin")
            if urlparse(self.path).path != "/callback":
                self._json(404, {"success": False, "error": "not found"}, origin)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except Exception as exc:
                self._json(400, {"success": False, "error": str(exc)}, origin)
                return
            if not isinstance(payload, dict) or payload.get("state") != expected_state:
                result["error"] = "OAuth state mismatch"
                self._json(400, {"success": False, "error": "state mismatch"}, origin)
                return
            token = str(payload.get("apiKey") or "").strip()
            if not token:
                result["error"] = "missing apiKey"
                self._json(400, {"success": False, "error": "missing apiKey"}, origin)
                return
            result["payload"] = payload
            self._json(200, {"success": True}, origin)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

    return _Handler, result


def _wait_for_callback(state: str, *, port: int = CALLBACK_PORT, timeout: float = 120.0) -> Dict[str, Any]:
    handler_cls, result = _callback_handler(state)

    class _Server(HTTPServer):
        allow_reuse_address = True

    server: Optional[HTTPServer] = None
    try:
        server = _Server(("127.0.0.1", port), handler_cls)
    except OSError:
        # 5959 busy: the studio page is told the callback URL, so an ephemeral port is fine.
        try:
            server = _Server(("127.0.0.1", 0), handler_cls)
        except OSError as exc:
            raise CommandCodeAuthError(f"Could not bind the Command Code callback server: {exc}") from exc
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    thread.start()
    deadline = time.monotonic() + max(5.0, timeout)
    try:
        while time.monotonic() < deadline:
            if result["payload"] or result["error"]:
                if result["error"]:
                    raise CommandCodeAuthError(f"Command Code sign-in failed: {result['error']}")
                return result["payload"]
            time.sleep(0.1)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)
    raise CommandCodeAuthError("Timed out waiting for the browser sign-in callback.")


def login(*, open_browser: bool = True, timeout: float = 120.0) -> Dict[str, Any]:
    """Import the CLI's grant, or fall back to the studio's loopback hand-off."""
    try:
        token = read_cli_token()
        identity = validate(token)
        if identity is not None:
            logger.debug("using the Command Code CLI grant for %s", identity.get("user_name") or "user")
            return {"token": token, "source": "commandcode-cli", **identity}
    except CommandCodeAuthError as exc:
        logger.debug("CLI grant unusable, falling back to browser: %s", exc)

    state = secrets.token_urlsafe(32)
    callback = f"http://127.0.0.1:{CALLBACK_PORT}/callback"
    auth_url = f"{STUDIO_URL}/studio/auth/cli?callback={callback}&state={state}"
    print("Sign in with Command Code in your browser:")
    print(f"  {auth_url}\n")
    print("Waiting for the sign-in callback…")
    if open_browser:
        try:
            import webbrowser

            webbrowser.open(auth_url)
        except Exception:
            logger.debug("could not open a browser; paste the URL above")
    payload = _wait_for_callback(state, timeout=timeout)
    token = str(payload["apiKey"]).strip()
    user_id = str(payload.get("userId") or "")
    user_name = str(payload.get("userName") or "")
    try:
        write_cli_token(token, user_id=user_id, user_name=user_name,
                        key_name=str(payload.get("keyName") or "hermes-plugin"))
    except Exception as exc:  # the pool row still matters more than the shared file
        logger.debug("could not write %s back: %s", CLI_AUTH_PATH, exc)
    return {"token": token, "source": "commandcode-studio", "user_id": user_id, "user_name": user_name}


# ── ProviderProfile hooks ────────────────────────────────────────────────────

def auth_handler(action: str, args: Any) -> bool:
    """Own ``hermes auth add|status|logout <provider>``; decline ``refresh`` to the pool."""
    from agent.credential_pool import AUTH_TYPE_OAUTH, PooledCredential, load_pool

    provider = pool_provider(args)
    pool = load_pool(provider)

    if action == "add":
        creds = login(open_browser=not getattr(args, "no_browser", False),
                      timeout=float(getattr(args, "timeout", None) or 120.0))
        entry = pool.add_entry(PooledCredential(
            provider=provider,
            id=uuid.uuid4().hex[:6],
            label=creds.get("user_name") or creds.get("user_id") or "command code",
            auth_type=AUTH_TYPE_OAUTH,
            priority=0,
            source=POOL_SOURCE,
            access_token=creds["token"],
            # The CLI grant has no expiry the pool can trust, so leave it unset:
            # _is_usable() treats a missing expiry as usable and a 401 still parks the row.
            extra={"commandcode": {"source": creds.get("source", ""), "user_name": creds.get("user_name", "")}},
        ))
        who = creds.get("user_name") or creds.get("user_id") or "your account"
        print(f"Signed in to Command Code as {who}; credential {entry.id} added to the pool.")
        return True

    if action == "status":
        entries = pool.entries()
        if not entries:
            print(f"{provider}: logged out (no credential pool entries)")
            return True
        token = str(entries[0].access_token or "").strip()
        identity = validate(token)
        who = (identity or {}).get("user_name") or (identity or {}).get("user_id") or "unknown"
        where = f" ({CLI_AUTH_PATH})" if CLI_AUTH_PATH.exists() else ""
        if identity is None:
            print(f"{provider}: credential present but Command Code rejected it — run `hermes auth add {provider}`")
        else:
            print(f"{provider}: logged in as {who}{where}\n  auth_type: oauth (commandcode cli grant)")
            print(f"  credentials: {len(entries)}")
        return True

    if action == "logout":
        count = len(pool.entries())
        for index in range(count, 0, -1):
            pool.remove_index(index)
        print(f"Logged out of {provider} ({count} credential(s) removed)")
        return True

    # ``refresh`` (and anything new) stays with the shell: pooled rows rotate through
    # refresh_credential() below.
    return False


def refresh_credential(entry: Any) -> Mapping[str, Any]:
    """Re-read the CLI's grant for a pooled row.

    There is no refresh_token grant to spend: the CLI file is the source of truth, so a
    rotation is "the file changed". Returning the same token is a successful no-op, which
    keeps the pool from parking a row that is in fact still valid.
    """
    token = read_cli_token()
    identity = validate(token)
    if identity is None:
        raise CommandCodeAuthError(
            "Command Code rejected the stored credential; run `hermes auth add commandcode-alpha` again."
        )
    return {
        "access_token": token,
        "last_refresh": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
