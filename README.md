# hermes-commandcode-oauth

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) model-provider plugin that adds
Command Code (`commandcode.ai`) as a provider for accounts that sign in with the Command Code CLI.

Hermes ships an in-tree `commandcode` API-key provider against the documented OpenAI-compatible Provider API (`/provider/v1`).
Accounts that sign in through the CLI are not covered by it; this plugin covers them, using the
same `/alpha/generate` endpoint the CLI uses and the sign-in it already stores.

## What it provides

| Hook | Behaviour |
|---|---|
| `register_provider` | provider `commandcode-oauth` (alias `commandcode-alpha`) |
| `create_client` | the `/alpha/generate` transport, OpenAI-shaped: sync calls, streaming chunks, and awaitable for the async auxiliary path |
| `auth_handler`, `refresh_credential` | `hermes auth add\|status\|logout commandcode-oauth`: reuses the CLI's existing sign-in, or the studio hand-off, and keeps the credential in the pool |
| `fetch_models` | the account's live catalog |
| `setup_status`, `discover_models` | verified login status with a login command, and live model rows from the CLI key or Hermes credential pool |
| `fetch_account_usage` | plan limits: credits, 5-hour and weekly windows, current-period spend |
| `classify_api_error` | maps endpoint failures onto Hermes' failover reasons (billing, rate limit, auth, server) |

Behaviour worth knowing, all covered by tests:

- prompt-cache usage (`cacheReadTokens`) is reported to the caller, so cache hits are visible;
- image parts carry `mimeType` — the endpoint accepts a part without it and ignores the pixels;
- models that do not take images get a text placeholder instead of losing the attachment;
- an empty or truncated stream, and an in-stream `error` event, fail closed rather than returning
  an empty success;
- `tool_choice: "none"` empties the tool list; `temperature` is forwarded.

## Install

Use a Hermes checkout containing the generic plugin OAuth auxiliary routing fix
[`f12e6b66`](https://github.com/NousResearch/hermes-agent/commit/f12e6b66d960a92d9d4779cfcc628ab9cb02555b)
(merged on October 6, 2026), or a later release that includes it. The September 24 release
predates this fix: declaring `oauth_external` there makes title generation, compression and
vision routing unavailable even though main chat works. This plugin extends the current core
through its provider hooks; it does not patch older cores.

```bash
# straight from the repository
hermes plugins install NanamiMio/hermes-commandcode-oauth

# ...or by hand, as a user-level plugin
cp -r . ~/.hermes/plugins/model-providers/commandcode-oauth/

# then
hermes auth add commandcode-oauth    # reuses the CLI sign-in, or opens the studio hand-off
hermes model                         # pick commandcode-oauth, then a model
```

A catalog entry is pending review upstream; once it lands, `hermes plugins search commandcode`
finds it and `hermes plugins install commandcode-oauth` works.

## Notes

- **Naming**: `commandcode-oauth` is the historical name; `commandcode-alpha` is kept as an alias.
  There is no OAuth exchange or refresh grant: authentication imports a static API key from the
  CLI file or receives one through the browser hand-off.
- **Setup**: `auth_type="oauth_external"` uses Hermes' generic login gate instead of prompting
  for an API key in `hermes model`. Run `hermes auth add commandcode-oauth` first; the model setup
  flow then fetches the live account catalog. `COMMANDCODE_CLI_TOKEN` is an optional key override
  that the auth command can import into the Hermes pool.
- **Picker compatibility**: on Hermes releases that probe live catalogs only for `api_key`
  profiles, `/model` and the Desktop picker use `fallback_models`; the `hermes model` setup flow
  still fetches live models. [Upstream PR #122203](https://github.com/NousResearch/hermes-agent/pull/122203)
  extends live probing to non-`api_key` profiles and is currently pending.
- **Credentials**: the plugin only reads `~/.commandcode/auth.json`; it never writes or refreshes
  the vendor CLI's store. Browser keys are saved only in Hermes' credential pool. A busy callback
  port (5959) fails immediately with instructions; the listener is bound before the URL is shown
  and closes after success, failure, timeout or cancellation.
- **`fallback_models` is an offline fallback**, not the catalog — the catalog is fetched live.
- **Vision** travels through the same endpoint; verified against a solid-colour test image.

## Disclosure

The plugin calls the CLI-internal `https://api.commandcode.ai/alpha/generate` endpoint with the
Command Code CLI's client headers, rather than the documented Provider API. Each inference call
sends the full conversation (including system prompts and attachments), tool schemas, cwd path,
and OS/architecture/Python version to that service. Catalog, login validation and account usage
also make authenticated requests to Command Code. A configured `base_url` changes the API origin.
Use Hermes' bundled `commandcode` provider if you want the documented Provider API instead.

## Tests

```bash
python -m unittest discover -s tests   # stdlib only, no network (local NDJSON server)

# Also exercise the real Hermes registry, pool, model setup and async dispatch:
HERMES_TREE=/path/to/hermes-agent HERMES_HOME=/path/to/temporary-test-home \
  python -m unittest discover -s tests -v
```

Hermes integration tests need its normal runtime dependencies. All provider traffic in the suite
is mocked or served by a local fixture; the tests do not contact a live Command Code account.
