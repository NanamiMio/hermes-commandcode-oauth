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
| `auth_handler`, `refresh_credential` | `hermes auth add\|status\|logout\|refresh commandcode-oauth`: import/sign in, report/remove credentials, or validate/reimport one static key without an OAuth refresh grant |
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

Version 0.1.3 also checks the provider contracts used by the desktop picker and recovery loop:
usage windows use Hermes' `AccountUsageWindow`, credentials respect the active profile,
error classifications include recovery hints, and screenshot tool results keep their media parts.
Custom `default_headers` and a supplied synchronous `httpx.Client` are used by sync, streaming
and async calls. A caller-supplied client remains the caller's responsibility to close.

## Install

The plugin currently declares `auth_type="api_key"` for compatibility with Hermes' live model
picker. Authentication still imports the Command Code CLI's static key or receives one through
the browser hand-off; inference still uses `/alpha/generate` through the plugin's own client.
This declaration does not switch requests to the vendor's Provider API.

```bash
# straight from the repository
hermes plugins install NanamiMio/hermes-commandcode-oauth

# ...or by hand, as a user-level plugin
cp -r . ~/.hermes/plugins/model-providers/commandcode-oauth/

# then
hermes auth add commandcode-oauth    # reuses the CLI sign-in, or opens the studio hand-off
hermes model                         # pick commandcode-oauth, then a model
```

The plugin is listed in Hermes' catalog: `hermes plugins search commandcode` finds it and
`hermes plugins install commandcode-oauth` installs the reviewed catalog SHA. New repository
commits reach catalog users only after a separate catalog pin update is reviewed and merged.

## Notes

- **Naming**: `commandcode-oauth` is the historical name; `commandcode-alpha` is kept as an alias.
  There is no OAuth exchange or refresh grant: authentication imports a static API key from the
  CLI file or receives one through the browser hand-off.
- **Setup**: run `hermes auth add commandcode-oauth` first. The provider's own auth handler
  imports the CLI key or opens the browser and saves the credential in Hermes' pool.
  `COMMANDCODE_CLI_TOKEN` remains an optional key override.
- **Picker compatibility**: keep `auth_type="api_key"` while
  [upstream PR #122203](https://github.com/NousResearch/hermes-agent/pull/122203) is pending,
  so `/model` and the Desktop picker can fetch live models on upstream Hermes.
  Switch to `oauth_external` only after a supported Hermes release includes both that picker
  change and the generic plugin OAuth auxiliary routing fix
  [`f12e6b66`](https://github.com/NousResearch/hermes-agent/commit/f12e6b66d960a92d9d4779cfcc628ab9cb02555b).
- **Credentials**: the plugin only reads `~/.commandcode/auth.json`; it never writes or refreshes
  the vendor CLI's store. Browser keys are saved only in Hermes' credential pool. A busy callback
  port (5959) fails immediately with instructions; the listener is bound before the URL is shown
  and closes after success, failure, timeout or cancellation.
- **Static-key refresh**: `hermes auth refresh commandcode-oauth [index-or-id-or-label]`
  validates a browser key, or reimports an updated CLI/environment key for the same account.
  With several credentials, specify the target. It preserves sibling credentials and does not
  manufacture an OAuth refresh token. An account change requires an explicit `hermes auth add`.
- **Profile isolation**: a served profile uses its scoped environment and own pool; it does not
  automatically inherit the host CLI key. Import that key explicitly with `hermes auth add`.
- **HTTP redirects**: stdlib requests strip sensitive headers on a different origin. A supplied
  HTTP client follows no redirects for generation requests, so its proxy/TLS policy is used
  without sending credentials to a redirect target.
- **`fallback_models` is an offline fallback**, not the catalog — the catalog is fetched live.
  Live discovery only reorders models returned by the service; it never inserts absent free
  tiers. The retired `meituan/LongCat-2.0:free` SKU is excluded from the offline fallback.
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

For release verification, run against a clean upstream `main` or supported release as well as
the deployed fork. Record both SHAs; a development branch carrying unmerged Hermes fixes does
not establish upstream compatibility. The contract regressions cover the actual usage cache
and picker, error classifier, static-key refresh command, scoped credentials and HTTP transport.

Separately verified on October 8, 2026 with a real CLI credential and Hermes main `08165d5`:
login import, live catalog, billing windows, a full Agent turn, sync and streaming chat, tool calls and results,
sync compression and async titling through Hermes' auxiliary router, and vision using a generated
blue PNG. DeepSeek V4.1 Flash handled text/tools and Qwen3.8 Omni Flash handled the image. The CLI
credential file remained unchanged. A live 403 for the retired LongCat free tier is classified as
a model failure rather than a credential failure.
Laguna S 2.1 Free also returned a valid reply after an initial provider-capacity error.
