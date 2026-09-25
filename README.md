# hermes-commandcode-oauth

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) model-provider plugin that adds
Command Code (`commandcode.ai`) as a provider for accounts that sign in with the Command Code CLI.

Upstream ships Command Code as an API-key provider against the OpenAI-compatible Provider API.
Accounts that sign in through the CLI are not covered by it; this plugin covers them, using the
same `/alpha/generate` endpoint the CLI uses and the sign-in it already stores.

## What it provides

| Hook | Behaviour |
|---|---|
| `register_provider` | provider `commandcode-oauth` (alias `commandcode-alpha`) |
| `create_client` | the `/alpha/generate` transport, OpenAI-shaped: sync calls, streaming chunks, and awaitable for the async auxiliary path |
| `auth_handler`, `refresh_credential` | `hermes auth add\|status\|logout commandcode-oauth`: reuses the CLI's existing sign-in, or the studio hand-off, and keeps the credential in the pool |
| `fetch_models` | the account's live catalog |
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

- **Naming**: `oauth` names the credential path; `commandcode-alpha` is kept as an alias.
- **`auth_type="oauth_external"`**: truthfully represents the CLI OAuth / browser loopback sign-in.
  With Hermes Agent upstream PR #122203, non-`api_key` providers with a custom `fetch_models()` are
  probed live, and `COMMANDCODE_CLI_TOKEN` remains supported as an optional environment override.
- **`fallback_models` is an offline fallback**, not the catalog — the catalog is fetched live.
- **Vision** travels through the same endpoint; verified against a solid-colour test image.

## Tests

```bash
python -m unittest discover -s tests   # stdlib only, no network (local NDJSON server)
```
