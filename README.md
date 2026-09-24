# commandcode-alpha — Hermes model-provider plugin

Command Code (`commandcode.ai`) accounts on the **Go / free tiers** reach models through the
CLI's private `/alpha/generate` protocol, not through the OpenAI-compatible Provider API that
Hermes' bundled `commandcode` profile targets. This plugin supplies that transport in-process.

## Why a plugin, not a core PR

[#105967](https://github.com/NousResearch/hermes-agent/issues/105967) asked for this in core and
the maintainer ruling was:

> a third-party-product integration on an undocumented protocol … our recommendation is to ship
> it as an external model-provider plugin so it can move at Command Code's pace.

[PR #105968](https://github.com/NousResearch/hermes-agent/pull/105968) is the core patch that
this replaces. Nothing here touches Hermes core: it uses the documented provider-plugin hooks
(`create_client`, `auth_handler`, `refresh_credential`, `fetch_models`, `fetch_account_usage`,
`classify_api_error`) and the `HERMES_SKIP_TRANSPORT_WRAP` / `HERMES_SKIP_ASYNC_WRAP` opt-outs.

## What it provides

| Hook | Behaviour |
|---|---|
| `create_client` | `/alpha/generate` transport behind an OpenAI-shaped facade. Honours a user-set `base_url` (proxy/self-hosted). Chunk streaming for `stream=True`, complete response otherwise. |
| `auth_handler` | Owns `hermes auth add\|status\|logout commandcode-alpha`: imports the grant the official CLI already stores in `~/.commandcode/auth.json`, or signs in through the studio's loopback hand-off (127.0.0.1:5959), writing the grant back so CLI and Hermes share one login. |
| `refresh_credential` | Re-reads the CLI grant for the pooled row. |
| `fetch_models` | Live catalog from `/provider/v1/models`, with the free models surfaced first. |
| `fetch_account_usage` | `hermes usage` / `/usage`: credits, 5-hour and weekly windows, period spend (`/alpha/whoami` → `/alpha/billing/credits` + `/alpha/usage/summary`). |
| `classify_api_error` | 402 → `billing`, 401/403 → `auth`, 429 → `rate_limit`, 5xx → `server_error`/`overloaded`, 504 → `timeout`, 400 → `context_overflow`/`format_error`. |

### Wire details worth knowing

* `totalUsage.inputTokenDetails.cacheReadTokens` is the cached share of the prompt. It is
  billed separately (`cacheCost` on the account's usage page) and is mapped onto
  `prompt_tokens_details.cached_tokens`, so Hermes reports real cache hits — measured 97.7% on
  a live call.
* The relay can answer **HTTP 200 with an `error` event inside the stream** (e.g. 402
  `Insufficient Balance`). That is raised as a `CommandCodeAPIError` carrying its status code,
  so failover can classify it.
* A stream that ends after `start` with no text and no `finish` is **failed closed** rather than
  returned as an empty turn.
* `tool_choice="none"` empties the tool list; a forced tool choice cannot be expressed on this
  wire, so tools are forwarded instead of failing the turn.

## Install

```bash
# user-level (fastest loop) — the plugin overrides any same-named bundled profile
cp -R commandcode-alpha ~/.hermes/plugins/model-providers/

# or as a distribution
#   [project.entry-points."hermes_agent.plugins"]
#   commandcode-alpha = "hermes_commandcode_alpha:register"
```

Then pick it like any provider (`hermes model`, or `model.provider: commandcode-oauth` — the
alias is kept for the name the PR and existing configs use).

## Naming

Canonical name is `commandcode-alpha` (upstream's product+wire convention: `commandcode` for the
Provider API, `commandcode-anthropic` for the Anthropic wire). `commandcode-oauth` is kept as the
single alias because that is what the integration PR/issue and existing configs call it. If
Command Code ever unifies the transports, fold this into `commandcode` and drop the plugin.

## Tests

```bash
python -m unittest discover -s tests -v   # or: pytest tests -q
```

18 tests, no network: the transport is pointed at a local NDJSON server through `base_url`.
`pool_provider` needs a Hermes tree on `sys.path`; set `HERMES_TREE` to run that case.

## Deliberate omissions

* **No `get_usage_cost`.** The vendor's per-model prices are not exposed on any endpoint this
  plugin can read, and a stale hardcoded price table would be worse than none. `hermes usage`
  (invoice-side: credits + period spend) and the account's usage page are the cost sources.
* **No browser-less login.** `hermes auth add` needs either the CLI's grant or a browser.

This is a third-party integration on an **undocumented** endpoint that Command Code may change at
any time; it is not endorsed by Command Code or Nous Research.
