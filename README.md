# hermes-opencode-free-provider

One Hermes provider (`opencode-free/<model>`) over **free backends** with
Hermes-local tools. Hermes stays the agent runtime — system prompt,
history, tools, execution, loop and conversation state. The provider only
talks to the model API.

No OpenCode CLI. No subprocess. No proxy. No core edits. No payment.

```text
Hermes Agent
→ provider (opencode-free/<model>)
→ BackendAdapter (pollinations → openrouter → groq)
→ API (/v1/chat/completions + SSE)
→ tool call → Hermes executes locally → result (same ID) → continue
```

## Backends (in fallback order)

| # | Backend | Base URL | Auth | Cost | Why |
|---|---------|----------|------|------|-----|
| 1 | **Pollinations** | `https://gen.pollinations.ai/v1` | `POLLINATIONS_API_KEY` (free signup at `enter.pollinations.ai/keys`, starter Pollen via Quests) | Pollen-metered, free entry, no card/deposit | Primary: official Hermes harness (`polli harness`), strong coding models, tools + streaming + reasoning + vision, 180 text models, public catalog |
| 2 | **OpenRouter** | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` (free, no card) | `:free` models are $0 (~20 req/min, ~50/day) | Fallback: OpenAI-compatible, tool calling + streaming |
| 3 | **Groq** | `https://api.groq.com/openai/v1` | `GROQ_API_KEY` (free, no card) | Free tier (~30 req/min, generous daily) | Fallback: very fast, Llama/Qwen instruct with tools |
| 4 | **Zen** (legacy) | `https://opencode.ai/zen/v1` | `OPENCODE_ZEN_API_KEY` only — anonymous is **blocked upstream** (`FreeTierError`) | Paid key | Opt-in only when you already have a key; never attempted anonymously |

Fallback tries the **same model string** on the next backend that serves
that exact id. A different model is never substituted silently — if none
serves it, the error lists tried backends and available ids.

Anonymous note (verified 2026-10-06): keyless `POST /v1/chat/completions`
returns 200 for a few plain-text models, but keyless + tools, keyless +
streaming, and keyless coding models return 401 (`Get one at
enter.pollinations.ai/keys`). Agent loops need tools, so they need a free
key. Nothing is bypassed — the server enforces this and the plugin
surfaces its message verbatim with signup guidance.

## Models

Dynamic discovery per backend (`GET /v1/models`, public on pollinations +
openrouter), merged, cached 6h, with curated fallback. Each model carries
explicit capabilities — `tool_calling, streaming, reasoning, vision,
context, endpoints, free/billed` — and the client **omits tools** for
models reporting no tool support instead of faking it.

Coding/agent picks on pollinations (all `tools: true` live).
Verified 2026-10-06 with real tool-calling loops; the first two run on
the free starter balance (0.25 Pollen), the rest need paid Pollen:

```text
opencode-free/inclusionai/ling-3.1-flash          # 262K ctx, default, free balance OK
opencode-free/openai/gpt-5.4-nano                # free balance OK
opencode-free/qwen/qwen3-coder-30b-a3b-instruct  # needs paid Pollen
opencode-free/qwen/qwen3-coder-next              # needs paid Pollen
opencode-free/moonshotai/kimi-k2.7-code          # needs paid Pollen
opencode-free/openai/gpt-5.3-codex               # needs paid Pollen
opencode-free/deepseek/deepseek-v4-flash         # needs paid Pollen
opencode-free/z-ai/glm-5.3                       # needs paid Pollen
opencode-free/xiaomi/mimo-v2.5                   # needs paid Pollen
```

OpenRouter `:free` (16 live, rotates — check `openrouter.ai/collections/free-models`):

```text
opencode-free/cohere/north-mini-code:free
opencode-free/poolside/laguna-s-2.1:free
opencode-free/nvidia/nemotron-3-ultra-550b-a55b:free
...
```

Nothing here claims "free" without verification: pollinations is
pollen-metered with free starter via Quests; OpenRouter `:free` is $0
within tight daily limits; Groq is free-tier rate-limited. Check each
service's current terms before sending confidential data.

## Install (Hermes 0.21.3+, Windows/Linux/macOS, stdlib only)

Via Git (repo root IS the plugin — `plugin.yaml` at top level):

```bash
hermes plugins install <owner>/hermes-opencode-free-provider --enable
# or pin a commit:
hermes plugins install <owner>/hermes-opencode-free-provider --ref <40-char-sha> --enable
```

Then:

```bash
hermes model   # Provider: opencode-free
```

From a local checkout:

```bash
hermes plugins install ./hermes-opencode-free-provider --enable
```

```bash
python -m unittest discover -s tests -v
hermes plugins validate . --json
hermes plugins doctor .
```

## Configuration

```yaml
opencode_free:
  enabled: true
  provider_name: opencode-free
  backend_order: [pollinations, openrouter, groq]  # zen only with key
  backend_fallback: true
  auto_discover: true
  cache_models: true
  transport: auto
  fallback_transport: true
  reasoning: auto
  streaming: true
  tool_translation: true
  request_timeout: 180
  retry: {enabled: true, max_attempts: 3}
```

Credentials — env vars only, never in code/README/tests/git:

```bash
export POLLINATIONS_API_KEY="sk-..."  # primary
export OPENROUTER_API_KEY="sk-or-..." # fallback
export GROQ_API_KEY="gsk-..."         # fallback
export OC_FREE_DEBUG=1
```

## Usage

```bash
hermes -z "list Python files and summarize" \
  --provider opencode-free -m inclusionai/ling-3.1-flash
```

Debug (redacted, never prints keys):

```bash
OC_FREE_DEBUG=1 hermes -z "refactor main.py" \
  --provider opencode-free -m moonshotai/kimi-k2.7-code
# MODEL / TRANSPORT / REQUEST / RESPONSE / TOOL_CALL / TOOL_RESULT /
# TOKENS / LATENCY / RETRY / ERROR
```

## Capabilities & limits

* System prompt: Hermes verbatim, zero injection, format-only adaptation.
* History: full chain preserved (system/user/assistant/tool/reasoning/IDs,
  multi-turn); oversized results truncated with explicit marker.
* Tool calling: names/IDs/args/order/parallelism/errors/multi-cycle kept;
  streamed JSON buffered until valid; tools omitted when unsupported.
* Streaming: SSE deltas for text/reasoning/tools/usage/finish separately.
* Reasoning: `reasoning_effort` + carriers preserved when advertised,
  clean degrade otherwise.
* Errors: 401/402/403/404/408/409/429/5xx + timeout/disconnect/malformed
  JSON classified; retry+backoff only when transient, never infinite;
  402 (budget) fails over to next backend; nothing masked as text.
* Limits: free quotas/rate limits per backend; anonymous Zen blocked;
  keyless agent loops impossible (server 401s tools) — create the free key.

## File layout

```text
plugin.yaml  __init__.py  provider.py  client.py  discovery.py
backends/base.py  backends/pollinations.py  backends/openrouter.py
backends/groq.py  backends/zen.py  backends/registry.py
tools_adapter.py  history.py  transports.py  protocol.py  models.py
config.py  errors.py  debug.py  tests/
```

## Terms

Independent, unofficial, no affiliation with OpenCode/Pollinations/
OpenRouter/Groq. Your own internal use on systems you control. Comply with
each service's Terms/privacy/acceptable-use; free means pricing, not
permission — prompts may train models. Never send secrets unless allowed.
No proxying for third parties, no resale, no quota evasion, no credential
bypass.
