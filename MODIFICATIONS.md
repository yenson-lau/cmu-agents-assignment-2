# Modifications

## OpenRouter replacement

### Existing setup

| Role | Model | Alias | Modal GPU | Used in |
| --- | --- | --- | --- | --- |
| Validator (judge) | `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8` | validator | L40S | Parts 1–4: every validator call |
| Agent (S) | `Qwen/Qwen2.5-Coder-3B-Instruct` | qwen | L4 | Part 3: workflow generate |
| Agent (M) | `mistralai/Ministral-3-14B-Instruct-2512` | ministral | L40S | Part 3 |
| Agent (L) | `zai-org/GLM-4.7-Flash` | glm-4.7 | H100 | Part 3 |

Required context length: 32,768


### Candidate models

| Role | Original | Replacement | Rationale |
| --- | --- | --- | --- |
| Validator (judge) | `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8` | `qwen/qwen3-vl-30b-a3b-instruct` (DeepInfra, FP8) | Same model at the same quantization, with image input |
| Agent (S) | `Qwen/Qwen2.5-Coder-3B-Instruct` | `meta-llama/llama-3.1-8b-instruct` (Groq) | Original has no tool-capable provider; only viable small model from a third lab, and likely still the weakest agent |
| Agent (M) | `mistralai/Ministral-3-14B-Instruct-2512` | `mistralai/ministral-14b-2512` (Mistral) | Same model, with `tools` and `tool_choice` support |
| Agent (L) | `zai-org/GLM-4.7-Flash` | `z-ai/glm-4.7-flash` (Novita, BF16) | Same model, with `tools` and reasoning matching the original vLLM setup |

#### Validator: `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8`

- **`qwen/qwen3-vl-30b-a3b-instruct` (262K context)**
  - Near exact replacement
  - Can fix provider to DeepInfra for FP8 quantization.
  - ✅ Same model as the original at the same FP8 quantization, with image input, so validator results stay comparable to the original setup


#### Agent (S): `Qwen/Qwen2.5-Coder-3B-Instruct`

- **`mistralai/ministral-3b-2512` (131K context)**
  - No quantization options
  - Different model family and overlaps with `Ministral-3-14B-Instruct-2512`
  - Released in November 2025 vs. April 2025 for Qwen 2.5

- **`meta-llama/llama-3.2-3b-instruct` (131K context)**
  - Can fix provider to Parasail for guaranteed BF16 quantization
  - Different model family to both Qwen and Mistral
  - Released in September 2024 vs. April 2025 for Qwen 2.5, so might have significant loss in tool calling abilities
  - ❌ No provider supports the `tools` parameter (Parasail, Cloudflare), so the agent cannot call its bash tool

- **`google/gemma-3-4b-instruct` (131K context)**
  - No quantization options
  - Different model family to both Qwen and Mistral
  - Released in September 2024 vs. April 2025 for Qwen 2.5, so might have significant loss in tool calling abilities
  - ❌ No provider supports the `tools` parameter (Parasail, DeepInfra), so the agent cannot call its bash tool

- **`meta-llama/llama-3.1-8b-instruct` (131K context)**
  - Can fix provider to Groq (quantization unknown). CoreWeave offers BF16 but its tool calls are broken for this model: arguments come back duplicated (`{"command": "echo ok"}{"command": "echo ok"}`), which fails `tests/test_openrouter.py`
  - Different model family to both Qwen and Mistral
  - 8B vs. 3B parameters, so likely somewhat stronger than the original
  - Released in July 2024 vs. November 2024 for Qwen 2.5 Coder, so tool calling is likely unreliable, which suits the weakest agent slot
  - ✅ Only viable small candidate from a third lab: keeps the agents to three different families and stays clearly the weakest of the three; pinned to Groq, the only provider whose tool calls pass the smoke test

#### Agent (M): `mistralai/Ministral-3-14B-Instruct-2512`

- **`mistralai/ministral-14b-2512` (262K context)**
  - Near exact replacement
  - No quantization options
  - Overlaps with `Ministral-3-14B-Instruct-2512`
  - ✅ Same model as the original, served by Mistral with `tools` and `tool_choice` support, so the medium agent is unchanged

- **`qwen/qwen3-14b` (41K context)**
  - Can fix provider to NextBit for 4-bit quantization or DeepInfra for FP8 quantization
  - Can use to swap with the model family against Agent (S), so it can use Ministral instead.
  - Released in April 2025 vs. November 2025 for Ministral

- **`google/gemma-3-12b-instruct` (131K context)**
  - Can fix provider to NextBit for 4-bit quantization or DeepInfra for BF16 quantization
  - Different model family to both Qwen and Mistral
  - Released in Mar 2025 vs. December 2025 for Ministral, so might have some loss in tool calling abilities

#### Agent (L): `zai-org/GLM-4.7-Flash`

- **`z-ai/glm-4.7-flash` (200K context)**
  - Near exact replacement
  - No recommended quantization options
  - ✅ Same model as the original; pinning Novita gives BF16 with `tools` and reasoning, matching the original vLLM setup (`glm47` tool parser, `glm45` reasoning parser)


### Infrastructure replacements

#### Where to build the adapters? (Claude)

There are two natural seams, one for each kind of model call. An adapter at each, plus one config source feeding both, covers everything without touching the scaffold, the runner or the workflow logic.

```text
               ┌───────────── one backend config ──────────────┐
               ▼                                                ▼
validator/solution.py, baseline.py            workflow generate → agent_runner (Modal)
        │                                                       │
        ▼                                                       ▼
 validator/model.py  ◄── seam 1                infrastructure/generation_gateway.py ◄── seam 2
 complete(), _client()                          complete(request)
        │                                                       │
        ▼                                                       ▼
 Modal vLLM  or  OpenRouter                      Modal vLLM  or  OpenRouter
```

##### Seam 1: `validator/model.py`, for the validator

Every judge call goes through `complete()` (line 38). `solution.py` and `baseline.py` both import from this file, and nothing else talks to the endpoint. The adapter here:

- Backend-specific request options: add them in `complete()`, e.g. `extra_body={"provider": {...}}` for OpenRouter, read from the environment.
- Tokenize calls: `tokenize()` and `detokenize()` (lines 57-64) only work on vLLM. Have them raise a clear "not supported on this backend" error, or fall back to a character-based estimate. That keeps `baseline.py` unchanged.

##### Seam 2: `infrastructure/generation_gateway.py`, for the agents

Every agent call from the runner goes through `complete(request)` (line 113), and the gateway already maps a model name to an endpoint (`_endpoint()`, line 24). Extend each registry entry from `{api_base, api_key}` to also carry:

- `model`: the name to send upstream, defaulting to the requested name.
- `extra`: backend-specific fields merged into the payload, such as `provider`.

That's about five lines in `complete()`. The gateway only receives settings through the Modal secret (its image has only `requests`, with no local source mounted), so the adapter there should consist of settings rather than imported code.

##### Config source: `infrastructure/configure_generation.py`

`endpoint_registry()` (line 17) builds the secret from `AGENTS` and `.env`. Make it read the per-agent `model` and `extra` fields too, either from `.env` or from a small checked-in file such as `infrastructure/backends.toml`. The validator side can read the same file, so both seams share one description of "which backend, which upstream model, which provider pin."

##### Decision: what counts as the agent's identity

`AGENTS[].model` serves as both the routing key and the identity recorded in `result.json` and checked by `check-submission` (`workflow/submission.py:171-177`). There are two options:

- Keep the original Hugging Face names as identities and let the adapter map them upstream. No workflow changes, but runs say "Qwen2.5-Coder-3B" when Llama produced them.
- Change `AGENTS` to the real models (three lines in `workflow/generate.py`). The adapter then only adds the provider pin, and the records stay honest. This is the recommended option.

##### Leave alone

`infrastructure/agent_runner.py` (the scaffold, plus the dead Qwen-only branch), `workflow/` apart from `AGENTS`, and the vLLM serving files (`validator_model.py`, `generation_models.py`, `serving.py`). They still work as-is when switching back to Modal.

#### OpenRouter tests

The tests need to validate that the models in `backends.toml` can be relied on to replace the functionality in the original Modal setup. They are located in `tests/test_openrouter.py`.

They call OpenRouter directly (`OPENROUTER_BASE_URL` and `OPENROUTER_API_KEY` from `.env`), using each role's `openrouter_model` and `openrouter_provider` from `backends.toml`. They test the models and providers only, not the adapter or the Modal gateway.

- **Validator:** asked "Which fruit has the tallest bar?" about [matplotlib's bar color demo](https://matplotlib.org/3.10.5/_images/sphx_glr_bar_colors_001.png) (sent as base64, like `validator/baseline.py`), the answer contains "blueberry". This confirms the pinned provider actually receives the image.
- **Each agent, turn 1:** with a minimal bash tool and the prompt "run `echo ok`", the response contains a real `tool_calls` entry with valid JSON arguments, not a tool call written as text.
- **Each agent, turn 2:** sending back the assistant tool call plus a matching `tool` message (`tool_call_id`, content `ok`) is accepted and gets a response. This catches multi-turn issues such as Mistral's tool-call ID and ordering rules and GLM's reasoning field.

Notes:

- Run with `uv run python -m tests.test_openrouter [validator] [small] [medium] [large]` (all roles by default).
- Billable (fractions of a cent per run), so it must never run as part of a default test command. Ask before running. The checks are not named `test_*`, so pytest won't collect them.
