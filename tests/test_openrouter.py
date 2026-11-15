"""Check that the OpenRouter models in backends.toml can stand in for Modal.

Calls OpenRouter directly with each role's model and provider pin. This is
billable, so it only runs when invoked explicitly; the checks are deliberately
not named test_* so pytest never collects them.

    uv run python -m tests.test_openrouter [validator] [small] [medium] [large]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import tomllib
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from openai import OpenAI

RELEASE = Path(__file__).resolve().parents[1]
BACKENDS = RELEASE / "infrastructure" / "backends.toml"

# matplotlib's "Bar color demo" for the version this repo pins. The tallest bar
# is blueberry (100); apple 40, cherry 30, orange 55.
IMAGE_URL = "https://matplotlib.org/3.10.5/_images/sphx_glr_bar_colors_001.png"
IMAGE_ANSWER = "blueberry"

# Same limits the agent runner uses (infrastructure/agent_runner.py).
MODEL_MAX_TOKENS = 4096

# Minimal stand-in for mini-swe-agent's BASH_TOOL.
BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command and return its output.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The command to run."}
            },
            "required": ["command"],
        },
    },
}


class CheckFailed(Exception):
    pass


def _client() -> OpenAI:
    base_url = os.getenv("OPENROUTER_BASE_URL")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not base_url or not api_key:
        raise SystemExit("missing OPENROUTER_BASE_URL or OPENROUTER_API_KEY in .env")
    return OpenAI(base_url=base_url, api_key=api_key, timeout=300)


def _create(client: OpenAI, backend: dict[str, Any], **request: Any):
    response = client.chat.completions.create(
        model=backend["openrouter_model"],
        temperature=0,
        extra_body={"provider": backend["openrouter_provider"]},
        **request,
    )
    if not response.choices:
        raise CheckFailed(f"no choices in response: {response.model_dump()}")
    served_by = (response.model_extra or {}).get("provider", "?")
    print(f"    served by {served_by}")
    return response.choices[0].message


def check_validator(client: OpenAI, backend: dict[str, Any]) -> None:
    request = Request(IMAGE_URL, headers={"User-Agent": "hw2-openrouter-check"})
    with urlopen(request, timeout=30) as response:
        encoded = base64.b64encode(response.read()).decode()
    message = _create(
        client,
        backend,
        max_tokens=50,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Which fruit has the tallest bar? Answer in one word."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
            ],
        }],
    )
    answer = (message.content or "").strip()
    if IMAGE_ANSWER not in answer.lower():
        raise CheckFailed(f"expected {IMAGE_ANSWER!r}, got {answer!r}")
    print(f"    answer: {answer!r}")


def check_agent(client: OpenAI, backend: dict[str, Any]) -> None:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Use the bash tool to run `echo ok`."}
    ]

    # Turn 1: the model must return a structured tool call, not text.
    message = _create(
        client, backend, max_tokens=MODEL_MAX_TOKENS, messages=messages, tools=[BASH_TOOL]
    )
    if not message.tool_calls:
        raise CheckFailed(f"turn 1: no tool_calls; content was {message.content!r}")
    call = message.tool_calls[0]
    try:
        arguments = json.loads(call.function.arguments)
    except json.JSONDecodeError as error:
        raise CheckFailed(f"turn 1: arguments are not JSON: {call.function.arguments!r}") from error
    if call.function.name != "bash" or "command" not in arguments:
        raise CheckFailed(f"turn 1: unexpected call {call.function.name}({arguments})")
    print(f"    turn 1: bash({arguments['command']!r})")

    # Turn 2: replay the assistant message as returned (including any reasoning
    # fields) plus the tool result, and the provider must accept the history.
    messages.append(message.model_dump(exclude_none=True))
    messages.append({"role": "tool", "tool_call_id": call.id, "content": "ok"})
    message = _create(
        client, backend, max_tokens=MODEL_MAX_TOKENS, messages=messages, tools=[BASH_TOOL]
    )
    if not (message.content or message.tool_calls):
        raise CheckFailed("turn 2: empty response")
    print(f"    turn 2: {'tool call' if message.tool_calls else repr(message.content[:80])}")


def main() -> None:
    load_dotenv()
    backends = tomllib.loads(BACKENDS.read_text())
    roles = ["validator", *backends["agents"]]

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # No default: argparse would check a list default against `choices` as one value.
    parser.add_argument("roles", nargs="*", choices=roles)
    arguments = parser.parse_args()

    client = _client()
    failures = 0
    for role in arguments.roles or roles:
        backend = backends["validator"] if role == "validator" else backends["agents"][role]
        check = check_validator if role == "validator" else check_agent
        provider = ", ".join(backend["openrouter_provider"]["order"])
        print(f"{role}: {backend['openrouter_model']} via {provider}")
        try:
            check(client, backend)
        except Exception as error:  # report every role, not just the first failure
            failures += 1
            print(f"  FAIL: {error}")
        else:
            print("  PASS")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
