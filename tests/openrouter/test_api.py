"""Check that the OpenRouter models in backends.toml can stand in for Modal.

Calls OpenRouter directly with each role's model and provider pin. Billable, so
every test here carries the `openrouter` marker and is deselected by default:

    uv run pytest -m openrouter tests/openrouter/test_api.py [-k validator|small|medium|large]
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from openai import OpenAI

pytestmark = pytest.mark.openrouter

# Same limit the agent runner uses (infrastructure/agent_runner.py).
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


def _create(client: OpenAI, backend: dict[str, Any], **request: Any):
    response = client.chat.completions.create(
        model=backend["openrouter_model"],
        temperature=0,
        extra_body={"provider": backend["openrouter_provider"]},
        **request,
    )
    assert response.choices, f"no choices in response: {response.model_dump()}"
    served_by = (response.model_extra or {}).get("provider", "?")
    print(f"served by {served_by}")
    return response.choices[0].message


def test_validator(
    client: OpenAI, backends: dict[str, Any], chart_message: dict[str, Any], image_answer: str
) -> None:
    """
    The pinned provider receives a chart image and names its tallest bar.
    """

    message = _create(client, backends["validator"], max_tokens=50, messages=[chart_message])
    answer = (message.content or "").strip()
    assert image_answer in answer.lower(), f"expected {image_answer!r}, got {answer!r}"


def test_agent(client: OpenAI, agent: dict[str, Any]) -> None:
    """
    The model makes a structured bash tool call, then accepts the tool result in
    a second turn.
    """

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Use the bash tool to run `echo ok`."}
    ]

    # Turn 1: the model must return a structured tool call, not text.
    message = _create(
        client, agent, max_tokens=MODEL_MAX_TOKENS, messages=messages, tools=[BASH_TOOL]
    )
    assert message.tool_calls, f"turn 1: no tool_calls; content was {message.content!r}"
    call = message.tool_calls[0]
    try:
        arguments = json.loads(call.function.arguments)
    except json.JSONDecodeError:
        pytest.fail(f"turn 1: arguments are not JSON: {call.function.arguments!r}")
    assert call.function.name == "bash" and "command" in arguments, (
        f"turn 1: unexpected call {call.function.name}({arguments})"
    )

    # Turn 2: replay the assistant message as returned (including any reasoning
    # fields) plus the tool result, and the provider must accept the history.
    messages.append(message.model_dump(exclude_none=True))
    messages.append({"role": "tool", "tool_call_id": call.id, "content": "ok"})
    message = _create(
        client, agent, max_tokens=MODEL_MAX_TOKENS, messages=messages, tools=[BASH_TOOL]
    )
    assert message.content or message.tool_calls, "turn 2: empty response"
