"""Check that validator/model.py reaches the validator model through OpenRouter.

Unlike test_api.py, which calls OpenRouter directly, this goes through the
adapter: `complete()` must pick the OpenRouter backend, resolve the model and
provider pin from openrouter.toml, and return the response as a dict. The config
tests are free; the `openrouter`-marked test is billable and deselected by default:

    uv run pytest tests/openrouter/test_validator.py                 # free tests
    uv run pytest -m openrouter tests/openrouter/test_validator.py   # billable test
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

from validator import model


def test_config(openrouter_config: dict[str, Any]) -> None:
    """
    The validator alias resolves to openrouter.toml's [validator] entry.
    """

    assert model._openrouter_backend() == openrouter_config["validator"]


def test_unknown_alias_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A VALIDATOR_MODEL that openrouter.toml doesn't know fails loudly rather than
    silently falling back to some model.
    """

    monkeypatch.setattr(model, "MODEL", "not-a-validator")
    with pytest.raises(RuntimeError, match="has no .* section"):
        model._openrouter_backend()


def _stub_client(monkeypatch: pytest.MonkeyPatch, prompt_tokens: int) -> None:
    # Replace the OpenRouter client with one returning a canned usage count, so
    # the context check runs without a request.
    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=prompt_tokens), model_dump=lambda: {}
    )
    completions = SimpleNamespace(create=lambda **request: response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(model, "BACKEND", "openrouter")
    monkeypatch.setattr(model, "_openrouter_client", lambda: client)


def test_over_modal_context_logs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """
    A request that would overflow the Modal validator's context logs a warning.
    """

    _stub_client(monkeypatch, prompt_tokens=model.MODAL_MAX_MODEL_LEN)
    with caplog.at_level(logging.WARNING, logger=model.__name__):
        model.complete([], max_tokens=1)
    assert [record.levelno for record in caplog.records] == [logging.WARNING]


def test_within_modal_context_is_silent(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """
    A request that fits the Modal validator's context logs nothing.
    """

    _stub_client(monkeypatch, prompt_tokens=model.MODAL_MAX_MODEL_LEN - 1)
    with caplog.at_level(logging.WARNING, logger=model.__name__):
        model.complete([], max_tokens=1)
    assert not caplog.records


@pytest.mark.openrouter
def test_complete(
    monkeypatch: pytest.MonkeyPatch,
    openrouter_config: dict[str, Any],
    chart_message: dict[str, Any],
    image_answer: str,
) -> None:
    """
    complete() sends a chart image via the pinned provider and returns a dict
    with the right answer.
    """

    # Exercise the OpenRouter path whatever VALIDATOR_BACKEND says in .env.
    monkeypatch.setattr(model, "BACKEND", "openrouter")
    response = model.complete([chart_message], max_tokens=50)
    assert isinstance(response, dict), f"expected a dict, got {type(response).__name__}"

    pinned = openrouter_config["validator"]["provider"]["order"][0]
    assert response.get("provider") == pinned, (
        f"served by {response.get('provider')!r}, expected pinned provider {pinned!r}"
    )

    answer = (response["choices"][0]["message"]["content"] or "").strip()
    assert image_answer in answer.lower(), f"expected {image_answer!r}, got {answer!r}"
