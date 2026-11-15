"""The judge model, reached through an OpenAI-compatible endpoint.

`complete` takes chat messages and returns the whole response, so a question can
be one turn or several, with or without a figure attached. ASSIGNMENT.md shows
how to put an image in a message.
"""

import logging
import os
import tomllib
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

#: Stable alias exposed by the canonical student/staff validator deployment.
BACKEND = os.getenv("VALIDATOR_BACKEND", "openrouter")
MODEL = os.getenv("VALIDATOR_MODEL", "validator")
BACKENDS = Path(__file__).resolve().parents[1] / "infrastructure" / "backends.toml"

#: --max-model-len of the Modal deployment (infrastructure/validator_model.py).
#: OpenRouter's context is far larger, so going over this is only logged.
MODAL_MAX_MODEL_LEN = 32768

logger = logging.getLogger(__name__)


def _modal_client() -> OpenAI:
    base_url = os.getenv("VALIDATOR_BASE_URL")
    api_key = os.getenv("VALIDATOR_API_KEY")
    missing = [
        name
        for name, value in {
            "VALIDATOR_BASE_URL": base_url,
            "VALIDATOR_API_KEY": api_key,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(
            f"missing {', '.join(missing)}; copy .env.example to .env and fill in your deployment"
        )
    return OpenAI(base_url=base_url, api_key=api_key, timeout=300, max_retries=3)


def _modal_complete(messages: list[dict[str, Any]], **generation: Any) -> dict[str, Any]:
    options = {"temperature": 0, **generation}
    response = _modal_client().chat.completions.create(
        model=MODEL,
        messages=messages,  # type: ignore[arg-type] - kept flexible for image parts
        **options,
    )
    return response.model_dump()


def _openrouter_client() -> OpenAI:
    base_url = os.getenv("OPENROUTER_BASE_URL")
    api_key = os.getenv("OPENROUTER_API_KEY")
    missing = [
        name
        for name, value in {
            "OPENROUTER_BASE_URL": base_url,
            "OPENROUTER_API_KEY": api_key,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(
            f"missing {', '.join(missing)}; copy .env.example to .env and fill in your OpenRouter key"
        )
    return OpenAI(base_url=base_url, api_key=api_key, timeout=300, max_retries=3)


def _openrouter_backend() -> dict[str, Any]:
    # backends.toml maps the Modal alias (MODEL) to its OpenRouter model and provider pin.
    backend = tomllib.loads(BACKENDS.read_text())["validator"]
    if backend["modal_key"] != MODEL:
        raise RuntimeError(
            f"VALIDATOR_MODEL={MODEL!r} does not match modal_key {backend['modal_key']!r} in {BACKENDS}"
        )
    return backend


def _openrouter_complete(messages: list[dict[str, Any]], **generation: Any) -> dict[str, Any]:
    backend = _openrouter_backend()
    options = {"temperature": 0, **generation}
    response = _openrouter_client().chat.completions.create(
        model=backend["openrouter_model"],
        messages=messages,  # type: ignore[arg-type] - kept flexible for image parts
        extra_body={"provider": backend["openrouter_provider"]},
        **options,
    )

    # vLLM rejects a request when prompt plus max_tokens exceeds max_model_len.
    prompt_tokens = response.usage.prompt_tokens if response.usage else 0
    max_tokens = options.get("max_tokens", 0)
    requested = prompt_tokens + max_tokens
    if requested > MODAL_MAX_MODEL_LEN:
        logger.warning(
            f"prompt {prompt_tokens} + max_tokens {max_tokens} = {requested} tokens "
            f"exceeds the Modal validator's {MODAL_MAX_MODEL_LEN}-token "
            "context; on Modal this input would have been compacted, so this "
            "judgment may differ"
        )
    return response.model_dump()


def complete(messages: list[dict[str, Any]], **generation: Any) -> dict[str, Any]:
    if BACKEND == "modal":
        return _modal_complete(messages, **generation)
    elif BACKEND == "openrouter":
        return _openrouter_complete(messages, **generation)
    else:
        raise ValueError(f"Invalid backend: {BACKEND}")


def _modal_tokenizer_request(endpoint: str, **body: Any) -> dict[str, Any]:
    # vLLM's tokenizer routes live beside /v1, not inside it. Keep any proxy
    # path prefix and use the same authenticated client as chat completions.
    client = _modal_client()
    root = str(client.base_url).rstrip("/").removesuffix("/v1")
    return client.post(
        f"{root}/{endpoint}", cast_to=dict[str, Any], body={"model": MODEL, **body}
    )


def _openrouter_tokenizer_request(endpoint: str, **body: Any) -> dict[str, Any]:
    # Not expected to be required for OpenRouter, but should fail if it happens.
    raise NotImplementedError(
        "Tokenization is not expected to be required for OpenRouter. OpenRouter has "
        "no tokenizer routes; tokenize/detokenize only work on the Modal backend."
    )


def _tokenizer_request(endpoint: str, **body: Any) -> dict[str, Any]:
    if BACKEND == "modal":
        return _modal_tokenizer_request(endpoint, **body)
    elif BACKEND == "openrouter":
        return _openrouter_tokenizer_request(endpoint, **body)
    else:
        raise ValueError(f"Invalid backend: {BACKEND}")


def tokenize(text: str) -> dict[str, Any]:
    """Return token IDs and max_model_len from the deployed vLLM server."""
    return _tokenizer_request("tokenize", prompt=text, add_special_tokens=False)


def detokenize(tokens: list[int]) -> str:
    return _tokenizer_request("detokenize", tokens=tokens)["prompt"]
