"""Store generation endpoint settings in the Modal gateway secret."""

from __future__ import annotations

import argparse
import json
import os
import tomllib

import modal
from dotenv import load_dotenv

from workflow.generate import AGENTS, BACKEND, OPENROUTER_CONFIG

SECRET_NAME = "hw2-generation-endpoints"


def endpoint_registry() -> dict[str, dict[str, str]]:
    api_key = os.getenv("GENERATION_API_KEY")
    missing = [] if api_key else ["GENERATION_API_KEY"]
    registry = {}
    for agent in AGENTS:
        base_url = os.getenv(agent.url_variable)
        if not base_url:
            missing.append(agent.url_variable)
        else:
            registry[agent.model] = {
                "api_base": base_url.rstrip("/"),
                "api_key": api_key or "",
            }
    if missing:
        raise ValueError(f"missing {', '.join(missing)} in .env")
    return registry


def openrouter_settings() -> dict[str, str]:
    names = ("OPENROUTER_BASE_URL", "OPENROUTER_API_KEY")
    missing = [name for name in names if not os.getenv(name)]
    if missing:
        raise ValueError(f"missing {', '.join(missing)} in .env")
    config = tomllib.loads(OPENROUTER_CONFIG.read_text())
    return {
        "OPENROUTER_BASE_URL": os.environ["OPENROUTER_BASE_URL"].rstrip("/"),
        "OPENROUTER_API_KEY": os.environ["OPENROUTER_API_KEY"],
        "OPENROUTER_CONFIG_JSON": json.dumps(config, sort_keys=True),
    }


def secret_values() -> dict[str, str]:
    if BACKEND == "modal":
        registry = endpoint_registry()
        values = {"GENERATION_ENDPOINTS_JSON": json.dumps(registry, sort_keys=True)}
    elif BACKEND == "openrouter":
        values = openrouter_settings()
    else:
        raise ValueError(f"Invalid backend: {BACKEND}")
    # Secret.update() merges keys, so always overwrite the backend selector.
    return {"GENERATION_BACKEND": BACKEND, **values}


def configure() -> None:
    values = secret_values()
    existing = {secret.name for secret in modal.Secret.objects.list()}
    if SECRET_NAME in existing:
        modal.Secret.from_name(SECRET_NAME).update(values)
    else:
        modal.Secret.objects.create(SECRET_NAME, values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    load_dotenv()
    try:
        configure()
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"configured Modal secret {SECRET_NAME!r}")


if __name__ == "__main__":
    main()
