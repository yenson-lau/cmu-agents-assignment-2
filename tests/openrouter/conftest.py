"""Shared fixtures for the OpenRouter tests: openrouter.toml, the client and a test chart."""

from __future__ import annotations

import base64
import os
import tomllib
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import pytest
from dotenv import load_dotenv
from openai import OpenAI

RELEASE = Path(__file__).resolve().parents[2]
OPENROUTER_CONFIG = RELEASE / "infrastructure" / "openrouter.toml"

load_dotenv(RELEASE / ".env")

# matplotlib's "Bar color demo" for the version this repo pins. The tallest bar
# is blueberry (100); apple 40, cherry 30, orange 55.
IMAGE_URL = "https://matplotlib.org/3.10.5/_images/sphx_glr_bar_colors_001.png"
IMAGE_ANSWER = "blueberry"
IMAGE_QUESTION = "Which fruit has the tallest bar? Answer in one word."


def _load_openrouter_config() -> dict[str, Any]:
    return tomllib.loads(OPENROUTER_CONFIG.read_text())


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    # One test case per agent role in openrouter.toml; select with e.g. `-k medium`.
    if "agent" in metafunc.fixturenames:
        agents = _load_openrouter_config()["agents"]
        metafunc.parametrize("agent", list(agents.values()), ids=list(agents))


@pytest.fixture(scope="session")
def openrouter_config() -> dict[str, Any]:
    return _load_openrouter_config()


@pytest.fixture(scope="session")
def client() -> OpenAI:
    base_url = os.getenv("OPENROUTER_BASE_URL")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not base_url or not api_key:
        pytest.fail("missing OPENROUTER_BASE_URL or OPENROUTER_API_KEY in .env")
    return OpenAI(base_url=base_url, api_key=api_key, timeout=300)


@pytest.fixture(scope="session")
def chart_message() -> dict[str, Any]:
    """
    A user message asking about the bar chart, with the image inlined as base64
    like validator/baseline.py does. The expected answer is IMAGE_ANSWER.
    """

    request = Request(IMAGE_URL, headers={"User-Agent": "hw2-openrouter-check"})
    with urlopen(request, timeout=30) as response:
        encoded = base64.b64encode(response.read()).decode()
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": IMAGE_QUESTION},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
        ],
    }


@pytest.fixture(scope="session")
def image_answer() -> str:
    return IMAGE_ANSWER
