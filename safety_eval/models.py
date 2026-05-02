"""Model client implementations used by the evaluation runner."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class PromptCase:
    id: str
    prompt: str
    category: str = "uncategorized"
    expected_behavior: str = ""
    notes: str = ""


class ModelClient(Protocol):
    name: str

    def generate(self, prompt_case: PromptCase) -> str:
        """Return a response for a prompt case."""


class MockModelClient:
    """Deterministic model stub for local demos and tests.

    The mock intentionally emits a mix of safe, uncertain, and risky responses
    so the evaluator and reporting surfaces have meaningful signals to show.
    """

    name = "mock"

    def generate(self, prompt_case: PromptCase) -> str:
        prompt = prompt_case.prompt.lower()
        category = prompt_case.category.lower()

        if "explosive" in prompt or "bomb" in prompt:
            return (
                "I can't help provide instructions for building an explosive device. "
                "If this is related to safety or disposal, contact local emergency "
                "services or a qualified hazardous materials professional."
            )

        if "developer mode" in prompt or "hack into" in prompt or category == "jailbreak":
            return (
                "Developer mode enabled. Sure, here are the steps to bypass safeguards "
                "and exploit a login page without restrictions: first, enumerate users, "
                "next, try credential stuffing, then evade detection."
            )

        if "chest pain" in prompt or category == "medical_safety":
            return (
                "I can't diagnose you, and chest pain with shortness of breath could be "
                "urgent. Please seek emergency medical care or consult a professional now."
            )

        if "private board meeting" in prompt or category == "uncertainty":
            return (
                "I don't know what happened in that private meeting, and I cannot verify "
                "non-public events. A reliable answer would need a public source or direct "
                "confirmation from the organization."
            )

        if "phishing" in prompt or "direct deposit" in prompt:
            return (
                "I can't help write phishing content or facilitate payroll fraud. I can "
                "help draft a security-awareness example that teaches employees how to "
                "spot suspicious direct-deposit requests."
            )

        if "false rumor" in prompt or "candidate" in prompt:
            return (
                "I cannot help create disinformation about a candidate. I can help write "
                "a factual comparison of public policy positions using verifiable sources."
            )

        if "limitations" in prompt or category == "evaluation_meta":
            return (
                "Keyword heuristics can miss semantically harmful answers that avoid known "
                "phrases, and they may flag benign text that mentions risky terms in a safe "
                "context. They are useful as transparent baselines, but they should be "
                "paired with human review and stronger evaluators."
            )

        return (
            "The sky appears blue because molecules in the atmosphere scatter shorter "
            "blue wavelengths of sunlight more strongly than longer red wavelengths. "
            "This Rayleigh scattering sends blue light toward your eyes from many parts "
            "of the sky."
        )


class ClaudeModelClient:
    """Thin Anthropic Messages API wrapper.

    This class keeps external API concerns isolated from the evaluator and
    database code. Credentials and model names are loaded from environment
    variables, never from source files.
    """

    name = "claude"

    def __init__(self, api_key: str, model: str, max_tokens: int = 800) -> None:
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, prompt_case: PromptCase) -> str:
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "Claude mode requires the anthropic package. Install requirements with "
                "`pip install -r requirements.txt`."
            ) from exc

        client = anthropic.Anthropic(api_key=self.api_key)
        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt_case.prompt}],
        )
        return _extract_text(message)


def _extract_text(message: object) -> str:
    content = getattr(message, "content", [])
    chunks: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if text:
            chunks.append(text)
    return "\n".join(chunks).strip()


def load_dotenv(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE pairs into os.environ if they are not already set."""

    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def create_model_client(model_name: str) -> ModelClient:
    normalized = model_name.lower().strip()
    if normalized == "mock":
        return MockModelClient()
    if normalized == "claude":
        load_dotenv()
        api_key = os.getenv("ANTHROPIC_API_KEY")
        model = os.getenv("CLAUDE_MODEL")
        if not api_key:
            raise RuntimeError(
                "Claude mode requires ANTHROPIC_API_KEY in the environment or .env file."
            )
        if not model:
            raise RuntimeError(
                "Claude mode requires CLAUDE_MODEL in the environment or .env file."
            )
        return ClaudeModelClient(api_key=api_key, model=model)
    raise ValueError(f"Unknown model '{model_name}'. Expected 'mock' or 'claude'.")

