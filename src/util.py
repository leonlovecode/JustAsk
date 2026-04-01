#!/usr/bin/env python3
"""
Shared utilities for system prompt extraction.

Provides OpenRouter API client using OpenAI SDK.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

_client = None


def get_client() -> OpenAI:
    """Get or create OpenAI client (lazy initialization)."""
    global _client
    if _client is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    return _client


def call_model(
    model_id: str,
    user_message: str,
    system_prompt: str | None = None,
    max_tokens: int = 5000,
    temperature: float = 0,
    provider: str | None = None,
) -> dict:
    """
    Make API call to OpenRouter via OpenAI SDK.

    Args:
        provider: Pin to a specific OpenRouter provider (e.g., "DeepInfra", "Together").
                  Eliminates variance from provider routing.

    Returns:
        dict with keys: success, content, length (on success) or error (on failure)
    """
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        extra_body = {}
        if provider:
            extra_body["provider"] = {"order": [provider], "allow_fallbacks": False}

        def _do_call(disable_reasoning: bool):
            eb = dict(extra_body)
            if disable_reasoning:
                eb["reasoning"] = {"effort": "none"}
            kwargs = {"extra_body": eb} if eb else {}
            return get_client().chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )

        # Try with reasoning disabled first (saves cost); fallback if model requires it
        try:
            response = _do_call(disable_reasoning=True)
        except Exception as e:
            if "400" in str(e) and "eason" in str(e).lower():
                response = _do_call(disable_reasoning=False)
            else:
                raise

        content = response.choices[0].message.content
        if content is None:
            content = "[No content returned]"
        return {"success": True, "content": content, "length": len(content)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def call_model_multiturn(
    model_id: str,
    messages: list[dict],
    system_prompt: str | None = None,
    max_tokens: int = 5000,
    temperature: float = 0,
) -> dict:
    """
    Make multi-turn API call to OpenRouter.

    Args:
        model_id: The model to call
        messages: List of {"role": "user"|"assistant", "content": "..."}
        system_prompt: Optional system prompt
        max_tokens: Max tokens for response
        temperature: Sampling temperature

    Returns:
        dict with keys: success, content, length (on success) or error (on failure)
    """
    try:
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        response = get_client().chat.completions.create(
            model=model_id,
            messages=full_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body={"reasoning": {"effort": "none"}},
        )
        content = response.choices[0].message.content
        if content is None:
            content = "[No content returned]"
        return {"success": True, "content": content, "length": len(content)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_embedding(text: str, model: str = "openai/text-embedding-3-large") -> list[float] | None:
    """Get embedding vector via OpenRouter API."""
    try:
        response = get_client().embeddings.create(model=model, input=text)
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding API error: {e}")
        return None
