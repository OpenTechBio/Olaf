"""
A minimal wrapper that mimics the subset of the OpenAI Python
client used in Interactive Auto Agent System Tester.

Only implements:
    client.chat.completions.create(model=..., messages=[...], temperature=...)
and returns an object whose shape matches the OpenAI response
access pattern:  resp.choices[0].message.content
"""

from __future__ import annotations
import requests
from types import SimpleNamespace
from typing import List, Dict, Any


class OllamaClient:
    """
    Example:
        client = OllamaClient(host="http://localhost:11434", default_model="llama2")
        resp   = client.chat.completions.create(model="llama2", messages=[...])
        print(resp.choices[0].message.content)
    """

    def __init__(self, host: str = "http://localhost:11434", default_model: str = "llama2"):
        self._host = host.rstrip("/")
        self._default_model = default_model
        # expose nested namespaces so that usage mirrors openai.ChatCompletion
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat_create))

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #
    def _chat_create(
        self,
        *,
        model: str | None = None,
        messages: List[Dict[str, str]],
        temperature: float | None = None,
        **kwargs: Any,
    ):
        """POST /api/chat and wrap the response to look like OpenAI’s."""
        payload: dict[str, Any] = {
            "model": model or self._default_model,
            "messages": messages,
        }
        if temperature is not None:
            payload["options"] = {"temperature": temperature}

        r = requests.post(f"{self._host}/api/chat", json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()                       # -> {'message': {...}, 'done': true}

        content = data["message"]["content"]
        # fabricate an OpenAI-shaped object tree
        message = SimpleNamespace(content=content, role="assistant")
        choice  = SimpleNamespace(message=message, index=0, finish_reason="stop")
        response_obj = SimpleNamespace(choices=[choice])

        return response_obj