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
import json

class OllamaClient:
    """
    Example:
        client = OllamaClient(host="http://localhost:11434", default_model="llama2")
        resp   = client.chat.completions.create(model="llama2", messages=[...])
        print(resp.choices[0].message.content)
    """

    def __init__(self, host: str = "http://localhost:11434", model: str = "deepseek-r1:70b"):
        if not host.startswith(("http://", "https://")):          # ← add
            host = "http://" + host         
        self._host = host.rstrip("/")
        self._default_model = model
        # expose nested namespaces so that usage mirrors openai.ChatCompletion
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat_create))

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #
    def _chat_create(
            self,
            *,
            messages: List[Dict[str, str]],
            temperature: float | None = None,
            **kwargs: Any,
        ):
            payload = {
                "model": self._default_model,
                "messages": messages,
                "stream": False,
            }
            if temperature is not None:
                payload["options"] = {"temperature": temperature}

            r = requests.post(f"{self._host}/api/chat", json=payload, timeout=300)
            r.raise_for_status()

            # ND-JSON → take the line that has the message
            for line in r.text.strip().splitlines():
                obj = json.loads(line)
                if "message" in obj:
                    content = obj["message"]["content"]
                    break
            else:
                raise ValueError("No message object found in Ollama response")

            message = SimpleNamespace(content=content, role="assistant")
            choice  = SimpleNamespace(message=message, index=0, finish_reason="stop")
            return SimpleNamespace(choices=[choice])