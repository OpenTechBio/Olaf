import abc
import os
import json
from openai import OpenAI
import anthropic

from datastructures.history import History
from models.chat_message import ChatMessageRole

class AIFacade(abc.ABC):
    @abc.abstractmethod
    def generate_response(self, prompt: str, history: History):
        """
        Abstract method to generate a response from an AI model.

        Args:
            prompt: The user's current prompt.
            history: A list representing the conversation history.

        Returns:
            A generator that yields chunks of the AI model's response.
        """
        pass

class ChatGPTFacade(AIFacade):
    def __init__(self):
        # Ensure the API key is set
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API key not found in environment variables.")
        self.client = OpenAI(api_key=openai_api_key)

    def generate_response(self, prompt: str, history: History):
        """
        Implements response generation using the ChatGPT API.
        """
        messages = history.get_history()
        # Add the current prompt as a user message
        messages.append({"role": ChatMessageRole.USER.value, "content": prompt})

        try:
            stream = self.client.chat.completions.create(
                model="gpt-4o", # Or another appropriate model
                messages=messages,
                temperature=0.1,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices:
                    content = chunk.choices[0].delta.content
                    if content:
                        # Yield chunks in a format similar to agent_service.py
                        yield {"type": "text", "content": content}

        except Exception as e:
            print(f"Error in ChatGPT API call: {e}")
            yield {"type": "error", "content": f"An error occurred with the ChatGPT API: {e}"}


class ClaudeFacade(AIFacade):
    def __init__(self):
        # Ensure the API key is set
        claude_api_key = os.environ.get("CLAUDE_API_KEY")
        if not claude_api_key:
            raise ValueError("Claude API key not found in environment variables.")
        self.client = anthropic.Anthropic(api_key=claude_api_key)

    def generate_response(self, prompt: str, history: History):
        """
        Implements response generation using the Claude API.
        """
        messages = history.get_history()
        # Add the current prompt as a user message
        messages.append({"role": ChatMessageRole.USER.value, "content": prompt})

        try:
            # Claude uses a different message structure and client method
            stream = self.client.messages.create(
                model="claude-3-opus-20240229", # Or another appropriate model
                max_tokens=4096,
                messages=messages,
                stream=True,
            )
            for chunk in stream:
                 if chunk.type == 'content_block_delta':
                    yield {"type": "text", "content": chunk.delta.text}

        except Exception as e:
            print(f"Error in Claude API call: {e}")
            yield {"type": "error", "content": f"An error occurred with the Claude API: {e}"}