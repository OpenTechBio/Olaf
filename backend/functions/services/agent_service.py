import requests
import re
import json
from fpdf import FPDF
from openai import OpenAI
import anthropic
from os import environ

from datastructures.history import History
from services.ai_facade import ChatGPTFacade, ClaudeFacade, AIFacade


def chat_completion(history: History, tools=None, provider: str = 'chatgpt'):
    """
    Provides a chat completion using the specified AI provider.

    Args:
        history (History): The conversation history.
        tools (list, optional): A list of tools to provide to the model. Defaults to None.
        provider (str, optional): The AI provider to use ('chatgpt' or 'claude'). Defaults to 'chatgpt'.

    Yields:
        dict: Chunks of the AI model's response.
    """
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "gpt-4o",
        "messages": history.get_history(),
        "temperature": 0.1,
        "stream": True,
    }

    response = requests.post(openai_api_url, headers=headers, json=payload, stream=True)

    if response.status_code == 200:
        for chunk in response.iter_lines(decode_unicode=True):
            if chunk:
                try:
                    chunk = chunk.replace("data: ", "").strip()
                    json_chunk = json.loads(chunk)
                    yield json_chunk
                except:
                    continue

    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None


def chat_completion_function(history: History, tools, provider: str = 'chatgpt'):
    """
    Provides a chat completion that focuses on function calling.

    Args:
        history (History): The conversation history.
        tools (list): A list of tools to provide to the model.
        provider (str, optional): The AI provider to use ('chatgpt' or 'claude'). Defaults to 'chatgpt'.

    Returns:
        The response stream from the AI provider.
    """
    ai_facade = _get_ai_facade(provider)
    # The system prompt for function calling might need to be adjusted per provider
    # or handled within the facade's generate_response method if providers have different ways
    # of handling tool calls. For now, passing it as part of history.
    # history.log(role="system", content="Call the function best suited to find the response", type="text")
    stream = ai_facade.generate_response(history=history, tools=tools, system_prompt="Call the function best suited to find the response")
        temperature=0.1,
        tools=tools,
    )

    return stream


def _get_ai_facade(provider: str) -> AIFacade:
    """Factory function to get the appropriate AI facade based on provider."""
    if provider == 'chatgpt':
        openai_api_key = environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API key not found in environment variables.")
        return ChatGPTFacade(api_key=openai_api_key)
    elif provider == 'claude':
        claude_api_key = environ.get("CLAUDE_API_KEY")
        if not claude_api_key:
            raise ValueError("Claude API key not found in environment variables.")
        return ClaudeFacade(api_key=claude_api_key)
    else:
        raise ValueError(f"Unsupported AI provider: {provider}")

def chat_completion_api(history: History, system_prompt: str, tools=None):
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": system_prompt},
        ]
        + history.get_history(),
        "temperature": 0.1,
        "tools": tools,
        "stream": True,
    }

    response = requests.post(openai_api_url, headers=headers, json=payload, stream=True)

    if response.status_code == 200:
        for chunk in response.iter_lines(decode_unicode=True):
            if chunk:
                try:
                    chunk = chunk.replace("data: ", "").strip()
                    json_chunk = json.loads(chunk)
                    yield json_chunk
                except:
                    continue

    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None


def chat_completion_plan(history: History, system_prompt: str, tools=None):
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "gpt-4o-2024-08-06",
        "messages": [
            {"role": "system", "content": system_prompt},
        ]
        + history.get_history(),
        "temperature": 0.1,
        "tools": tools,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "plan_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "overview": {
                            "type": "string",
                            "description": "A brief overview of the entire plan.",
                        },
                        "steps": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "step_number": {"type": "integer"},
                                    "title": {
                                        "type": "string",
                                        "description": "A short title or name for the step.",
                                    },
                                    "description": {
                                        "type": "string",
                                        "description": "A detailed explanation of what this step involves.",
                                    },
                                    "dependencies": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "Any prerequisites or dependencies required for this step.",
                                    },
                                    "expected_output": {
                                        "type": "string",
                                        "description": "The expected result or output of this step.",
                                    },
                                },
                                "required": [
                                    "step_number",
                                    "title",
                                    "description",
                                    "dependencies",
                                    "expected_output",
                                ],
                                "additionalProperties": False,
                            },
                        },
                        "final_check": {
                            "type": "string",
                            "description": "A final review or checklist after all steps are completed.",
                        },
                    },
                    "required": ["overview", "steps", "final_check"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        },
        "stream": True,
    }

    response = requests.post(openai_api_url, headers=headers, json=payload, stream=True)

    if response.status_code == 200:
        for chunk in response.iter_lines(decode_unicode=True):
            if chunk:
                try:
                    chunk = chunk.replace("data: ", "").strip()
                    json_chunk = json.loads(chunk)
                    yield json_chunk
                except:
                    continue

    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None


def extract_code_and_text(content: str) -> tuple[str, str]:
    code_blocks = re.findall(r"```(.*?)```", content, re.DOTALL)
    text_parts = re.split(r"```.*?```", content, flags=re.DOTALL)
    text = " ".join([part.strip() for part in text_parts if part.strip()])
    code = "\n".join([block.strip() for block in code_blocks if block.strip()])
    return text, code


class AgentService:
    """
    Service class containing utility methods for agents.

    NOTE: These functions are are very messy and need refactoring. But since I
    haven't worked on this part, I'm leaving it to the relevant people.

    NOTE: The above methods haven't been moved into the AgentService class
    because they are used by the agents, which I will also leave to be worked on
    by the relevant people. Please move them into this class during refactoring.

    Some notes on what to fix:
    1. Many methods do the same thing or are unused. Please keep consistent.
    2. Please put type annotations on parameters and returns.
    3. Some generator functions mix yield and return statements. Please keep
       consistent, returns are unnecessary for generators.
    4. Our GPT calls seem to be a mix of HTTP requests and the OpenAI Python SDK.
       Please keep consistent. I'd recommend using the SDK for type annotating +
       convenience + no gigantic string payloads.
    5. Basic docstrings on functions would be nice. No need to be complicated,
       but a short description would be helpful. (ie. Previously a function was
       accepting @param: history as List[ChatMessage] instead of a History object.)
    """

    def __init__(self):
        # we should move this to a secret store and load from there eventually
        pass # API keys are now handled in _get_ai_facade
        """
        Returns a summary of a chat history as a string. Returns None if response
        fails.
        """
        headers = {
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json",
        }

        history.log(
            role="user",
            content="Please provide a clear and structured summary of the entire conversation in a user-assistant format, detailing exactly who asked what and who responded with what. Focus solely on the main points discussed without any introductory or concluding remarks. Ensure the summary is easily understandable to non-technical individuals.",
            type="text",
        )

        payload = {"model": "gpt-4o", "messages": history, "temperature": 0.1}

        response = requests.post(
            openai_api_url, headers=headers, json=payload, stream=True
        )
        response_data = response.json()

        if response.status_code == 200 and "choices" in response_data:
            return response_data["choices"][0]["message"]["content"]

        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None

    def stream(self, agent):
        for chunk in agent.generate():
            try:
                content = chunk["choices"][0]["delta"]["content"]
                yield content.encode("utf-8")
            except:
                if chunk.startswith("Response: "):
                    yield chunk.replace("Response: ", "").encode("utf-8")
                continue

    def extract_python_code(self, text: str) -> str:
        """
        Extract the Python code from the provided text string.

        Args:
        text (str): The input text containing the Python code

        Returns:
        str: The extracted Python code
        """
        code_pattern = re.compile(r"```python(.*?)```", re.DOTALL)
        match = code_pattern.search(text)

        if match:
            return match.group(1).strip()
        else:
            return "No Python code found in the input text."

    def create_pdf_from_summary(
        self, conversation_summary: str, session_id: str
    ) -> bytes:
        """
        Generates a structured PDF with a clear heading and formatted content.
        """
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # Set title font for the heading
        pdf.set_font("Helvetica", style="B", size=16)
        pdf.cell(0, 10, "Session Summary", ln=True, align="C")
        pdf.ln(10)

        # Add session ID as a subtitle
        pdf.set_font("Helvetica", style="B", size=12)
        pdf.cell(0, 10, f"Session ID: {session_id}", ln=True, align="L")
        pdf.ln(5)

        # Add conversation summary content
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(0, 8, conversation_summary)

        # Output PDF as a string (bytes)
        pdf_output: str = pdf.output(dest="S")  # type: ignore
        pdf_bytes = pdf_output.encode("latin1")
        return pdf_bytes
