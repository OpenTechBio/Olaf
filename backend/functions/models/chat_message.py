from typing import Literal, TypedDict


ChatMessageType = Literal[
    "text", "code", "executedCode", "plan", "error", "image", "result", "hidden", "terminal"
]
ChatMessageRole = Literal["assistant", "user", "system"]


class ChatMessage(TypedDict):
    """
    ChatMessages are JSON objects stored as dictionaries.
    """

    type: ChatMessageType
    role: ChatMessageRole
    content: str
