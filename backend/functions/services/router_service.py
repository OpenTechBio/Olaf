from typing import List

from agents.codemaster_agent import CodeMasterAgent
from functions.models.chat_message import ChatMessage
from functions.routes.router import Router
from functions.models.history import History


def master_route_function(session_history: List[ChatMessage]):
    # session_history is a list of dict: [{"role": "...", "content": "...", "type": "..."}]
    history = History(session_history)
    master_agent = CodeMasterAgent("python", history)  # type: ignore
    return master_agent.generate_response()


def get_master_router():
    router = Router()
    router.add_route("master", master_route_function)
    return router
