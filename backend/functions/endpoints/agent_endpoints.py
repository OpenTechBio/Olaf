from flask import stream_with_context
import json

from firebase_functions.https_fn import Request, Response, on_request
from firebase_functions.options import CorsOptions

from datastructures.history import History
from datastructures.router import Router
from routes.masteragent_route import masteragent_route
from routes.l3_masteragent_route import l3_masteragent_route
from routes.l3_coderagent_route import l3_coderagent_route
from services.agent_service import chat_completion
from services.session_service import SessionService
from utils.validation import (
    validate_master_agent_request,
    validate_name_maker_request,
)

from collections import deque

@on_request(cors=CorsOptions(cors_origins="*", cors_methods=["post"]))
def l3_master_agent_interaction(req: Request) -> Response:
    """
    Endpoint that can handle multiple back-and-forth calls 
    between the L3 Master Agent and other agents (e.g., CoderAgent).
    Uses a queue to orchestrate routing tasks in a single SSE stream.
    """
    try:
        # Validate request & manage session
        session_service = SessionService()
        session_id, user_id, project_id, message = validate_master_agent_request(req)
        if not session_id:
            session_id = None

        # Extract the optional 'provider' argument, defaulting to 'chatgpt'
        try:
            provider = req.get_json().get("provider", "chatgpt")
        except Exception:
            provider = "chatgpt"

        # Retrieve or create session
        if session_id is None:
            session_data = session_service.create_session(
                user_id, project_id, {"type": "text", "role": "user", "content": message}
            )
            session_id = session_data["id"]
        else:
            session_data = session_service.add_message_to_session(
                session_id, {"type": "text", "role": "user", "content": message}
            )

        if session_data is None:
            raise ValueError("Session not found")

        # Filter out unwanted message types
        session_data["history"] = [
            m
            for m in session_data["history"]
            if m.get("type") in ["text", "code", "error", "result", "executedCode", "plan"]
        ]

        # Initialize router & add routes for master and coder
        router = Router()
        router.add_route("l3_master_agent", l3_masteragent_route)
        router.add_route("coder_agent", l3_coderagent_route)
        def generate_stream():
            """
            This generator processes a queue of (route_key, data) tasks,
            streams each agent's output, and may enqueue new tasks if the agent
            yields function_router instructions.
            """
            # Create a queue of tasks. The first task is to call the L3 master agent.
            tasks = deque()
            # We pass the entire session history as the initial data
            tasks.append(("l3_master_agent", session_data["history"]))
            full_response_accumulator = ""

            # Process tasks until none remain
            while tasks:
                route_key, route_data = tasks.popleft() # Route_data here will be sessions data
                # Invoke the route → returns a generator of chunk dictionaries
                response_generator = router.route(route_key, route_data)

                partial_response = "" # This might not be needed with the new streaming method

                for chunk in response_generator:
                    try:
                        # If the chunk instructs us to route to another agent:
                        if chunk["type"] == "function_router":
                            # e.g. chunk["content"] = {"destination": "coder_agent", "plan": "..."}
                            content = chunk["content"]
                            destination = content.get("destination")

                            # if plan exists in content append it to the data
                            if "plan" in content:
                                plan_data = content["plan"]
                                route_data = session_service.add_message_to_session(
                                    session_id, {"type": "hidden", "role": "assistant", "content": plan_data} #THIS
                                )
                            if destination in router.routes:
                                print("route_data", route_data)
                                history = History(route_data["history"]) # type: ignore
                                tasks.append((destination, history))
                            else:
                                # Log and skip malformed chunk
                                print(f"Bad chunk: {chunk}")  
                                continue
                        else:
                            # Normal chunk from the agent
                            data = json.dumps({"type": chunk["type"], "content": chunk["content"]})
                            yield f"data: {data}\n\n".encode("utf-8")

                            partial_response += chunk["content"]
                    except Exception as e:
                        print(f"Error streaming chunk: {e}")
                        continue
                
                # we could do something with the partial response here, but for now we just accumulate it
                full_response_accumulator += partial_response

            # Store the assistant's response in firestore
            session_service.add_message_to_session(
                session_id,
                {"type": "text", "role": "assistant", "content": full_response_accumulator},
            )

        # Return SSE
        return Response(
            stream_with_context(generate_stream()),
            headers={"session_id": session_id},
            mimetype="text/event-stream",
        )

    except Exception as e:
        print(f"Error in l3_master_agent_interaction: {str(e)}")
        return Response(json.dumps({"error": str(e)}), status=500)

@on_request(cors=CorsOptions(cors_origins="*", cors_methods=["post"]))
def master_agent_interaction(req: Request) -> Response:
    try:
        session_service = SessionService()

        # Extract necessary parameters and validate them
        session_id, user_id, project_id, message = validate_master_agent_request(req)
        if not session_id:
            session_id = None

        # Extract the optional 'provider' argument, defaulting to 'chatgpt'
        try:
            provider = req.get_json().get("provider", "chatgpt")
        except Exception:
            provider = "chatgpt"

        # Retrieve or create session with the new user message
        if session_id is None:
            session_data = session_service.create_session(
                user_id,
                project_id,
                {"type": "text", "role": "user", "content": message},
            )
            session_id = session_data["id"]
        else:
            session_data = session_service.add_message_to_session(
                session_id, {"type": "text", "role": "user", "content": message}
            )

        if session_data is None:
            raise ValueError("Session not found")

        # Filter out non-text/code/error messages
        # NOTE this logic can be put in History class
        session_data["history"] = [
            m
            for m in session_data["history"]
            if m.get("type") in ["text", "code", "error", "result", "executedCode"]
        ]

        router = Router()
        router.add_route("master", masteragent_route)
        router.add_route("master", masteragent_route)
        response_generator = router.route("master", session_data["history"], provider)  # type: ignore

        # NOTE can probably be put into agent service.
        # the add_message_to_session() call can probably be done by putting a
        # callback parameter in AgentService.generate_stream(). this would
        # let you keep the SessionService.add_message_to_session() call here
        # without having to put a SessionService object to the agent_service.

        def generate_stream():
            full_response = ""
            for chunk in response_generator:
                try:
                    data = json.dumps(
                        {"type": chunk["type"], "content": chunk["content"]}
                    )
                    yield f"data: {data}\n\n".encode("utf-8")
                    full_response += chunk["content"]
                except Exception as e:
                    # Log and skip malformed chunk
                    print(f"Error streaming chunk: {e}")
                    continue
            # Store the assistant's response
            session_service.add_message_to_session(
                session_id,
                {"type": "text", "role": "assistant", "content": full_response},
            )

        return Response(
            stream_with_context(generate_stream()),
            headers={"session_id": session_id},
            mimetype="text/event-stream",
        )

    except Exception as e:
        print(f"Error in master_agent_interaction_handler: {str(e)}")
        return Response(json.dumps({"error": str(e)}), status=500)


@on_request(cors=CorsOptions(cors_origins="*", cors_methods=["post"]))
def name_maker(req: Request) -> Response:
    provider = req.get_json().get("provider", "chatgpt") # Add provider
    try:
        history = validate_name_maker_request(req)

        system = """
        You are given a conversation history.
        Please construct a super short title for the conversation.
        """
        hist = History(history)
        hist.log("system", system, "text")

        response = chat_completion(hist, provider=provider) # Pass provider
        response_str = ""
        for chunk in response:
            response_str += (
                chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
            )

        response_data = {"message": response_str}
        return Response(json.dumps(response_data), status=200)
    except Exception as e:
        return Response(json.dumps({"error": str(e)}), status=500)
