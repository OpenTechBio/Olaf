import logging
import sys
import os
import json
import asyncio
import base64
import tempfile
from contextlib import asynccontextmanager
from queue import Empty
import time

# --- FastAPI Imports ---
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field # Updated import for Field

# --- Jupyter Client Imports ---
# Using AsyncClient for better compatibility with FastAPI
from jupyter_client.manager import AsyncKernelManager
# Corrected import path for AsyncKernelClient
from jupyter_client.asynchronous.client import AsyncKernelClient

# --- Logging Setup ---
# Configure logging to see messages from this API and jupyter_client
logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format='%(asctime)s - %(name)s - %(levelname)s - [FastAPI_Kernel] %(message)s',
    force=True
)
log = logging.getLogger(__name__)

# --- Global Variables ---
# Global kernel manager and client to potentially reuse connection (or manage lifecycle)
# Using lifespan events is generally preferred for managing resources like this.
kernel_manager: AsyncKernelManager | None = None
kernel_client: AsyncKernelClient | None = None
KERNEL_CONNECTION_FILE = "/home/sandboxuser/kernel-connection.json" # Path inside container

# --- Pydantic Models ---
class CodeExecutionRequest(BaseModel):
    """Request model for code execution."""
    code: str = Field(..., description="The Python code string to execute.")
    timeout: int = Field(60, description="Execution timeout in seconds.")

class StreamOutput(BaseModel):
    """Model for stdout/stderr stream messages."""
    type: str = "stream"
    name: str # 'stdout' or 'stderr'
    text: str

class ErrorOutput(BaseModel):
    """Model for execution errors."""
    type: str = "error"
    ename: str # Error name (e.g., 'ValueError')
    evalue: str # Error value (message)
    traceback: list[str] # List of traceback lines

class DisplayDataOutput(BaseModel):
    """Model for display_data messages (like images)."""
    type: str = "display_data"
    data: dict[str, str] # Mime-type -> Base64 encoded data (e.g., 'image/png': 'base64...')
    metadata: dict = Field(default_factory=dict)

class ExecuteResultOutput(BaseModel):
    """Model for the result of the last expression."""
    type: str = "execute_result"
    data: dict[str, str] # Mime-type -> String data (e.g., 'text/plain': 'result')
    metadata: dict = Field(default_factory=dict)

class ExecutionStatus(BaseModel):
    """Model for kernel status updates."""
    type: str = "status"
    execution_state: str # e.g., 'busy', 'idle'

# Union type for different output possibilities
OutputType = StreamOutput | ErrorOutput | DisplayDataOutput | ExecuteResultOutput | ExecutionStatus

class CodeExecutionResponse(BaseModel):
    """Response model containing execution results."""
    outputs: list[OutputType] = Field(..., description="List of outputs from the kernel.")
    final_status: str = Field("unknown", description="Final status from execute_reply ('ok', 'error', 'aborted').")

# --- Helper Functions ---
async def get_kernel_client() -> AsyncKernelClient:
    """
    Connects to the kernel using the connection file.
    Raises FileNotFoundError or TimeoutError if connection fails.
    """
    log.info(f"Attempting to connect to kernel using {KERNEL_CONNECTION_FILE}")
    if not os.path.exists(KERNEL_CONNECTION_FILE):
        log.error(f"Kernel connection file not found at {KERNEL_CONNECTION_FILE}")
        raise FileNotFoundError("Kernel connection file not found.")

    # Create a client connected to the existing kernel
    kc = AsyncKernelClient(connection_file=KERNEL_CONNECTION_FILE)
    kc.load_connection_file()

    # Start channels - crucial for communication
    # This method is synchronous in the async client, it starts background tasks/threads.
    try:
        log.debug("Starting kernel client channels (synchronous call)...")
        kc.start_channels() # <-- REMOVED await asyncio.wait_for()
        log.info("Kernel client channels started.")
    except Exception as e:
        log.error(f"Error starting kernel client channels: {e}", exc_info=True)
        raise e # Re-raise other exceptions

    # Check if kernel is alive with timeout
    try:
        log.info("Waiting for kernel to be ready...")
        # wait_for_ready IS awaitable
        await asyncio.wait_for(kc.wait_for_ready(timeout=15.0), timeout=20.0)
        log.info("Kernel is ready.")
        return kc
    except asyncio.TimeoutError:
        log.error("Timeout waiting for kernel to become ready.")
        # Attempt to stop channels if started
        try:
            # stop_channels might also be sync, handle potential errors
            if kc.channels_running:
                 kc.stop_channels()
        except Exception:
            pass # Ignore errors during cleanup
        raise TimeoutError("Timeout waiting for kernel readiness.")
    except RuntimeError as e:
        log.error(f"Kernel readiness check failed: {e}")
        try:
            if kc.channels_running:
                 kc.stop_channels()
        except Exception:
            pass
        raise RuntimeError(f"Kernel readiness check failed: {e}")
    except Exception as e:
        log.error(f"Unexpected error during kernel readiness check: {e}", exc_info=True)
        try:
            if kc.channels_running:
                 kc.stop_channels()
        except Exception:
            pass
        raise e


async def execute_code_on_kernel(kc: AsyncKernelClient, code: str, timeout: int) -> CodeExecutionResponse:
    """
    Executes code using the provided async kernel client and gathers results.
    """
    log.info(f"Executing code (timeout={timeout}s):\n---\n{code}\n---")
    outputs = []
    final_status = "unknown"

    # Send execute request
    msg_id = kc.execute(code=code, store_history=True)
    log.debug(f"Execute request sent, msg_id: {msg_id}")

    # Process messages until idle or error
    start_time = time.time()
    execution_done = False
    while time.time() - start_time < timeout:
        try:
            # Get message from IOPub channel with a short timeout
            msg = await asyncio.wait_for(kc.get_iopub_msg(timeout=1.0), timeout=1.5)
            msg_type = msg['header']['msg_type']
            content = msg['content']
            log.debug(f"Received IOPub message type: {msg_type}")

            if msg_type == 'status':
                outputs.append(ExecutionStatus(execution_state=content['execution_state']))
                if content['execution_state'] == 'idle':
                    log.debug("Kernel reported idle status.")
                    execution_done = True
                    # Don't break immediately, wait for shell reply below
            elif msg_type == 'stream':
                outputs.append(StreamOutput(name=content['name'], text=content['text']))
            elif msg_type == 'display_data':
                # Base64 encode binary data for JSON transfer
                encoded_data = {}
                for mime, data in content.get('data', {}).items():
                    if isinstance(data, bytes):
                        encoded_data[mime] = base64.b64encode(data).decode('utf-8')
                    elif isinstance(data, str): # Assume text is already appropriate string
                         if mime.startswith('image/') or mime == 'text/html':
                              encoded_data[mime] = data # Keep as string
                         else:
                              pass
                    else:
                         log.warning(f"Unsupported data type '{type(data)}' in display_data for mime '{mime}'")

                if encoded_data:
                    outputs.append(DisplayDataOutput(data=encoded_data, metadata=content.get('metadata', {})))

            elif msg_type == 'execute_result':
                 outputs.append(ExecuteResultOutput(data=content.get('data', {}), metadata=content.get('metadata', {})))
            elif msg_type == 'error':
                outputs.append(ErrorOutput(
                    ename=content.get('ename', 'UnknownError'),
                    evalue=content.get('evalue', ''),
                    traceback=content.get('traceback', [])
                ))
                log.error(f"Kernel execution error: {content.get('ename')}")
                execution_done = True # Error means execution finished
                # Don't break immediately, wait for shell reply

        except asyncio.TimeoutError:
            if execution_done:
                 log.debug("IOPub processing finished after kernel idle/error.")
                 break
            else:
                 pass
        except Empty:
             log.debug("IOPub queue empty, continuing wait...")
             if execution_done: break
        except Exception as e:
            log.error(f"Error processing IOPub message: {e}", exc_info=True)
            outputs.append(ErrorOutput(ename="ClientError", evalue=f"Error reading IOPub: {e}", traceback=[]))
            execution_done = True
            break

    # After loop (timeout or kernel idle/error), get the shell reply
    try:
        shell_reply = await asyncio.wait_for(kc.get_shell_msg(timeout=5.0), timeout=6.0)
        if shell_reply['parent_header'].get('msg_id') == msg_id:
            final_status = shell_reply['content']['status']
            log.info(f"Execution final status from shell reply: {final_status}")
            if final_status == 'error' and not any(o.type == 'error' for o in outputs):
                 outputs.append(ErrorOutput(
                     ename=shell_reply['content'].get('ename', 'ShellError'),
                     evalue=shell_reply['content'].get('evalue', 'Error reported by shell'),
                     traceback=shell_reply['content'].get('traceback', [])
                 ))
        else:
            log.warning(f"Received shell message {shell_reply.get('msg_id')} doesn't match request {msg_id}")
            final_status = "mismatched_reply"
    except asyncio.TimeoutError:
        log.warning("Timeout waiting for shell reply.")
        final_status = "timeout_shell_reply"
    except Empty:
         log.warning("Shell reply queue empty.")
         final_status = "empty_shell_reply"
    except Exception as e:
        log.error(f"Error getting shell reply: {e}", exc_info=True)
        final_status = "error_shell_reply"

    if not execution_done and time.time() - start_time >= timeout:
         log.error("Execution timed out.")
         if not any(o.type == 'error' and 'Timeout' in o.evalue for o in outputs):
              outputs.append(ErrorOutput(ename="TimeoutError", evalue=f"Execution timed out after {timeout} seconds", traceback=[]))
         final_status = "timeout"


    return CodeExecutionResponse(outputs=outputs, final_status=final_status)


# --- FastAPI App ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("FastAPI application startup...")
    yield
    log.info("FastAPI application shutdown...")

app = FastAPI(lifespan=lifespan, title="Jupyter Kernel Execution API")

@app.get("/status", summary="Check API status")
async def get_status():
    """Returns the status of the API."""
    log.info("Status endpoint called.")
    kernel_file_exists = os.path.exists(KERNEL_CONNECTION_FILE)
    return JSONResponse(content={
        "status": "ok",
        "kernel_connection_file_found": kernel_file_exists
    })

@app.post("/execute",
          response_model=CodeExecutionResponse,
          summary="Execute Python code in the kernel")
async def execute_code_endpoint(payload: CodeExecutionRequest):
    """
    Receives Python code, executes it using the Jupyter kernel,
    and returns captured outputs (stdout, stderr, errors, display data).
    """
    log.info(f"Received code execution request (timeout={payload.timeout}s).")
    kc = None
    try:
        kc = await get_kernel_client()
        response = await execute_code_on_kernel(kc, payload.code, payload.timeout)
        log.info(f"Execution finished with final status: {response.final_status}")
        return response

    except FileNotFoundError:
        log.error("Kernel connection file missing.")
        raise HTTPException(status_code=503, detail="Kernel connection file not found. Is the kernel running?")
    except TimeoutError as e:
        log.error(f"Timeout during kernel connection or execution: {e}")
        raise HTTPException(status_code=504, detail=f"Timeout: {e}")
    except RuntimeError as e:
         log.error(f"Runtime error during kernel connection: {e}")
         raise HTTPException(status_code=503, detail=f"Kernel connection runtime error: {e}")
    except Exception as e:
        log.error(f"Unexpected error during code execution: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
    finally:
        # Ensure kernel client channels are stopped after each request
        if kc and kc.channels_running:
            log.debug("Stopping kernel client channels for this request.")
            try:
                # stop_channels is likely synchronous, call directly
                kc.stop_channels()
            except Exception as e:
                log.warning(f"Error stopping kernel client channels for request: {e}")

# --- Uvicorn Entry Point (for direct execution if needed) ---
if __name__ == "__main__":
    import uvicorn
    log.info("Starting Uvicorn server directly for debugging...")
    uvicorn.run("kernel_api:app", host="0.0.0.0", port=8000, log_level="debug", reload=True) # Added reload for dev
