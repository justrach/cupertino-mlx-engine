import json
from typing import Generator, AsyncGenerator, Dict, Any

# Replace FastAPI imports with TurboAPI
from turboapi import TurboAPI
# Remove specific response types, TurboAPI handles them differently
# from fastapi import APIRouter
# from fastapi.responses import JSONResponse, StreamingResponse

from .mlx.models import load_model
from .schema import ChatCompletionRequest # Keep schema, assuming TurboAPI can use Pydantic or it will be adapted
from .text_models import BaseTextModel

# Instantiate TurboAPI instead of APIRouter
chat_app = TurboAPI(title="MLX Engine Chat - Chat Module") # Renamed from app


# Update decorators to use 'chat_app' and remove 'response_model'
@chat_app.post("/chat/completions")
@chat_app.post("/v1/chat/completions")
async def create_chat_completion(request): # Changed signature: receive raw request
    """Create a chat completion"""
    # Manually parse and validate the request body
    data = await request.json()
    try:
        # Assuming ChatCompletionRequest is Pydantic, validate manually
        chat_request = ChatCompletionRequest(**data)
    except Exception as e: # Replace with specific validation error if possible
        return {"error": f"Invalid request format: {e}"}, 400 # TurboAPI often returns tuple for status code

    text_model = _create_text_model(
        chat_request.model, chat_request.get_extra_params().get("adapter_path")
    )

    if not chat_request.stream:
        completion = text_model.generate(chat_request)
        # Return dict directly, TurboAPI handles JSON conversion
        return completion.model_dump(exclude_none=True)

    # Modify generator for TurboAPI streaming (assuming it handles SSE from async generator yielding dicts)
    async def event_generator() -> AsyncGenerator[Dict[str, Any], None]:
        async for chunk in text_model.stream_generate(chat_request):
            yield chunk.model_dump(exclude_none=True) # Yield dicts directly

        # TurboAPI might handle the stream end signal, or require a specific sentinel
        # For now, we just end the generator. The [DONE] message might need explicit handling
        # if the client relies on it and TurboAPI doesn't add it automatically.

    # Return the generator directly; TurboAPI should detect it and stream SSE
    # We might need to explicitly set headers if TurboAPI doesn't default to event-stream
    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }
    return event_generator(), 200, headers # Returning tuple with status and headers


_last_model_id = None
_last_text_model = None


def _create_text_model(model_id: str, adapter_path: str = None) -> BaseTextModel:
    global _last_model_id, _last_text_model
    if model_id == _last_model_id:
        return _last_text_model

    model = load_model(model_id, adapter_path)
    _last_text_model = model
    _last_model_id = model_id
    return model
