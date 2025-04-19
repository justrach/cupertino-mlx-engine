import json
from typing import Generator

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from .mlx.models import load_model
from .schema import ChatCompletionRequest, ChatCompletionResponse
from .text_models import BaseTextModel

router = APIRouter(tags=["chat—completions"])


@router.post("/chat/completions", response_model=ChatCompletionResponse)
@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion"""

    text_model = _create_text_model(
        request.model, request.get_extra_params().get("adapter_path")
    )

    if not request.stream:
        completion = text_model.generate(request)
        return JSONResponse(content=completion.model_dump(exclude_none=True))

    async def event_generator() -> Generator[str, None, None]:
        for chunk in text_model.stream_generate(request):
            yield f"data: {json.dumps(chunk.model_dump(exclude_none=True))}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


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
