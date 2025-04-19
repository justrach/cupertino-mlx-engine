import json
import re  # Import re for regex
import time # Import time for timestamps
from typing import Generator, List, Dict, Any, Union # Added Dict, Any, Union
import collections.abc # To check for Mapping/Sequence types

from starlette.responses import StreamingResponse
from starlette.requests import Request
from turboapi import APIRouter, JSONResponse

from .mlx.models import load_model
# Import the base Model class from satya to check instance types
from satya import Model
# Import necessary schema components
from .schema import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Role, # Assuming Role is needed by ChatMessage or used elsewhere
    ChatCompletionChunk,         # Needed for constructing chunks
    ChatCompletionChunkChoice,   # Needed for constructing chunks
    # Import the chunk types if needed for type hints, though Model check is key
    # ChatCompletionChunk,
    # ChatCompletionChunkChoice
)
from .text_models import BaseTextModel

router = APIRouter(tags=["chat—completions"])


# --- Helper function for recursive serialization ---
def recursive_to_dict(item: Any) -> Any:
    """Recursively converts satya.Model instances to dictionaries."""
    if isinstance(item, Model):
        # Call .dict() on the model instance
        try:
            d = item.dict() # Removed exclude_unset=True
        except AttributeError:
             # Fallback if .dict() doesn't exist - adapt as needed for satya
             # This example assumes fields are attributes or stored in __fields__
             try:
                 d = {f: getattr(item, f) for f in item.__fields__ if hasattr(item, f)} # Check hasattr
             except AttributeError:
                 # Last resort: return as is, hoping it's serializable or error later
                 return item # Or raise an error?

        # Recursively process the dictionary values
        return recursive_to_dict(d)
    elif isinstance(item, collections.abc.Mapping):
        # If it's a dictionary-like object, process its values
        return {k: recursive_to_dict(v) for k, v in item.items()}
    elif isinstance(item, collections.abc.Sequence) and not isinstance(item, (str, bytes)):
        # If it's a list/tuple-like object (but not string/bytes), process its elements
        return [recursive_to_dict(elem) for elem in item]
    else:
        # Assume it's a primitive type (int, str, float, bool, None)
        return item
# --- End Helper function ---


@router.post("/chat/completions", response_model=ChatCompletionResponse)
@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: Request):
    """Create a chat completion"""
    request_id = f"chatcmpl-{int(time.time() * 1000)}" # Generate a request ID
    created_time = int(time.time())

    try:
        body = await request.json()
        chat_request = ChatCompletionRequest(**body)

        # --- Explicit Deserialization for Nested Models ---
        if chat_request.messages:
            try:
                 chat_request_data = chat_request.dict()
            except AttributeError:
                 chat_request_data = {f: getattr(chat_request, f) for f in chat_request.__fields__} # Example fallback

            raw_messages = chat_request_data.get('messages', [])
            typed_messages = []
            if raw_messages:
                for msg in raw_messages:
                    if isinstance(msg, dict):
                        typed_messages.append(ChatMessage(**msg))
                    elif isinstance(msg, ChatMessage):
                        typed_messages.append(msg)
                    # else: handle unexpected types if needed

            chat_request_data['messages'] = typed_messages
            chat_request = ChatCompletionRequest(**chat_request_data)
        # --- End Explicit Deserialization ---

        text_model = _create_text_model(
            chat_request.model, chat_request.get_extra_params().get("adapter_path")
        )

        if not chat_request.stream:
            completion = text_model.generate(chat_request)
            # Ensure the completion response has the correct ID, created time, etc.
            # If text_model.generate returns a complete ChatCompletionResponse object:
            if isinstance(completion, ChatCompletionResponse):
                completion.id = request_id
                completion.created = created_time
                completion.model = chat_request.model # Ensure model name is set
            # If it returns something else, adapt accordingly.
            response_content = recursive_to_dict(completion)
            return JSONResponse(content=response_content)

        # Handling streaming response
        async def event_generator() -> Generator[str, None, None]:
            stream_buffer = "" # Buffer for handling tags spanning multiple chunks
            last_chunk_was_thought = False # Track if the last yield was a thought
            chunk_index = 0 # For logging

            if chat_request.thinkingModel:
                print("\n--- [Server] Event Generator (thinkingModel=True) Starting ---") # Server Log
                # Logic for thinking models: parse tags
                for chunk in text_model.stream_generate(chat_request): # Reverted back to standard for loop
                    chunk_index += 1
                    print(f"--- [Server] Raw Chunk {chunk_index} Received: {chunk!r} ---") # Server Log raw chunk

                    # Assuming chunk.delta.content contains the raw text stream
                    raw_delta_content = None
                    finish_reason = None # Track finish_reason from the chunk
                    if chunk.choices:
                        delta = getattr(chunk.choices[0], 'delta', None)
                        if delta:
                             raw_delta_content = getattr(delta, 'content', None)
                        # Check finish_reason on the choice, not delta
                        finish_reason = getattr(chunk.choices[0], 'finish_reason', None)

                    print(f"--- [Server] Chunk {chunk_index}: Content='{raw_delta_content}', FinishReason='{finish_reason}' ---") # Server Log parsed parts

                    if raw_delta_content:
                        stream_buffer += raw_delta_content

                    # Process buffer for thoughts and content - Yield COMPLETE parts found
                    parts = re.split(r"(<think>.*?</think>)", stream_buffer, flags=re.DOTALL)
                    processed_buffer_len = 0 # Track how much of the buffer we processed and yielded

                    for i, part in enumerate(parts):
                        if not part: continue # Skip empty parts

                        is_thought = part.startswith("<think>") and part.endswith("</think>")
                        is_last_part = (i == len(parts) - 1)

                        # Yield complete thoughts or content that is NOT the potentially incomplete last part
                        should_yield = False
                        if is_thought:
                            should_yield = True
                        elif not is_last_part and part: # Content between thoughts or before first thought
                            should_yield = True
                        # else: last part is potentially incomplete, keep in buffer until next chunk or end

                        if should_yield:
                            if is_thought:
                                thought_content = part[len("<think>"):-len("</think>")]
                                print(f"--- [Server] Yielding Thought Chunk: '{thought_content[:50]}...' ---") # Server Log
                                delta_msg = ChatMessage(role=Role.ASSISTANT, thought=thought_content)
                                last_chunk_was_thought = True
                            else: # It's content
                                print(f"--- [Server] Yielding Content Chunk: '{part[:50]}...' ---") # Server Log
                                delta_msg = ChatMessage(role=Role.ASSISTANT, content=part)
                                last_chunk_was_thought = False

                            chunk_to_send = ChatCompletionChunk(
                                id=request_id,
                                object="chat.completion.chunk",
                                created=created_time,
                                model=chat_request.model,
                                choices=[ChatCompletionChunkChoice(index=0, delta=delta_msg, finish_reason=None)]
                            )
                            serializable_chunk_dict = recursive_to_dict(chunk_to_send)
                            yield f"data: {json.dumps(serializable_chunk_dict)}\n\n"
                            processed_buffer_len += len(part) # Add length of yielded part
                        # No else needed, unyielded part stays implicitly

                    # Update buffer: keep only the part that wasn't processed/yielded (the last part if incomplete)
                    stream_buffer = stream_buffer[processed_buffer_len:]

                    # If finish_reason was received in *this* chunk, break the loop
                    # We will handle the remaining buffer *after* the loop.
                    if finish_reason:
                        print(f"--- [Server] Finish reason '{finish_reason}' received in chunk {chunk_index}, breaking loop. ---")
                        break

                # --- AFTER THE LOOP --- (Handles natural end or break due to finish_reason)
                print(f"--- [Server] Event Generator Loop Finished. Final Buffer: '{stream_buffer[:100]}...' ---") # Server Log

                # Yield any remaining content in the buffer
                if stream_buffer:
                    # Final check: is the remaining buffer a complete thought?
                    is_final_thought = stream_buffer.startswith("<think>") and stream_buffer.endswith("</think>")
                    if is_final_thought:
                        thought_content = stream_buffer[len("<think>"):-len("</think>")]
                        print(f"--- [Server] Yielding Final Buffered Thought: '{thought_content[:50]}...' ---") # Server Log
                        delta_msg = ChatMessage(role=Role.ASSISTANT, thought=thought_content)
                    else:
                        # Treat remaining as content
                        print(f"--- [Server] Yielding Final Buffered Content: '{stream_buffer[:50]}...' ---") # Server Log
                        delta_msg = ChatMessage(role=Role.ASSISTANT, content=stream_buffer)

                    chunk_to_send = ChatCompletionChunk(
                        id=request_id,
                        object="chat.completion.chunk",
                        created=created_time,
                        model=chat_request.model,
                        choices=[ChatCompletionChunkChoice(index=0, delta=delta_msg, finish_reason=None)] # Finish reason is sent in [DONE] or separate final chunk if needed
                    )
                    serializable_chunk_dict = recursive_to_dict(chunk_to_send)
                    yield f"data: {json.dumps(serializable_chunk_dict)}\n\n"
                    stream_buffer = "" # Clear buffer

                print(f"--- [Server] Event Generator Post-Loop Processing Done ---") # Server Log

            else:
                # Original logic for non-thinking models
                print("\n--- [Server] Event Generator (thinkingModel=False) Starting ---") # Server Log
                chunk_index_else = 0
                for chunk in text_model.stream_generate(chat_request): # Reverted back to standard for loop
                    chunk_index_else += 1
                    print(f"--- [Server] Raw Chunk {chunk_index_else} Received (non-thinking): {chunk!r} ---") # Server Log
                    # Ensure ID, created, model are set correctly if not done by stream_generate
                    if isinstance(chunk, ChatCompletionChunk):
                        chunk.id = request_id
                        chunk.created = created_time
                        chunk.model = chat_request.model
                    serializable_chunk_dict = recursive_to_dict(chunk)
                    yield f"data: {json.dumps(serializable_chunk_dict)}\n\n"
                print(f"--- [Server] Event Generator Loop Finished (Processed {chunk_index_else} chunks) ---") # Server Log


            # Send DONE signal
            print("--- [Server] Yielding [DONE] ---") # Server Log
            yield "data: [DONE]\n\n"
            print("--- [Server] Event Generator Fully Complete ---") # Server Log

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    except Exception as e:
        import traceback # Import traceback for detailed logging
        print(f"Error during chat completion: {e}")
        traceback.print_exc() # Print the full traceback for debugging
        # Return error with the generated request_id if possible
        error_content = {"error": str(e), "request_id": request_id}
        return JSONResponse(status_code=500, content=error_content)


# --- Model Caching Logic ---
_last_model_id = None
_last_text_model = None

def _create_text_model(model_id: str, adapter_path: str = None) -> BaseTextModel:
    """Loads or retrieves a cached text model."""
    global _last_model_id, _last_text_model
    cache_key = f"{model_id}_{adapter_path}" if adapter_path else model_id
    if cache_key == _last_model_id:
        return _last_text_model

    print(f"Loading model: {model_id}" + (f" with adapter: {adapter_path}" if adapter_path else ""))
    model = load_model(model_id, adapter_path)
    _last_text_model = model
    _last_model_id = cache_key
    print(f"Model {cache_key} loaded and cached.")
    return model