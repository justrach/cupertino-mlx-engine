# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

-   **Fixed serialization errors for `transformers` chat template.**
    -   Addressed `TypeError: Object of type Function is not JSON serializable` by ensuring `Tool` objects passed to `apply_chat_template` are fully serialized to dictionaries using `recursive_to_dict`.
    -   Resolved `UndefinedError: 'dict object' has no attribute 'content'` by ensuring the `content` key exists in message dictionaries passed to the template, even if `None`.
    -   Fixed `UndefinedError: 'None' has no attribute 'split'` by setting `content` to an empty string (`""`) instead of `None` for messages without content (like tool calls) to satisfy the template's expectation of a string.
-   **Resolved `AttributeError` for structured output (`json_schema`).**
    -   The initial error (`'dict' object has no attribute 'json_schema'`) occurred because the `response_format` field in the request was parsed as a dictionary, not a `ResponseFormat` model instance.
    -   A subsequent error (`'dict' object has no attribute 'schema_def'` or `'schema'`) occurred because the nested `json_schema` field within `response_format` was *also* being treated as a dictionary, and there was a field name mismatch (`schema_def` in the model vs. `schema` in the request/code).
    -   The fix involved:
        1.  Renaming the `schema_def` field in the `JsonSchemaFormat` model (in `src/mlxengine/chat/schema.py`) to `schema` to align with the expected input and usage.
        2.  Updating the code using this field (`src/mlxengine/chat/mlx/outlines_logits_processor.py`) to access `.schema`.
        3.  Implementing explicit, recursive deserialization in the chat router (`src/mlxengine/chat/router.py`) to ensure both `response_format` and its nested `json_schema` field are correctly converted from dictionaries to their respective `satya.Model` instances (`ResponseFormat` and `JsonSchemaFormat`) before being used.

## [0.0.2] - 2025-04-19

### Changed

-   **Refactored web framework from FastAPI to TurboAPI.**
    -   Replaced `FastAPI` instance with `TurboAPI`.
    -   Updated `APIRouter`, `JSONResponse` imports to use `turboapi`.
    -   Imported `StreamingResponse` directly from `starlette.responses`.
    -   Modified middleware registration to use the `middleware` parameter in `TurboAPI` constructor instead of `app.add_middleware`.
-   **Adjusted request handling to align with TurboAPI/Starlette.**
    -   Modified API endpoints (e.g., `/chat/completions`) to accept raw `starlette.requests.Request` objects.
    -   Implemented manual JSON body parsing (`await request.json()`) within endpoints.
    -   Added explicit instantiation and validation of `satya` models (like `ChatCompletionRequest` and nested `ChatMessage`) from parsed request bodies.
-   **Updated model serialization.**
    -   Replaced Pydantic's `.model_dump()` with `satya`-compatible serialization (initially `.to_dict()`, then a custom `recursive_to_dict` helper) for API responses and streaming chunks.
-   **Improved Error Handling.**
    -   Added `try...except` block in `create_chat_completion` endpoint with traceback printing for easier debugging.
-   **Refined Model Caching.**
    -   Updated `_create_text_model` to use a combined `model_id` and `adapter_path` as the cache key for more accurate model reuse. Added logging for model loading/caching events.

### Added

-   Helper function `recursive_to_dict` for robust serialization of nested `satya.Model` instances.

### Fixed

-   Resolved `AttributeError` related to accessing attributes (`.model`, `.stream`, `.role`) on request/message objects by implementing manual parsing and explicit model instantiation.
-   Corrected `ImportError` for `StreamingResponse` by importing from `starlette.responses`.
-   Fixed `AttributeError` for `add_middleware` by using the correct TurboAPI initialization pattern.

## [0.0.1] - 2025-04-19

### Added

-   Initial project setup. 
-   Forked from MLX-OMNI-ENGINE https://github.com/madroidmaq/mlx-omni-server by @madroidmaq