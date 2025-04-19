# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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