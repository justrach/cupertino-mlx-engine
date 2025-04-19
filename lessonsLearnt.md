# Lessons Learnt: Debugging Structured Output (`json_schema`)

This document summarizes the debugging process for enabling structured output using the `json_schema` feature, which involved resolving several `AttributeError` exceptions.

## The Problem

The goal was to support the OpenAI-compatible `response_format` parameter with `type: "json_schema"` to enforce model output according to a provided JSON schema.

## Debugging Journey & Fixes

1.  **Initial Error:** `AttributeError: 'dict' object has no attribute 'json_schema'`
    *   **Location:** `src/mlxengine/chat/mlx/mlx_model.py` (when accessing `request.response_format.json_schema`)
    *   **Diagnosis:** When the `ChatCompletionRequest` was instantiated in `src/mlxengine/chat/router.py` using `ChatCompletionRequest(**body)`, the `response_format` field, being a dictionary in the JSON request body, was not automatically converted into the expected `ResponseFormat` `satya.Model` instance. It remained a plain dictionary.
    *   **Attempt 1:** Implement explicit deserialization for the `response_format` field in `router.py`, similar to how `messages` were handled, by creating a `ResponseFormat` instance from the dictionary (`ResponseFormat(**raw_response_format)`).

2.  **Second Error:** `AttributeError: 'dict' object has no attribute 'schema_def'`
    *   **Location:** `src/mlxengine/chat/mlx/outlines_logits_processor.py` (when accessing `response_format.json_schema.schema_def`)
    *   **Diagnosis:** Two issues were identified:
        *   The `JsonSchemaFormat` model in `src/mlxengine/chat/schema.py` defined the schema field as `schema_def`.
        *   The incoming request (and the standard OpenAI API) uses the key `schema`.
        *   The code in `outlines_logits_processor.py` was trying to access `schema_def`.
    *   **Attempt 2:** Rename the field in the `JsonSchemaFormat` model from `schema_def` to `schema` to align with the standard API request format.

3.  **Third Error:** `AttributeError: 'dict' object has no attribute 'schema'`
    *   **Location:** `src/mlxengine/chat/mlx/outlines_logits_processor.py` (when accessing `response_format.json_schema.schema` after Attempt 2)
    *   **Diagnosis:** Although the outer `response_format` dictionary was correctly converted to a `ResponseFormat` object in Attempt 1, the *nested* `json_schema` dictionary *within* it was *still* not being automatically converted into a `JsonSchemaFormat` model instance by `satya`. Therefore, accessing `response_format.json_schema.schema` failed because `response_format.json_schema` was still a dictionary.
    *   **Attempt 3 (Final Fix):**
        *   Update `outlines_logits_processor.py` to access `.schema` (correcting the mismatch from Attempt 2).
        *   Enhance the explicit deserialization logic in `router.py` to be recursive. Before creating the `ResponseFormat` instance, check if its `json_schema` field is a dictionary. If it is, first convert *that nested dictionary* into a `JsonSchemaFormat` instance. Then, create the outer `ResponseFormat` instance using the data structure that now contains the correctly typed nested object.

## Key Takeaways

*   **Explicit Nested Deserialization:** When using libraries like `satya` or `Pydantic` with manually parsed request bodies (as necessitated by the switch to TurboAPI/Starlette), don't assume automatic recursive deserialization of nested model structures. It might be necessary to explicitly check and convert nested dictionaries into their corresponding model types before instantiating the parent model or passing the structure onwards.
*   **Verify Field Names:** Double-check that field names defined in models (`schema.py`) exactly match the keys expected in incoming data (JSON requests) and the attribute names accessed in the code that uses these models.
*   **Traceback Analysis:** Carefully read tracebacks to pinpoint the exact location and the state of variables (like `json_schema_obj` being a `dict` instead of `JsonSchemaFormat`) leading to the error. 