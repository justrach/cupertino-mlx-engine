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

# Lessons Learnt: Debugging Tool Usage & Serialization

This section covers issues encountered when implementing OpenAI-compatible tool usage, specifically around serialization and deserialization of nested structures when interacting with the `transformers` library's chat templates.

## The Problem

After adding explicit deserialization for nested `Tool`, `Function`, and `FunctionParameters` models in `router.py` (similar to the `response_format` fix), new errors emerged during the prompt encoding phase within `chat_tokenizer.py` which uses `transformers.apply_chat_template`.

## Debugging Journey & Fixes

1.  **Initial Error:** `AttributeError: 'dict' object has no attribute 'dict'`
    *   **Location:** `src/mlxengine/chat/mlx/tools/chat_tokenizer.py` (when trying to call `.dict()` on a tool dictionary)
    *   **Diagnosis:** Similar to the `response_format` issue, the `tools` list passed from `router.py` to `mlx_model.py` and then to `chat_tokenizer.py` contained dictionaries, not `Tool` model instances, despite the deserialization attempt in `router.py`. The deserialization in `router.py` needed to happen *before* creating the `ChatCompletionRequest` instance.
    *   **Attempt 1:** Refactor the `router.py` logic to perform explicit, recursive deserialization of `messages`, `response_format`, and `tools` on the raw request body *before* initializing `ChatCompletionRequest`.

2.  **Second Error:** `TypeError: Object of type Function is not JSON serializable`
    *   **Location:** Inside `transformers.apply_chat_template` called from `src/mlxengine/chat/mlx/tools/chat_tokenizer.py`.
    *   **Diagnosis:** While the `ChatCompletionRequest` now held correctly typed `Tool` objects, the `chat_tokenizer.py` code was preparing the `schema_tools` list for `apply_chat_template` by only calling `.dict()` at the top level (`Tool`). This left nested objects like `Function` and `FunctionParameters` as model instances, which `json.dumps` (used internally by the Jinja template's `tojson` filter) cannot handle.
    *   **Attempt 2:** Introduce a `recursive_to_dict` helper function (moved to `utils/serialization.py`) and use it in `chat_tokenizer.py` to fully convert the `Tool` objects (including nested `Function` and `FunctionParameters`) into plain dictionaries before passing them to `apply_chat_template`.

3.  **Third Error:** `UndefinedError: 'dict object' has no attribute 'content'`
    *   **Location:** Inside the Jinja template rendering within `transformers.apply_chat_template`.
    *   **Diagnosis:** The chat template expects every message dictionary in the conversation list to have a `content` key. When processing assistant messages that only contain `tool_calls` (where `content` is naturally `None`), the `chat_tokenizer.py` code was filtering out the `content` key because its value was `None`. The Jinja template then failed when trying to access `message.content`.
    *   **Attempt 3:** Modify `chat_tokenizer.py` to explicitly keep the `content` key in the message dictionary even if its value is `None`.

4.  **Fourth Error:** `UndefinedError: 'None' has no attribute 'split'`
    *   **Location:** Inside the Jinja template rendering.
    *   **Diagnosis:** Although the `content` key was now present (with a value of `None`), the Jinja template was attempting string operations (like `.split()`) on it, which fails for `None`.
    *   **Attempt 4 (Final Fix):** Modify `chat_tokenizer.py` again. Instead of setting `content` to `None` when it's missing, set it to an empty string (`""`). This satisfies the template's expectation of a string-like value without adding unwanted text.

## Key Takeaways

*   **Deserialization Point:** Ensure nested model deserialization happens *before* the parent model is instantiated if the parent model relies on the nested types during its own processing or passes them downstream.
*   **Serialization Boundaries:** When passing complex objects (like nested models) to external libraries or components (like Jinja templates via `transformers`), ensure they are fully serialized into primitive types (dicts, lists, strings, numbers, booleans, None) if the receiving component expects plain data.
*   **Template Expectations:** Be mindful of the exact data structure and type expectations of templates (like Jinja used in `apply_chat_template`). They might require keys to always be present or expect specific types (e.g., strings) even if logically `None` might seem appropriate.
*   **Helper Functions:** Utility functions for recursive operations (like `recursive_to_dict`) can simplify serialization/deserialization logic across different modules. 